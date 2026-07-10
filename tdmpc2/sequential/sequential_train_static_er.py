"""Sequential TD-MPC2 + Experience Replay — naive static-K MoE variant.

Same agent architecture as `sequential_train.py` (DreamerV3 split:
shared `PrismBackbone` + per-task `TaskModules`, static full-K MoE) plus
Experience Replay across tasks:

  - During each task: every completed episode is fed to an online reservoir
    (Vitter's Algorithm R) sized to
        K = (er_buffer_ratio · cfg.steps) / cfg.episode_length
    episodes. Memory is O(K) regardless of task length.
  - At task end: the reservoir is dumped to `{task_logdir}/er_episodes.pt`.
  - At the start of task t > 0: every prior task's `er_episodes.pt` is loaded
    and inserted into the new task's buffer BEFORE any new-task data is
    collected. Subsequent agent updates draw from the mixture.

Thin wrapper around `sequential_train.run_continual_sequential` with
`progressive=False` and ER enabled.

Example:
    python sequential/sequential_train_static_er.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --num-experts 8 \\
        --er-buffer-ratio 0.025 \\
        --logdir ../logdir/seq_prism_naive_er/run0 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_PrismNaive_ER
"""

import os
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = '1'
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['XDG_RUNTIME_DIR'] = '/tmp'
os.environ['EGL_LOG_LEVEL'] = 'fatal'

import warnings
warnings.filterwarnings('ignore', message='Constant.*may be too high')
warnings.filterwarnings('ignore', message='.*Please upgrade to Gymnasium.*')
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
warnings.filterwarnings('ignore')

import argparse
import sys
from pathlib import Path

import torch

# Add tdmpc2/ to sys.path so we can import the shared sequential driver.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from sequential_train import run_continual_sequential


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def main(args):
    run_continual_sequential(
        args,
        progressive=False,
        er_buffer_ratio=args.er_buffer_ratio,
        er_seed=args.er_seed,
        per_task_tau=args.per_task_tau,
        reward_anchor_coef=args.reward_anchor_coef,
        termination_anchor_coef=args.termination_anchor_coef,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Naive sequential TD-MPC2 + Experience Replay '
                    '(DreamerV3 split, static full-K PRISM-MoE).',
    )

    # Task list
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--task-steps', nargs='+', type=int, required=True)

    # Logging / checkpoint
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--logger', type=str, default='wandb',
                   choices=['wandb', 'none'])
    p.add_argument('--wandb-entity', type=str, default='haoyu-a2i')
    p.add_argument('--wandb-project', type=str,
                   default='Metaworld_Tdmpc2_Sequential_PrismNaive_ER')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None,
                   help='Wandb group; sweep launcher passes the taskset name.')
    p.add_argument('--wandb-tags', nargs='*', default=None,
                   help='Wandb tags; sweep launcher fills with method/taskset/etc.')
    p.add_argument('--exp-name', type=str, default='seq_naive_er_tdmpc2')
    p.add_argument('--from-checkpoint', type=str, default=None,
                   help='Directory containing backbone.pt (+ optional '
                        'task_modules.pt) to seed task 0.')

    p.add_argument('--seed', type=int, default=1)

    # TDMPC2 hyperparams
    p.add_argument('--model-size', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--task-buffer-sizes', nargs='+', type=int, default=None)
    p.add_argument('--horizon', type=int, default=3)
    p.add_argument('--mpc', dest='mpc', action='store_true')
    p.add_argument('--no-mpc', dest='mpc', action='store_false')
    p.set_defaults(mpc=True)
    p.add_argument('--eval-freq', type=int, default=10000)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--save-freq', type=int, default=0)
    p.add_argument('--save-video', dest='save_video', action='store_true')
    p.add_argument('--no-save-video', dest='save_video', action='store_false')
    p.set_defaults(save_video=True)
    p.add_argument('--skip-pretrain', action='store_true')

    # Env knobs
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--max-episode-steps', type=int, default=250)
    p.add_argument('--action-repeat', type=int, default=2)
    p.add_argument('--cameras', nargs='+',
                   default=['topview', 'front', 'gripperPOV'])
    p.add_argument('--episodic', dest='episodic', action='store_true')
    p.add_argument('--no-episodic', dest='episodic', action='store_false')
    p.set_defaults(episodic=True)

    # Replay-buffer persistence
    p.add_argument('--save-episodes', dest='save_episodes',
                   action='store_true',
                   help='Persist completed episodes to {task_logdir}/train_eps/ '
                        'and replay them back into the buffer on resume. '
                        'Independent of the cross-task ER dump (er_episodes.pt).')
    p.add_argument('--no-save-episodes', dest='save_episodes',
                   action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5)
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])

    # PRISM-WM static-K MoE
    p.add_argument('--num-experts', type=int, default=8,
                   help='Number of experts in the (static, full-K) dynamics '
                        'MoE. All experts are always active and trainable.')
    p.add_argument('--use-orthogonal', dest='use_orthogonal',
                   action='store_true',
                   help='Enable Gram-Schmidt orthogonalization of the expert '
                        'feature stack inside the dynamics MoE.')
    p.add_argument('--no-use-orthogonal', dest='use_orthogonal',
                   action='store_false')
    p.set_defaults(use_orthogonal=True)

    # ER-specific
    p.add_argument('--er-buffer-ratio', type=float, default=0.025,
                   help='Fraction of each prev task\'s cfg.steps to keep as '
                        'ER transitions (online reservoir). 0 disables ER. '
                        'Default: 0.025.')
    p.add_argument('--er-seed', type=int, default=42,
                   help='RNG seed for the per-task online reservoir '
                        '(offset by task_idx). Default: 42.')

    # Per-task τ snapshot (opt-in). Matches the progmoe-ER entry so the
    # naive baseline can be a single-variable (freeze/mask only) ablation.
    p.add_argument('--per-task-tau', dest='per_task_tau',
                   action='store_true',
                   help='At cross-task eval, restore each prior task\'s '
                        'converged τ from its backbone.pt snapshot.')
    p.add_argument('--no-per-task-tau', dest='per_task_tau',
                   action='store_false')
    p.set_defaults(per_task_tau=False)

    # Supervised anchors through frozen teacher heads (off by default →
    # consistency-only aux updates, matching the original behavior).
    p.add_argument('--reward-anchor-coef', type=float, default=0.0,
                   help='Weight of the soft-CE reward anchor in the aux '
                        'update (frozen teacher reward head applied to the '
                        'live rolled-out latent; gradient into encoder + '
                        'dynamics only). Default: 0.0 (disabled).')
    p.add_argument('--termination-anchor-coef', type=float, default=0.0,
                   help='Weight of the BCE termination anchor in the aux '
                        'update (only when cfg.episodic). Default: 0.0 '
                        '(disabled).')

    # Offline-pretrained encoder (see tdmpc2/pretrain/). Default: off.
    p.add_argument('--pretrained-encoder', type=str, default=None,
                   help='Path to an encoder checkpoint from '
                        'pretrain/pretrain_encoder.py. Loaded into the '
                        'backbone encoder; None -> fresh random encoder.')
    p.add_argument('--freeze-encoder', dest='freeze_encoder',
                   action='store_true',
                   help='Freeze the (pretrained) encoder for ALL tasks '
                        '(requires_grad=False; excluded from the optimizer).')
    p.add_argument('--no-freeze-encoder', dest='freeze_encoder',
                   action='store_false')
    p.set_defaults(freeze_encoder=False)

    # rgb-branch encoder type. 'dino' = frozen DINOv2 ViT-S/14 backbone +
    # trainable pooling/projection head (mutually exclusive with
    # --pretrained-encoder). dino_* knobs come from config.yaml defaults.
    p.add_argument('--encoder-type', type=str, default='default',
                   choices=['default', 'dino'],
                   help="rgb encoder: 'default' CNN or frozen 'dino' "
                        '(DINOv2 + trainable head; forces compile=false).')

    main(p.parse_args())
