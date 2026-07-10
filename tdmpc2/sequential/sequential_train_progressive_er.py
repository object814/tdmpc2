"""Sequential TD-MPC2 + Experience Replay — progressive masked MoE,
*merged-buffer* (DreamerV3-style multitask) variant.

Progressive masked-MoE TD-MPC2 with cross-task Experience Replay (DreamerV3 split:
shared `PrismBackbone` encoder + `MaskedMoEBlock` dynamics with
total_K = num_tasks * K_per_task experts; a new K-slot opened per task and
prior experts FROZEN; per-task `TaskModules` for reward / Q / π / termination,
saved + reloaded at eval). The difference is *how* old-task data is used.

Instead of the separate routing-aware aux step (consistency-only, or anchored),
this variant treats each task as a multitask problem over {seen tasks}, exactly
like DreamerV3-ER:

  - Prior tasks' reservoir-sampled ER is kept in anchored, never-FIFO'd buffers.
  - Each training step samples ONE *within-batch* task-mix: rows are drawn from
    the current-task buffer + the ER buffers, in proportion to each source's
    transition count (early in a task → mostly old ER; as the current buffer
    fills → mostly current — the DreamerV3 merged-replay dynamic). A single
    FULL-loss update runs on that mixed batch.
  - Routing is PER SAMPLE: old-task rows flow through their FROZEN experts
    (task-j one-hot), current-task rows through the live experts. The current
    per-task heads see the whole mix (DreamerV3 trains the current heads on the
    task-mixed batch; harmless to old eval, which uses the saved old heads).
  - The encoder + router + new experts are thereby held compatible with prior
    tasks by the old-task rows in every batch — the anti-forgetting mechanism.

Evaluation is UNCHANGED from progmoe: task j is evaluated on its own expert
window [0, (j+1)·K) with task-j's one-hot and task-j's SAVED heads.

Thin wrapper around `sequential_train.run_continual_sequential` with
`progressive=True, merged=True`.

Example:
    python sequential/sequential_train_progressive_er.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --K-per-task 3 --use-orthogonal \\
        --er-buffer-ratio 0.05 \\
        --logdir ../logdir/seq_progmoe_merged_er/run0 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_ProgMoE_Merged_ER
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

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from sequential_train import run_continual_sequential


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def main(args):
    run_continual_sequential(
        args,
        progressive=True,
        er_buffer_ratio=args.er_buffer_ratio,
        er_seed=args.er_seed,
        per_task_tau=args.per_task_tau,
        merged=True,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Sequential TD-MPC2 + PRISM-WM continual learning, '
                    'merged-buffer (DreamerV3-style multitask) ER '
                    '(progressive masked MoE).',
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
                   default='Metaworld_Tdmpc2_Sequential_ProgMoE_Merged_ER')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--wandb-group', type=str, default=None,
                   help='Wandb group; sweep launcher passes the taskset name.')
    p.add_argument('--wandb-tags', nargs='*', default=None,
                   help='Wandb tags; sweep launcher fills with method/taskset/etc.')
    p.add_argument('--exp-name', type=str, default='progmoe_merged_er_tdmpc2')
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
                        'and replay them back into the buffer on resume.')
    p.add_argument('--no-save-episodes', dest='save_episodes',
                   action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5)
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])

    # Progressive-MoE knob
    p.add_argument('--K-per-task', dest='K_per_task',
                   type=int, default=3,
                   help='Number of expert slots opened per task. total_K = '
                        'num_tasks * K_per_task; at task t experts '
                        '[t·K, (t+1)·K) train and [0, t·K) are frozen.')

    # Gram-Schmidt orthogonalization
    p.add_argument('--use-orthogonal', dest='use_orthogonal',
                   action='store_true',
                   help='Enable Gram-Schmidt orthogonalization of the expert '
                        'feature stack inside the dynamics MoE.')
    p.add_argument('--no-use-orthogonal', dest='use_orthogonal',
                   action='store_false')
    p.set_defaults(use_orthogonal=True)

    # ER
    p.add_argument('--er-buffer-ratio', type=float, default=0.05,
                   help='Fraction of each prev task\'s cfg.steps kept as ER '
                        'transitions (online reservoir). Used as the merged '
                        'replay source. 0 disables ER (degenerate: no '
                        'anti-forgetting). Default: 0.05.')
    p.add_argument('--er-seed', type=int, default=42)

    # Per-task τ snapshot (eval-time; unchanged from progmoe).
    p.add_argument('--per-task-tau', dest='per_task_tau',
                   action='store_true',
                   help='At cross-task eval, restore each prior task\'s '
                        'converged τ from its backbone.pt snapshot.')
    p.add_argument('--no-per-task-tau', dest='per_task_tau',
                   action='store_false')
    p.set_defaults(per_task_tau=True)

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
