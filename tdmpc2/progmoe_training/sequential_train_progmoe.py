"""Sequential TD-MPC2 + Progressive-MoE continual learning.

At every task boundary:
  - Adds `--experts-per-task` new experts to BOTH the dynamics and
    reward MoE blocks.
  - Freezes every expert that existed before the new task. They still
    produce outputs during forward and are still routable by the gate —
    they just don't receive gradient updates.
  - Switches the task one-hot fed into the gate/experts via
    `agent.set_current_task(t)`.
  - Continues training with a fresh empty replay buffer, agent loaded
    from the previous task's final checkpoint.

Cross-task evaluation: every `eval_freq` steps and at task end, the
current agent is evaluated on ALL tasks seen so far; the router's task
one-hot is switched to the eval task before each eval and restored to
the training task afterwards.

Differences from `sequential_train.py`:
  - Builds `ContinualTDMPC2` (overridden world model with GrowingMoE).
  - Forces `cfg.use_moe=True`, `cfg.compile=False`,
    `cfg.num_tasks=len(--tasks)`.
  - For task t (t >= 1) the agent is built with K = t * K_per_task to
    match the saved checkpoint, the checkpoint is loaded, then
    `agent.expand_for_new_task(K_per_task)` grows both MoE blocks by
    K_per_task and freezes everything that came before.

Does NOT touch the baseline `sequential_train.py` — that path is the
no-MoE (or naive MoE) sequential baseline.

Example:
    python progmoe_training/sequential_train_progmoe.py \
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \
        --task-steps 200000 500000 \
        --logdir ../logdir/seq_progmoe/run0 \
        --model-size 19 \
        --experts-per-task 4 \
        --wandb-project Metaworld_Tdmpc2_ProgMoE
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
import gc
import shutil
import sys
from pathlib import Path

import torch
from termcolor import colored

# Add tdmpc2/ to sys.path so we can import common/, sequential_train, etc.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.buffer import Buffer
from common.seed import set_seed
from envs import make_env

# Reuse helpers from the baseline sequential script.
from sequential_train import (
    build_task_cfg,
    save_progress,
    load_progress,
    save_wandb_run_id,
    SequentialLogger,
    eval_on_env,
    train_one_task,
)

from progmoe_training.continual_tdmpc2 import ContinualTDMPC2


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# Main
# ==========================================================================

def main(args):
    assert torch.cuda.is_available(), 'TD-MPC2 requires CUDA.'

    num_tasks = len(args.tasks)
    if len(args.task_steps) != num_tasks:
        raise ValueError('--task-steps must have one value per --tasks entry.')
    if args.task_buffer_sizes is not None:
        if len(args.task_buffer_sizes) != num_tasks:
            raise ValueError(
                '--task-buffer-sizes must have one value per --tasks entry.')
        task_buffer_sizes = list(args.task_buffer_sizes)
    else:
        task_buffer_sizes = [args.buffer_size] * num_tasks

    K_per_task = int(args.experts_per_task)
    if K_per_task <= 0:
        raise ValueError('--experts-per-task must be > 0.')

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ---- shared overrides (identical across tasks except num_experts) -----
    shared = dict(
        model_size=args.model_size,
        batch_size=args.batch_size,
        horizon=args.horizon,
        mpc=args.mpc,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        image_size=args.image_size,
        max_episode_steps=args.max_episode_steps,
        action_repeat=args.action_repeat,
        cameras=list(args.cameras),
        episodic=args.episodic,
        enable_wandb=False,  # we drive a single wandb run across tasks
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        # Compile must be off — MoE.experts grows mid-run.
        compile=False,
        save_episodes=args.save_episodes,
        episode_dir=None,
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
        # PRISM-MoE
        use_moe=True,
        moe_residual_dynamics=args.moe_residual_dynamics,
        num_tasks=num_tasks,
        # num_experts is set per-task below — must match prev checkpoint size.
    )

    # ---- Per-task configs --------------------------------------------------
    # Task 0:  K_at_build = K_per_task (fresh agent, no expansion).
    # Task t:  K_at_build = t * K_per_task (matches prev task's saved ckpt;
    #          we expand by K_per_task AFTER loading).
    task_cfgs = []
    for i in range(num_tasks):
        task_logdir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
        K_at_build = K_per_task if i == 0 else i * K_per_task
        per_task = {
            **shared,
            'buffer_size': int(task_buffer_sizes[i]),
            'num_experts': K_at_build,
        }
        cfg = build_task_cfg(
            args.tasks[i], args.task_steps[i], task_logdir, per_task, args.seed,
        )
        task_cfgs.append(cfg)

    set_seed(args.seed)

    # ---- Resume detection --------------------------------------------------
    resume_from = 0
    global_step = 0
    prev_ckpt = None
    progress = load_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            entry = progress.get('tasks', {}).get(str(i))
            if entry and entry.get('completed'):
                resume_from = i + 1
                global_step = int(entry['global_step_at_end'])
                ck = (base_logdir / f'task{i + 1}_{args.tasks[i]}'
                      / 'models' / 'final.pt')
                if ck.exists():
                    prev_ckpt = ck
                print(colored(
                    f'>>> RESUME: task {i + 1} ({args.tasks[i]}) '
                    f'already done (@g={global_step}).', 'cyan'))
            else:
                break
    if resume_from >= num_tasks:
        print(colored('>>> All tasks already completed. Nothing to do.',
                      'green', attrs=['bold']))
        return

    # ---- One wandb run across all tasks ------------------------------------
    stored_run_id = (progress or {}).get('wandb_run_id') if progress else None
    effective_run_id = args.wandb_run_id or stored_run_id

    wandb_run = None
    if args.logger == 'wandb':
        import wandb
        run_name = args.wandb_run_name or (
            f'progmoe_tdmpc2_{"-".join(args.tasks)}_s{args.seed}'
        )
        if effective_run_id is not None:
            print(colored(
                f'>>> RESUME: attaching to existing wandb run id='
                f'{effective_run_id}', 'cyan',
            ))
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            id=effective_run_id,
            resume='allow' if effective_run_id else None,
            dir=str(base_logdir),
            config={
                'tasks': args.tasks,
                'task_steps': args.task_steps,
                'seed': args.seed,
                'experts_per_task': K_per_task,
                'method': 'progmoe',
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'progmoe', 'prism'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> PROGRESSIVE-MoE TD-MPC2 TRAINING', 'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        K_at_end = (i + 1) * K_per_task
        print(f'  Task {i + 1}: {t}  steps={s}  '
              f'(K at end = {K_at_end}, +{K_per_task} fresh){mark}')
    print(f'  Logdir: {base_logdir}')
    print(f'  Seed: {args.seed}')
    print(f'  Wandb project: {args.wandb_project}')
    print(colored(
        f'  PRISM-WM (progressive MoE): K_per_task={K_per_task},  '
        f'residual_dynamics={args.moe_residual_dynamics},  '
        f'compile=false  (no Gram-Schmidt)',
        'cyan'))
    print('=' * 64)

    # ---- Eval env cache ----------------------------------------------------
    eval_env_cache = {}

    def _get_eval_env(task_idx):
        if task_idx in eval_env_cache:
            return eval_env_cache[task_idx]
        env = make_env(task_cfgs[task_idx])
        eval_env_cache[task_idx] = env
        return env

    # ---- Task loop ---------------------------------------------------------
    for task_idx in range(num_tasks):
        if task_idx < resume_from:
            continue

        cfg = task_cfgs[task_idx]
        task_name = args.tasks[task_idx]
        task_logdir = Path(cfg.work_dir)
        task_logdir.mkdir(parents=True, exist_ok=True)

        K_at_build = K_per_task if task_idx == 0 else task_idx * K_per_task
        K_at_end = (task_idx + 1) * K_per_task

        print('=' * 64)
        print(colored(
            f'>>> TASK {task_idx + 1}/{num_tasks}: {task_name}',
            'cyan', attrs=['bold'],
        ))
        print(f'    K at build (= prev ckpt size): {K_at_build}')
        if task_idx > 0:
            print(f'    will expand by {K_per_task} -> total {K_at_end}, '
                  f'freezing first {K_at_build} experts')
        print(f'    internal steps: {cfg.steps}')
        print(f'    global step start: {global_step}')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        # Buffer isolation: clear completed previous tasks' replay dirs.
        for j in range(task_idx):
            prev_done = (
                progress is not None
                and progress.get('tasks', {}).get(str(j), {}).get('completed')
            )
            if not prev_done:
                continue
            prev_eps = (
                base_logdir / f'task{j + 1}_{args.tasks[j]}' / 'train_eps'
            )
            if prev_eps.is_dir():
                print(colored(
                    f'>>> Removing previous-task replay dir: {prev_eps}',
                    'yellow'))
                shutil.rmtree(prev_eps, ignore_errors=True)

        # Fresh env + buffer for this task.
        train_env = make_env(cfg)
        buffer = Buffer(cfg)

        task_episode_dir = task_logdir / 'train_eps'
        task_buffer_cap = int(min(cfg.buffer_size, cfg.steps))
        if args.save_episodes and task_episode_dir.is_dir():
            stats = buffer.load_from_directory(
                task_episode_dir, max_total_steps=task_buffer_cap)
            if stats['episodes_restored'] > 0:
                print(colored(
                    f'>>> Replayed {stats["transitions_restored"]:,} '
                    f'transitions from {stats["episodes_restored"]} '
                    f'episodes (cap {task_buffer_cap:,}).',
                    'green'))
                if stats['kept_files']:
                    dropped = Buffer.erase_over_episode_files(
                        task_episode_dir, stats['kept_files'])
                    if dropped:
                        print(colored(
                            f'>>> Pruned {dropped} stale episode files.',
                            'yellow'))

        logger = SequentialLogger(
            cfg, task_idx, num_tasks, task_name, wandb_run,
            save_video=args.save_video,
        )

        # ---- Build agent, load prev ckpt (if any), expand --------------
        agent = ContinualTDMPC2(cfg)

        if task_idx > 0 and prev_ckpt is not None and prev_ckpt.exists():
            print(colored(
                f'>>> Loading previous-task checkpoint: {prev_ckpt}', 'cyan'))
            agent.load(prev_ckpt)
            print(colored(
                f'>>> Expanding MoE: +{K_per_task} experts per block, '
                f'freezing first {K_at_build}',
                'cyan'))
            agent.expand_for_new_task(K_per_task)
        elif task_idx == 0 and args.from_checkpoint is not None:
            ck = Path(args.from_checkpoint)
            if not ck.exists():
                raise FileNotFoundError(ck)
            print(colored(f'>>> Loading external checkpoint: {ck}', 'cyan'))
            agent.load(ck)

        # Set router/expert task one-hot to the current training task.
        agent.set_current_task(task_idx)

        # ---- Cross-task eval closure (switches one-hot per eval task) --
        def eval_fn(agent, local_step, gstep,
                    _task_idx=task_idx, _logger=logger, _cfg=cfg):
            for j in range(_task_idx + 1):
                is_current = (j == _task_idx)
                agent.set_current_task(j)
                env_j = _get_eval_env(j)
                video_rec = _logger.video if (is_current and _logger.video) else None
                if video_rec is not None:
                    video_rec.set_prefix(
                        f'eval/task{j + 1}_{args.tasks[j]}/videos/eval_video'
                    )
                metrics = eval_on_env(
                    agent, env_j, _cfg.eval_episodes,
                    video_recorder=video_rec,
                    video_step=gstep if video_rec else None,
                )
                metrics['step'] = local_step
                _logger.log_eval(j, args.tasks[j], metrics, gstep)
            # Restore training task one-hot.
            agent.set_current_task(_task_idx)

        # ---- Train this task -------------------------------------------
        train_one_task(
            cfg, agent, train_env, buffer, logger,
            global_step_start=global_step,
            eval_fn=eval_fn,
            skip_pretrain=args.skip_pretrain or (task_idx > 0),
            task_idx=task_idx,
            num_tasks=num_tasks,
            task_name=task_name,
            episode_dir=task_episode_dir,
            save_episodes=args.save_episodes,
            buffer_cap=task_buffer_cap,
            prune_every_n_eps=args.prune_every_n_episodes,
        )

        # ---- Final cross-task eval + save ------------------------------
        eval_fn(agent, cfg.steps, global_step + cfg.steps)
        logger.save_agent(agent, identifier='final')
        prev_ckpt = logger.model_dir / 'final.pt'

        global_step += cfg.steps
        save_progress(
            base_logdir, task_idx, task_name=task_name,
            completed=True, global_step_at_end=global_step,
        )

        try:
            train_env.close()
        except Exception:
            pass
        del train_env, buffer, agent
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(colored(
            f'>>> TASK {task_idx + 1} ({task_name}) done @g={global_step}.',
            'green', attrs=['bold'],
        ))
        print()

    for e in eval_env_cache.values():
        try:
            e.close()
        except Exception:
            pass

    if wandb_run is not None:
        wandb_run.finish()

    print('=' * 64)
    print(colored('>>> ALL TASKS COMPLETED.', 'green', attrs=['bold']))
    print(f'>>> Final global step: {global_step}')
    print('=' * 64)


# ==========================================================================
# Entry point
# ==========================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Sequential TD-MPC2 + progressive-MoE continual learning.',
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
                   default='Metaworld_Tdmpc2_ProgMoE')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--exp-name', type=str, default='progmoe_tdmpc2')
    p.add_argument('--from-checkpoint', type=str, default=None)

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
    p.add_argument('--save-episodes', dest='save_episodes', action='store_true')
    p.add_argument('--no-save-episodes', dest='save_episodes',
                   action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5)
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'])

    # Progressive-MoE specifics
    p.add_argument('--experts-per-task', type=int, default=4,
                   help='Number of NEW experts appended to both MoE blocks '
                        'at each task boundary. Total experts at end of '
                        'task t = (t+1) * experts-per-task.')
    p.add_argument('--moe-residual-dynamics', dest='moe_residual_dynamics',
                   action='store_true',
                   help='If set, dynamics MoE applies residual + post-residual '
                        'SimNorm. Empirically the non-residual setting works '
                        'better with tdmpc2 SimNorm latents — kept false by '
                        'default.')
    p.add_argument('--no-moe-residual-dynamics',
                   dest='moe_residual_dynamics', action='store_false')
    p.set_defaults(moe_residual_dynamics=False)

    main(p.parse_args())
