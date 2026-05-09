"""Sequential TD-MPC2 training with Experience Replay (ER).

Mirrors `sequential_train.py` but adds an ER buffer drawn from previous tasks:

  1. During each task's training, every completed episode is appended to a
     Python-side log of episode TensorDicts.
  2. At the end of a task, the log is reservoir-sampled down to a budget
     (in transitions, computed as `er_buffer_ratio * cfg.steps`) and saved
     to disk at `<task_logdir>/er_episodes.pt`.
  3. Before training a subsequent task, episodes from every previous task's
     dump are loaded and inserted into the new (otherwise empty) replay
     buffer via `Buffer.add` BEFORE training begins. The buffer therefore
     contains a mix of old-task and (eventually) new-task transitions.

Notes / caveats:
  - TD-MPC2 is monolithic: the same reward and Q heads see ER transitions
    from all tasks (with their original task-1 reward/done labels). The
    consistency loss benefits from ER, but reward / Q losses are pulled
    toward fitting a mixture of old + new reward functions. This is the
    fundamental ER limitation we discussed for monolithic TD-MPC2.
  - We rely on the same checkpoint loading + cross-task eval as the naive
    sequential script.

Resume support is identical to `sequential_train.py`. Additionally, the ER
dump from each completed task lives at `<task_logdir>/er_episodes.pt` and
is reused by all later tasks.

Example:
    python sequential_train_er.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2_er/run0 \\
        --er-buffer-ratio 0.025 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_ER
"""

import os

os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["XDG_RUNTIME_DIR"] = "/tmp"
os.environ["EGL_LOG_LEVEL"] = "fatal"

import warnings
warnings.filterwarnings("ignore", message="Constant.*may be too high")
warnings.filterwarnings("ignore", message=".*Please upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
warnings.filterwarnings('ignore')

import argparse
import gc
import json
import random
import shutil
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from termcolor import colored
from tqdm.auto import tqdm

# Make `tdmpc2/` importable so we can reuse logger / progress helpers.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2

# Reuse helpers from the naive sequential script — keeps the two scripts in sync.
from sequential_train import (
    build_task_cfg,
    save_progress,
    load_progress,
    save_wandb_run_id,
    SequentialLogger,
    eval_on_env,
    _to_td,
)


CONFIG_PATH = _PARENT / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# ER episode dump / load
# ==========================================================================

def _episode_transitions(td):
    """Number of transitions stored in a single-episode TensorDict.

    `_to_td` produces one td per env step (including the initial reset row),
    and `train_one_task` calls `torch.cat(tds)` before passing to the buffer.
    The first row is the reset (no real transition), so the transition count
    is `len(td) - 1` — same convention as the Dreamer ER reservoir code.
    """
    return max(int(td.batch_size[0]) - 1, 0)


def reservoir_sample(episodes, budget, seed=0):
    """Reservoir-sample episodes until the transition budget is met.

    Args:
        episodes: list of single-episode TensorDicts (one entry per episode).
        budget:   max total transitions to keep across the sampled episodes.
        seed:     RNG seed for the shuffle (reproducible across resumes).

    Returns:
        A new list (subset of `episodes`) whose total transition count is
        <= budget. Empty list if budget <= 0 or no episodes.
    """
    if budget <= 0 or not episodes:
        return []
    rng = random.Random(seed)
    order = list(range(len(episodes)))
    rng.shuffle(order)
    out = []
    total = 0
    for idx in order:
        td = episodes[idx]
        n = _episode_transitions(td)
        if n <= 0:
            continue
        if total + n > budget and out:
            # We've already filled the budget; stop here rather than partially
            # truncating the episode (TD-MPC2 sampling needs full episodes).
            break
        out.append(td)
        total += n
        if total >= budget:
            break
    return out


def save_er_episodes(episodes, path, budget, seed=0):
    """Reservoir-sample `episodes` to `budget` transitions and save to disk."""
    sampled = reservoir_sample(episodes, budget, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Move to CPU before saving to avoid CUDA-device mismatches on resume.
    cpu_eps = [td.detach().cpu() for td in sampled]
    torch.save(cpu_eps, str(path))
    n_trans = sum(_episode_transitions(td) for td in cpu_eps)
    print(colored(
        f'>>> ER: saved {len(cpu_eps)} episodes ({n_trans} transitions) -> {path}',
        'cyan',
    ))
    return len(cpu_eps), n_trans


def load_er_episodes(path):
    """Load a list of single-episode TensorDicts from disk; [] if missing."""
    path = Path(path)
    if not path.exists():
        return []
    eps = torch.load(str(path), map_location='cpu', weights_only=False)
    return eps


# ==========================================================================
# Per-task online training (mirrors `sequential_train.train_one_task` but
# additionally captures every completed episode for ER dumping).
# ==========================================================================

def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   episode_log,
                   task_idx=0, num_tasks=1, task_name='',
                   episode_dir=None, save_episodes=True,
                   buffer_cap=None, prune_every_n_eps=5):
    """Run TD-MPC2's online training loop and record every completed episode.

    Behaves identically to `sequential_train.train_one_task`, but appends
    each completed-episode TensorDict to `episode_log` (in CPU memory)
    AFTER it has been added to the replay buffer. The caller then
    reservoir-samples `episode_log` for the on-disk ER dump at task end.

    `episode_dir` (when ``save_episodes=True``) is where each completed
    episode is persisted to disk for full-state resume; FIFO-pruned to
    ``buffer_cap`` transitions. This is independent of the cross-task ER
    dump (`er_episodes.pt`) — both can coexist.
    """
    logger.global_step_fn = lambda s: global_step_start + s
    if save_episodes and episode_dir is not None:
        Path(episode_dir).mkdir(parents=True, exist_ok=True)
    if buffer_cap is None:
        buffer_cap = int(min(cfg.buffer_size, cfg.steps))

    pbar = tqdm(
        total=cfg.steps,
        desc=f'T{task_idx + 1}/{num_tasks} {task_name}',
        unit='step',
        dynamic_ncols=True,
    )

    local_step = 0
    ep_idx = 0
    start_time = time()
    train_metrics = {}
    done = True
    eval_next = False
    tds = []
    obs = None
    info = None

    def _common():
        elapsed = time() - start_time
        return dict(
            step=local_step,
            global_step=global_step_start + local_step,
            episode=ep_idx,
            elapsed_time=elapsed,
            steps_per_second=local_step / max(elapsed, 1e-9),
        )

    try:
        while local_step <= cfg.steps:
            if local_step % cfg.eval_freq == 0:
                eval_next = True

            if (cfg.save_freq > 0 and local_step > 0
                    and local_step % cfg.save_freq == 0):
                logger.save_agent(agent, identifier=local_step)

            if done:
                if eval_next:
                    eval_fn(agent, local_step, global_step_start + local_step)
                    eval_next = False

                if local_step > 0:
                    if info['terminated'] and not cfg.episodic:
                        raise ValueError(
                            'Termination detected but cfg.episodic=false. '
                            'Set episodic=true to support episodic tasks.'
                        )
                    train_metrics.update(
                        episode_reward=torch.tensor(
                            [td['reward'] for td in tds[1:]]).sum(),
                        episode_success=info['success'],
                        episode_length=len(tds),
                        episode_terminated=info['terminated'],
                    )
                    train_metrics.update(_common())
                    logger.log_train(train_metrics)
                    ep_td = torch.cat(tds)
                    ep_idx = buffer.add(ep_td)
                    # ER capture: keep a CPU copy of every completed episode.
                    # `tds` are already on CPU (see _to_td), so this is cheap.
                    episode_log.append(ep_td.detach().cpu())
                    # On-disk persistence for full-state resume.
                    if save_episodes and episode_dir is not None:
                        try:
                            Buffer.save_episode(ep_td, episode_dir, ep_idx)
                            if ep_idx % prune_every_n_eps == 0:
                                Buffer.prune_episode_dir_to_cap(episode_dir, buffer_cap)
                        except Exception as e:
                            print(colored(f'[seq_train_er] save_episode failed: {e}', 'red'))

                obs = env.reset()
                tds = [_to_td(env, obs)]

            # Experience
            if local_step > cfg.seed_steps:
                action = agent.act(obs, t0=(len(tds) == 1))
            else:
                action = env.rand_act()
            obs, reward, done, info = env.step(action)
            tds.append(_to_td(env, obs, action, reward, info['terminated']))

            # Update agent
            if local_step >= cfg.seed_steps:
                if local_step == cfg.seed_steps and not skip_pretrain:
                    num_updates = cfg.seed_steps
                    pbar.write(colored(
                        'Pretraining agent on seed data...', 'yellow'))
                else:
                    num_updates = 1
                for _ in range(num_updates):
                    _m = agent.update(buffer)
                train_metrics.update(_m)

            local_step += 1
            if local_step <= cfg.steps:
                pbar.update(1)
    finally:
        pbar.close()

    return local_step


# ==========================================================================
# Main sequential ER loop
# ==========================================================================

def main(args):
    assert torch.cuda.is_available(), 'TD-MPC2 requires CUDA.'

    num_tasks = len(args.tasks)
    if len(args.task_steps) != num_tasks:
        raise ValueError('--task-steps must have one value per task.')
    if args.task_buffer_sizes is not None:
        if len(args.task_buffer_sizes) != num_tasks:
            raise ValueError('--task-buffer-sizes must have one value per task.')
        task_buffer_sizes = list(args.task_buffer_sizes)
    else:
        task_buffer_sizes = [args.buffer_size] * num_tasks

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ---- shared overrides --------------------------------------------------
    shared = dict(
        model_size=args.model_size,
        batch_size=args.batch_size,
        # buffer_size is set per-task below (see task_buffer_sizes).
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
        enable_wandb=False,  # we manage wandb ourselves
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        compile=args.compile,
        # Replay-buffer persistence + storage knobs (read by Buffer / trainer).
        save_episodes=args.save_episodes,
        episode_dir=None,                     # per-task dir is built inside the task loop
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
    )

    # ---- per-task configs --------------------------------------------------
    task_cfgs = []
    for i in range(num_tasks):
        task_logdir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
        per_task = {**shared, 'buffer_size': int(task_buffer_sizes[i])}
        cfg = build_task_cfg(
            args.tasks[i], args.task_steps[i], task_logdir,
            per_task, args.seed,
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

    # ---- Single wandb run spanning every task ------------------------------
    stored_run_id = (progress or {}).get('wandb_run_id') if progress else None
    effective_run_id = args.wandb_run_id or stored_run_id

    wandb_run = None
    if args.logger == 'wandb':
        import wandb
        run_name = args.wandb_run_name or (
            f'seq_tdmpc2_er_{"-".join(args.tasks)}_s{args.seed}'
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
                'er_buffer_ratio': args.er_buffer_ratio,
                'er_seed': args.er_seed,
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'er'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 ER TRAINING', 'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir:           {base_logdir}')
    print(f'  Seed:             {args.seed}')
    print(f'  ER buffer ratio:  {args.er_buffer_ratio} '
          f'(budget per prev task = ratio * that task\'s cfg.steps)')
    print(f'  ER seed:          {args.er_seed}')
    print(f'  Wandb project:    {args.wandb_project}')
    print('=' * 64)

    # Eval env cache (lazy; persist across tasks for cross-task eval) --------
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

        print('=' * 64)
        print(colored(
            f'>>> TASK {task_idx + 1}/{num_tasks}: {task_name}',
            'cyan', attrs=['bold'],
        ))
        print(f'    internal steps: {cfg.steps}  '
              f'(action_repeat={cfg.action_repeat} -> '
              f'{cfg.steps * cfg.action_repeat} env frames)')
        print(f'    global step start: {global_step}')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        # Buffer isolation for in-task `train_eps/`: cross-task replay in ER
        # comes from each previous task's reservoir-sampled `er_episodes.pt`,
        # not from the in-task `train_eps/` dir. Once a previous task is
        # fully completed its `train_eps/` is no longer needed for resume,
        # so delete it on entry to a later task. (A partially-trained earlier
        # task is protected so its resume path stays intact.)
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

        # Fresh env + buffer for this task --------------------------------
        train_env = make_env(cfg)
        buffer = Buffer(cfg)

        # Per-task on-disk episode dir for full-state resume.
        task_episode_dir = task_logdir / 'train_eps'
        task_buffer_cap = int(min(cfg.buffer_size, cfg.steps))
        if args.save_episodes and task_episode_dir.is_dir():
            stats = buffer.load_from_directory(
                task_episode_dir, max_total_steps=task_buffer_cap)
            if stats['episodes_restored'] > 0:
                print(colored(
                    f'>>> Replayed {stats["transitions_restored"]:,} transitions '
                    f'from {stats["episodes_restored"]} episodes '
                    f'(cap {task_buffer_cap:,}).',
                    'green'))
                if stats['kept_files']:
                    dropped = Buffer.erase_over_episode_files(
                        task_episode_dir, stats['kept_files'])
                    if dropped:
                        print(colored(f'>>> Pruned {dropped} stale episode files.', 'yellow'))

        logger = SequentialLogger(
            cfg, task_idx, num_tasks, task_name, wandb_run,
            save_video=args.save_video,
        )

        # Build agent. For task > 0 load previous final checkpoint.
        agent = TDMPC2(cfg)
        if task_idx > 0 and prev_ckpt is not None and prev_ckpt.exists():
            print(colored(
                f'>>> Loading previous-task checkpoint: {prev_ckpt}',
                'cyan',
            ))
            agent.load(prev_ckpt)
        elif task_idx == 0 and args.from_checkpoint is not None:
            ck = Path(args.from_checkpoint)
            if ck.exists():
                print(colored(f'>>> Loading external checkpoint: {ck}',
                              'cyan'))
                agent.load(ck)
            else:
                raise FileNotFoundError(ck)

        # ============================================================
        # [ER] Prefill replay buffer with episodes from previous tasks
        # ============================================================
        if task_idx > 0 and args.er_buffer_ratio > 0:
            print(colored(
                f'>>> ER: loading episodes from {task_idx} previous task(s)...',
                'cyan',
            ))
            total_eps_loaded = 0
            total_trans_loaded = 0
            for j in range(task_idx):
                er_path = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                           / 'er_episodes.pt')
                eps = load_er_episodes(er_path)
                if not eps:
                    print(colored(
                        f'    WARNING: no ER dump for task {j + 1} '
                        f'({args.tasks[j]}) at {er_path} — skipping',
                        'yellow',
                    ))
                    continue
                # Insert each episode into the buffer. Buffer.add expects a
                # single-episode TensorDict (batch_dim = T) and assigns its
                # own episode index, so just call it per episode.
                for td in eps:
                    buffer.add(td)
                n_trans = sum(_episode_transitions(td) for td in eps)
                total_eps_loaded += len(eps)
                total_trans_loaded += n_trans
                print(f'    task {j + 1} ({args.tasks[j]}): '
                      f'{len(eps)} eps, {n_trans} transitions')
            print(colored(
                f'>>> ER: prefilled buffer with {total_eps_loaded} eps '
                f'({total_trans_loaded} transitions, '
                f'buffer.num_eps={buffer.num_eps})',
                'cyan',
            ))

        # Cross-task evaluation closure -----------------------------------
        def eval_fn(agent, local_step, gstep,
                    _task_idx=task_idx, _logger=logger, _cfg=cfg):
            for j in range(_task_idx + 1):
                is_current = (j == _task_idx)
                env_j = _get_eval_env(j)
                video_rec = _logger.video if (is_current and _logger.video
                                              ) else None
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

        # Train -----------------------------------------------------------
        # `episode_log` holds every completed episode of THIS task (CPU side),
        # to be reservoir-sampled into the on-disk ER dump at task end.
        episode_log = []
        train_one_task(
            cfg, agent, train_env, buffer, logger,
            global_step_start=global_step,
            eval_fn=eval_fn,
            skip_pretrain=args.skip_pretrain or (task_idx > 0),
            episode_log=episode_log,
            task_idx=task_idx,
            num_tasks=num_tasks,
            task_name=task_name,
            episode_dir=task_episode_dir,
            save_episodes=args.save_episodes,
            buffer_cap=task_buffer_cap,
            prune_every_n_eps=args.prune_every_n_episodes,
        )

        # Final cross-task eval + save ------------------------------------
        eval_fn(agent, cfg.steps, global_step + cfg.steps)
        logger.save_agent(agent, identifier='final')
        prev_ckpt = logger.model_dir / 'final.pt'

        # ============================================================
        # [ER] Dump reservoir-sampled episodes for this task
        # ============================================================
        if args.er_buffer_ratio > 0:
            er_budget = int(args.er_buffer_ratio * cfg.steps)
            er_path = task_logdir / 'er_episodes.pt'
            save_er_episodes(
                episode_log, er_path, er_budget,
                seed=args.er_seed + task_idx,
            )
        del episode_log

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

    # Close cached eval envs
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
        description='Sequential TD-MPC2 training with Experience Replay.',
    )

    # Task list
    p.add_argument('--tasks', nargs='+', required=True,
                   help='Task names (e.g. metaworld_drawer-open-v3 ...).')
    p.add_argument('--task-steps', nargs='+', type=int, required=True,
                   help='Training steps per task (internal steps; one per task).')

    # Logging / checkpoint
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--logger', type=str, default='wandb',
                   choices=['wandb', 'none'])
    p.add_argument('--wandb-entity', type=str, default='haoyu-a2i')
    p.add_argument('--wandb-project', type=str,
                   default='Metaworld_Tdmpc2_Sequential_ER')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None,
                   help="Existing wandb run id to resume with resume='allow'.")
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2_er')
    p.add_argument('--from-checkpoint', type=str, default=None,
                   help='Optional external checkpoint for task 1.')

    p.add_argument('--seed', type=int, default=1)

    # TDMPC2 hyperparams (shared across tasks)
    p.add_argument('--model-size', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=50000,
                   help='Single buffer size, broadcast to every task. '
                        'Ignored if --task-buffer-sizes is set.')
    p.add_argument('--task-buffer-sizes', nargs='+', type=int, default=None,
                   help='Per-task buffer caps (one per --tasks entry). '
                        'Overrides --buffer-size when given.')
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
    p.add_argument('--compile', dest='compile', action='store_true')
    p.add_argument('--no-compile', dest='compile', action='store_false')
    p.set_defaults(compile=True)
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

    # Replay-buffer persistence (mirrors STORM's train_eps/ scheme).
    p.add_argument('--save-episodes', dest='save_episodes', action='store_true',
                   help='Persist completed episodes to {task_logdir}/train_eps/ '
                        'and replay them back into the buffer on resume. '
                        'Independent of the cross-task ER dump (er_episodes.pt).')
    p.add_argument('--no-save-episodes', dest='save_episodes', action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5,
                   help='How often to FIFO-prune the on-disk episode dir down to '
                        '<= buffer_size transitions (1 = every episode).')
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'],
                   help='Override the auto CUDA/CPU heuristic for the replay '
                        'buffer storage. `cuda` forces GPU (OOMs if it does not fit).')

    # ER knobs
    p.add_argument(
        '--er-buffer-ratio', type=float, default=0.025,
        help='Fraction of each previous task\'s cfg.steps to keep as ER '
             'transitions. E.g. 0.025 = 2.5%% of that task\'s steps. '
             '0 disables ER (default: 0.025).',
    )
    p.add_argument(
        '--er-seed', type=int, default=42,
        help='Random seed for reservoir sampling of ER episodes (default: 42).',
    )

    main(p.parse_args())
