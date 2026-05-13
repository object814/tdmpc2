"""Sequential TD-MPC2 training with cross-task evaluation.

Trains TD-MPC2 on a sequence of tasks. For each new task:
  - The agent (world model + actor + critics, one state dict) is loaded from
    the previous task's final checkpoint (random init for task 1).
  - A fresh, empty replay buffer is used (no access to previous-task data).
  - Training proceeds with TD-MPC2's standard online loop.
  - At every eval_freq and at task end, the current agent is evaluated on
    ALL tasks seen so far (current + all previous), to measure forward
    transfer and catastrophic forgetting.

Unlike DreamerV3's sequential script, TD-MPC2 keeps everything in one
checkpoint — there is no need to separate a shared world model from
per-task heads.

Checkpoint layout per task:
    <logdir>/task{N}_<task>/
        models/final.pt        -- final agent state dict for the task
        models/{step}.pt       -- periodic snapshots (if save_freq > 0)
        eval_video/            -- TD-MPC2 video recorder output

Example:
    python sequential_train.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2/run0 \\
        --wandb-project Metaworld_Tdmpc2_Sequential
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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import VideoRecorder
from envs import make_env
from tdmpc2 import TDMPC2


CONFIG_PATH = _HERE / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# Config builder
# ==========================================================================

def build_task_cfg(task_name, task_steps, task_logdir, shared_overrides, seed):
    """Build a parsed TD-MPC2 config for a single task.

    `shared_overrides` is a dict of hyperparameters identical across tasks
    (model_size, batch_size, image_size, cameras, etc.).
    """
    cfg = OmegaConf.load(CONFIG_PATH)

    cfg.task = task_name
    cfg.steps = int(task_steps)
    cfg.seed = int(seed)
    cfg.work_dir = str(Path(task_logdir).resolve())
    cfg.data_dir = str(Path(task_logdir).resolve())

    for k, v in shared_overrides.items():
        if v is None:
            continue
        cfg[k] = v

    return parse_cfg(cfg)


# ==========================================================================
# Sequential progress tracking (for resume support)
# ==========================================================================

def _progress_file(base_logdir):
    return Path(base_logdir) / "sequential_progress.json"


def save_progress(base_logdir, task_idx, completed=False,
                  global_step_at_start=None, global_step_at_end=None,
                  task_name=None):
    path = _progress_file(base_logdir)
    prog = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
    entry = prog["tasks"].setdefault(str(task_idx), {})
    if task_name is not None:
        entry["task_name"] = task_name
    if global_step_at_start is not None:
        entry["global_step_at_start"] = int(global_step_at_start)
    if completed:
        entry["completed"] = True
    if global_step_at_end is not None:
        entry["global_step_at_end"] = int(global_step_at_end)
    path.write_text(json.dumps(prog, indent=2))


def load_progress(base_logdir):
    path = _progress_file(base_logdir)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_wandb_run_id(base_logdir, run_id):
    """Persist the wandb run id at the top of progress.json so resumes reuse it."""
    path = _progress_file(base_logdir)
    prog = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
    prog["wandb_run_id"] = run_id
    path.write_text(json.dumps(prog, indent=2))


# ==========================================================================
# Minimal logger that shares ONE wandb run across all tasks
# ==========================================================================

def _to_scalar(v):
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item()) if v.numel() == 1 else None
    if isinstance(v, (bool, int, float, np.integer, np.floating)):
        return float(v)
    return None


class SequentialLogger:
    """Duck-typed drop-in for TD-MPC2's Logger with cross-task awareness."""

    def __init__(self, cfg, task_idx, num_tasks, task_name, wandb_run,
                 save_video=True):
        self._cfg = cfg
        self._task_idx = task_idx
        self._num_tasks = num_tasks
        self._task_name = task_name
        self._work_dir = Path(cfg.work_dir)
        self._model_dir = self._work_dir / 'models'
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._wandb = wandb_run
        self.global_step_fn = lambda s: s  # overwritten by trainer per task
        if save_video and wandb_run is not None:
            self._video = _PrefixedVideoRecorder(cfg, wandb_run)
        else:
            self._video = None

    # ---- OnlineTrainer-compatible interface --------------------------------

    @property
    def video(self):
        return self._video

    @property
    def model_dir(self):
        return self._model_dir

    def log_train(self, d):
        """Log per-episode training metrics under a single continuous
        `train/*` namespace so the curve spans all tasks (visible drop at
        each task boundary)."""
        local_step = int(_to_scalar(d.get('step', 0)) or 0)
        gstep = int(self.global_step_fn(local_step))
        if self._wandb is not None:
            payload = {}
            for k, v in d.items():
                s = _to_scalar(v)
                if s is None:
                    continue
                payload[f'train/{k}'] = s
            payload['global_step'] = gstep
            payload['train/current_task_idx'] = self._task_idx + 1
            try:
                self._wandb.log(payload, step=gstep)
            except Exception as e:
                print(colored(f'[wandb.log warning] {e}', 'red'))
        # Training prints stay off — tqdm bar is enough.

    def log_eval(self, eval_task_idx, eval_task_name, d, gstep):
        """Log evaluation of the CURRENT agent on task `eval_task_idx`.

        Key scheme: `eval/task{j+1}_{name}/*`. Works uniformly whether the
        evaluated task is the current training task or a previous one —
        each task's eval curve is continuous across the full run.
        """
        gstep = int(gstep)
        if self._wandb is not None:
            pref = f'eval/task{eval_task_idx + 1}_{eval_task_name}'
            payload = {}
            for k, v in d.items():
                s = _to_scalar(v)
                if s is None:
                    continue
                payload[f'{pref}/{k}'] = s
            try:
                self._wandb.log(payload, step=gstep)
            except Exception as e:
                print(colored(f'[wandb.log warning] {e}', 'red'))
        r = _to_scalar(d.get('episode_reward', float('nan')))
        s = _to_scalar(d.get('episode_success', float('nan')))
        r = float('nan') if r is None else r
        s = float('nan') if s is None else s
        tag = colored('[eval]', 'green')
        print(
            f'  {tag} curr=T{self._task_idx + 1}({self._task_name}) '
            f'-> eval=T{eval_task_idx + 1}({eval_task_name}): '
            f'R={r:.2f} S={s:.2f}  @g={gstep}'
        )

    def save_agent(self, agent=None, identifier='final'):
        if agent is None:
            return
        fp = self._model_dir / f'{identifier}.pt'
        agent.save(fp)

    def finish(self, agent=None):
        if agent is not None:
            try:
                self.save_agent(agent, identifier='final')
            except Exception as e:
                print(colored(f'Save failed: {e}', 'red'))

class _PrefixedVideoRecorder(VideoRecorder):
    """VideoRecorder that writes under a caller-supplied per-eval prefix so
    videos from evaluation on different tasks don't collide in wandb."""

    def __init__(self, cfg, wandb_run):
        # TD-MPC2's VideoRecorder expects the `wandb` *module* (it uses
        # `wandb.Video` + `wandb.log`, which implicitly targets the active
        # run). Passing the Run object breaks `.Video` lookup.
        import wandb as _wandb_module
        super().__init__(cfg, _wandb_module)
        self._prefix = 'videos/eval_video'

    def set_prefix(self, prefix):
        self._prefix = prefix

    def save(self, step, key=None):
        if key is None:
            key = self._prefix
        return super().save(step, key=key)


# ==========================================================================
# Cross-task evaluation (no buffer needed)
# ==========================================================================

@torch.no_grad()
def eval_on_env(agent, env, episodes, video_recorder=None, video_step=None):
    rewards, successes, lengths = [], [], []
    for i in range(episodes):
        obs, done, ep_r, t = env.reset(), False, 0.0, 0
        if video_recorder is not None:
            video_recorder.init(env, enabled=(i == 0))
        while not done:
            torch.compiler.cudagraph_mark_step_begin()
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = env.step(action)
            ep_r += float(reward)
            t += 1
            if video_recorder is not None:
                video_recorder.record(env)
        rewards.append(ep_r)
        successes.append(float(info.get('success', 0.0)))
        lengths.append(t)
        if video_recorder is not None and video_step is not None:
            video_recorder.save(video_step)
    return {
        'episode_reward': float(np.nanmean(rewards)),
        'episode_success': float(np.nanmean(successes)),
        'episode_length': float(np.nanmean(lengths)),
    }


# ==========================================================================
# Per-task online training (mirrors OnlineTrainer.train with eval hooks)
# ==========================================================================

def _to_td(env, obs, action=None, reward=None, terminated=None):
    if isinstance(obs, dict):
        obs = TensorDict(
            {k: v.unsqueeze(0).cpu() for k, v in obs.items()},
            batch_size=(1,), device='cpu',
        )
    elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
        obs = obs.unsqueeze(0).cpu() if obs.batch_size == () else obs.cpu()
    else:
        obs = obs.unsqueeze(0).cpu()
    if action is None:
        action = torch.full_like(env.rand_act(), float('nan'))
    if reward is None:
        reward = torch.tensor(float('nan'))
    if terminated is None:
        terminated = torch.tensor(float('nan'))
    return TensorDict(
        obs=obs,
        action=action.unsqueeze(0),
        reward=reward.unsqueeze(0),
        terminated=terminated.unsqueeze(0),
        batch_size=(1,),
    )


def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   task_idx=0, num_tasks=1, task_name='',
                   episode_dir=None, save_episodes=True,
                   buffer_cap=None, prune_every_n_eps=5):
    """Run TD-MPC2's online training loop for a single task.

    `eval_fn(agent, local_step, global_step)` runs the cross-task evaluation.
    Returns the final local step count.
    `episode_dir` (when ``save_episodes=True``) is where each completed
    episode is dumped; FIFO-pruned to ``buffer_cap`` transitions.
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
                    if save_episodes and episode_dir is not None:
                        try:
                            Buffer.save_episode(ep_td, episode_dir, ep_idx)
                            if ep_idx % prune_every_n_eps == 0:
                                Buffer.prune_episode_dir_to_cap(episode_dir, buffer_cap)
                        except Exception as e:
                            print(colored(f'[seq_train] save_episode failed: {e}', 'red'))

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
# Main sequential loop
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

    # ---- shared overrides (same hyperparams across tasks) ------------------
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
        enable_wandb=False,  # we handle wandb ourselves
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        compile=args.compile,
        # Replay-buffer persistence + storage knobs (read by Buffer / trainer).
        save_episodes=args.save_episodes,
        episode_dir=None,                     # per-task dir is built inside the task loop
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
        # PRISM-WM (MoE) — wired into WorldModel via these flags. Defaults
        # to False (use_moe), so existing non-MoE sequential runs are
        # byte-identical. See run_prismatic_single.sh for the canonical
        # single-task launcher.
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        moe_residual_dynamics=args.moe_residual_dynamics,
    )

    # ---- Build per-task configs up-front -----------------------------------
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
    # Auto-resume: if progress.json already stored a wandb run id from a
    # previous (interrupted) invocation, reuse it so the wandb run is
    # continuous.  An explicit --wandb-run-id always wins.
    stored_run_id = (progress or {}).get('wandb_run_id') if progress else None
    effective_run_id = args.wandb_run_id or stored_run_id

    wandb_run = None
    if args.logger == 'wandb':
        import wandb
        run_name = args.wandb_run_name or (
            f'seq_tdmpc2_{"-".join(args.tasks)}_s{args.seed}'
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
                **shared,
            },
            tags=['sequential', 'tdmpc2'] + list(args.tasks),
        )
        # Persist run id so a later interruption + restart can re-attach.
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 TRAINING', 'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir: {base_logdir}')
    print(f'  Seed: {args.seed}')
    print(f'  Wandb project: {args.wandb_project}')
    if args.use_moe:
        print(colored(
            f'  PRISM-WM: K={args.num_experts} experts, '
            f'residual_dynamics={args.moe_residual_dynamics}  '
            f'(no Gram-Schmidt)',
            'cyan'))
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

        # Buffer isolation: regular sequential training has no access to
        # previous-task data, so delete every prior task's `train_eps/` dir
        # the first time we enter that task. (Only previous tasks already
        # marked completed get cleared — a partially-trained earlier task is
        # protected so its resume path stays intact.)
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

        # Per-task episode dir mirrors STORM's train_eps/ scheme.
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

        # Build agent.  For task > 0 load previous final checkpoint.
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

        # Final cross-task eval + save ------------------------------------
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
        description='Sequential TD-MPC2 training with cross-task evaluation.',
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
                   default='Metaworld_Tdmpc2_Sequential')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None,
                   help="Existing wandb run id to resume with resume='allow'.")
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2')
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

    # Env knobs (align with single_train_drawer.sh defaults)
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
                        'and replay them back into the buffer on resume.')
    p.add_argument('--no-save-episodes', dest='save_episodes', action='store_false')
    p.set_defaults(save_episodes=True)
    p.add_argument('--prune-every-n-episodes', type=int, default=5,
                   help='How often to FIFO-prune the on-disk episode dir down to '
                        '<= buffer_size transitions (1 = every episode).')
    p.add_argument('--buffer-storage-device', type=str, default='auto',
                   choices=['auto', 'cuda', 'cpu'],
                   help='Override the auto CUDA/CPU heuristic for the replay '
                        'buffer storage. `cuda` forces GPU (OOMs if it does not fit).')

    # PRISM-WM (MoE) — Prismatic World Model. When --use-moe is set, the
    # monolithic dynamics + reward heads are replaced by K-expert mixtures
    # with a softmax router. Gram-Schmidt orthogonalization is intentionally
    # omitted so we can observe whether experts collapse without it.
    # Defaults match config.yaml (use_moe=false), so omitting these flags
    # reproduces the original sequential pipeline byte-for-byte.
    p.add_argument('--use-moe', dest='use_moe', action='store_true',
                   help='Enable PRISM-WM MoE dynamics + reward heads.')
    p.add_argument('--no-use-moe', dest='use_moe', action='store_false')
    p.set_defaults(use_moe=False)
    p.add_argument('--num-experts', type=int, default=4,
                   help='Number of experts in each MoE block (only used when --use-moe).')
    p.add_argument('--moe-residual-dynamics', dest='moe_residual_dynamics',
                   action='store_true',
                   help='If set, dynamics MoE applies residual + post-residual '
                        'SimNorm: z_{t+1} = SimNorm(z + ∆z). Empirically the '
                        'non-residual setting works better with tdmpc2 SimNorm '
                        'latents — kept false by default.')
    p.add_argument('--no-moe-residual-dynamics', dest='moe_residual_dynamics',
                   action='store_false')
    p.set_defaults(moe_residual_dynamics=False)

    main(p.parse_args())
