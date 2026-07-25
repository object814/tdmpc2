"""Sequential TD-MPC2 continual learning — naive static-K MoE baseline.

DreamerV3-style split, identical to the progressive-MoE method
(`progmoe_training/*`) except for how the MoE expert pool is managed:

  PrismBackbone  — task-agnostic, persists across all tasks
      ├── _encoder    multi-modal RGB + proprio fusion
      └── _dynamics   MaskedMoEBlock, `num_experts` experts, gate routed by
                      the current-task one-hot.

  TaskModules    — task-specific, FRESH at every task boundary
      └── reward, termination, π, Q-ensemble (+ target Q), RunningScale.

In this *naive* baseline the MoE runs in **static full-K mode**: every one
of the `num_experts` experts is always active and always trainable — no
masking, no freezing. Task boundaries only swap the per-task `TaskModules`.
That isolates the contribution of progmoe's CL machinery (progressive
expert masking + freezing) from the shared backbone + per-task-heads split
and the task-conditioned routing, which both pipelines share.

The per-task training loop, cross-task evaluation, resume handling and
wandb wiring are shared with the progressive-MoE pipeline via
`run_continual_sequential(args, progressive=...)`.

Checkpoint layout per task:
    <logdir>/task{N}_<task>/
        backbone.pt        -- shared backbone weights + backbone optim state
        task_modules.pt    -- this task's heads + task/π optim + MPC prev-mean
        train_eps/         -- persisted replay episodes (FIFO-pruned)

Example:
    python sequential_train.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --num-experts 8 \\
        --logdir ../logdir/seq_prism_naive/run0 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_PrismNaive
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

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common.logger import VideoRecorder
from envs import make_env


CONFIG_PATH = _PARENT / 'config.yaml'

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
        # Periodic (step-numbered) saves keep only the latest snapshot on
        # disk; named checkpoints (final.pt, …) are never pruned.
        if str(identifier).isdigit():
            for old in self._model_dir.glob('*.pt'):
                if old.stem.isdigit() and int(old.stem) != int(identifier):
                    try:
                        old.unlink()
                    except OSError:
                        pass

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
                   buffer_cap=None, prune_every_n_eps=5,
                   episode_callback=None,
                   aux_buffers=None, aux_rng=None,
                   teacher_modules=None,
                   reward_anchor_coef=0.0,
                   termination_anchor_coef=0.0,
                   merged=False, merged_main_task_id=None, aux_sizes=None):
    """Run TD-MPC2's online training loop for a single task.

    `eval_fn(agent, local_step, global_step)` runs the cross-task evaluation.
    Returns the final local step count.
    `episode_dir` (when ``save_episodes=True``) is where each completed
    episode is dumped; FIFO-pruned to ``buffer_cap`` transitions.
    `episode_callback` (optional): when non-None, invoked as
    `episode_callback(ep_td)` after each completed episode has been added to
    the buffer. Used by the ER pipeline to feed an online reservoir.
    `aux_buffers` (optional): dict {task_idx_j: Buffer} of per-prior-task
    auxiliary ER buffers (anchored — never FIFO'd by new-task data). When
    non-empty, every main `agent.update(buffer)` is followed by one
    `agent.aux_update(aux_buffers[j], j)` for a randomly chosen j — the
    routing-aware anchored-ER step.
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
    cur_transitions = 0        # tracks current-task buffer fill (merged mode)
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
                    cur_transitions = min(
                        cur_transitions + _episode_transitions(ep_td),
                        buffer_cap)
                    if episode_callback is not None:
                        try:
                            episode_callback(ep_td)
                        except Exception as e:
                            print(colored(f'[seq_train] episode_callback failed: {e}', 'red'))
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
                _aux_rng = aux_rng or random
                for _ in range(num_updates):
                    if merged:
                        # Merged-buffer (multitask-style) update: ONE full-loss
                        # step on a within-batch task-mix from the current
                        # buffer + anchored ER buffers. Per-sample routing
                        # sends old-task rows through their frozen experts; the
                        # current heads see the whole mix (DreamerV3 recipe).
                        mb = _sample_merged_batch(
                            buffer, merged_main_task_id, cur_transitions,
                            aux_buffers or {}, aux_sizes or {},
                            cfg.batch_size, agent.device)
                        _m = agent.update_merged(*mb)
                    else:
                        _m = agent.update(buffer)
                        # Anchored + routing-aware ER step: for each main update
                        # do one auxiliary update on a random prior-task aux
                        # buffer, routed via that task's one-hot (frozen experts
                        # for that task produce the features) — world-model
                        # update only, task_modules are untouched.
                        if aux_buffers:
                            j_key = _aux_rng.choice(list(aux_buffers.keys()))
                            tm_j = (teacher_modules.get(j_key)
                                    if teacher_modules else None)
                            _aux = agent.aux_update(
                                aux_buffers[j_key], j_key,
                                teacher_modules=tm_j,
                                reward_anchor_coef=reward_anchor_coef,
                                termination_anchor_coef=termination_anchor_coef,
                            )
                            for k in _aux.keys():
                                _m[f'aux_{k}'] = _aux[k]
                train_metrics.update(_m)

            local_step += 1
            if local_step <= cfg.steps:
                pbar.update(1)
    finally:
        pbar.close()

    return local_step


# ==========================================================================
# ER helpers (online reservoir + on-disk dump/load)
# ==========================================================================

def _episode_transitions(td):
    """Number of transitions in a single-episode TensorDict.

    `_to_td` produces one td per env step (including the initial reset row),
    and `train_one_task` calls `torch.cat(tds)` before passing to the buffer.
    The first row is the reset (no real transition), so the transition count
    is `len(td) - 1`.
    """
    return max(int(td.batch_size[0]) - 1, 0)


def _make_er_reservoir(reservoir_K, rng):
    """Build an online reservoir-sample callback (Vitter's Algorithm R).

    Returns `(callback, reservoir_list)`. Each `callback(ep_td)` invocation
    either appends a CPU copy of `ep_td` to the reservoir (until K episodes
    are held) or, with probability K/i, replaces a random reservoir slot.
    Memory is O(K) regardless of how many episodes the task produces — same
    statistical property as the offline accumulate-then-sample approach but
    bounded for long-running tasks.
    """
    reservoir = []
    counter = {'i': 0}

    def callback(ep_td):
        ep_cpu = ep_td.detach().cpu()
        counter['i'] += 1
        i = counter['i']
        if len(reservoir) < reservoir_K:
            reservoir.append(ep_cpu)
        else:
            j = rng.randrange(i)
            if j < reservoir_K:
                reservoir[j] = ep_cpu

    return callback, reservoir


def save_er_episodes(episodes, path):
    """Save a list of single-episode TensorDicts to disk (CPU-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_eps = [td.detach().cpu() for td in episodes]
    torch.save(cpu_eps, str(path))
    n_trans = sum(_episode_transitions(td) for td in cpu_eps)
    return len(cpu_eps), n_trans


def load_er_episodes(path):
    """Load a list of single-episode TensorDicts from disk; [] if missing."""
    path = Path(path)
    if not path.exists():
        return []
    return torch.load(str(path), map_location='cpu', weights_only=False)


# ==========================================================================
# Shared continual-learning loop (naive static-K  +  progressive masked)
# ==========================================================================

def _build_aux_buffers(base_logdir, task_names, task_idx, cfg,
                       task_cfgs=None, device=None,
                       load_teacher_modules=False):
    """Build per-prior-task auxiliary ER buffers (anchored, never FIFO'd).

    Loads each prior task's reservoir-sampled `er_episodes.pt` into its OWN
    Buffer instance sized exactly to fit those transitions (plus a small
    headroom). That way new-task experience never evicts ER data — the
    encoder + dynamics MoE see prior-task transitions for the entire
    duration of the new task, not just the first ~25% before FIFO kicks in.

    When `load_teacher_modules=True`, also loads each prior task's saved
    `task_modules.pt` into a frozen `TaskModules` instance (used as teacher
    for the reward / termination anchors in `aux_update`). Requires
    `task_cfgs` and `device` to be passed.

    Returns: (aux_buffers, teacher_modules, aux_sizes). teacher_modules is an
    empty dict when load_teacher_modules=False or no checkpoints exist.
    aux_sizes maps task_idx -> transition count (used by the merged-buffer
    driver for proportional within-batch sampling).
    """
    import copy as _copy
    aux_buffers = {}
    teacher_modules = {}
    aux_sizes = {}
    if load_teacher_modules:
        from progmoe_training.backbone import TaskModules
        assert task_cfgs is not None and device is not None, (
            'load_teacher_modules=True requires task_cfgs and device.'
        )
    for j in range(task_idx):
        er_path = (
            base_logdir / f'task{j + 1}_{task_names[j]}' / 'er_episodes.pt'
        )
        eps = load_er_episodes(er_path)
        if not eps:
            print(colored(
                f'    WARNING: no ER dump for task {j + 1} '
                f'({task_names[j]}) at {er_path} — skipping.', 'yellow'))
            continue
        total_trans = sum(_episode_transitions(td) for td in eps)
        aux_cfg = _copy.copy(cfg)
        aux_cfg.buffer_size = max(1024, total_trans + 1024)
        aux_buf = Buffer(aux_cfg)
        for td in eps:
            aux_buf.add(td)
        aux_buffers[j] = aux_buf
        aux_sizes[j] = int(total_trans)
        print(f'    task {j + 1} ({task_names[j]}): aux buffer with '
              f'{len(eps)} eps, {total_trans} transitions '
              f'(cap {aux_cfg.buffer_size}, never FIFO).')

        if load_teacher_modules:
            tm_path = (
                base_logdir / f'task{j + 1}_{task_names[j]}'
                / 'task_modules.pt'
            )
            if not tm_path.exists():
                print(colored(
                    f'    WARNING: no task_modules.pt for task {j + 1} '
                    f'({task_names[j]}) at {tm_path} — anchors disabled '
                    f'for this task.', 'yellow'))
                continue
            tm = TaskModules(task_cfgs[j]).to(device)
            ck_tm = torch.load(
                tm_path, map_location=device, weights_only=False)
            tm.load_state_dict(ck_tm['model'])
            tm.eval()
            # Frozen teacher: weights never update. Gradient still flows
            # *through* them into upstream (encoder / dynamics).
            for p in tm.parameters():
                p.requires_grad = False
            teacher_modules[j] = tm
            print(f'    task {j + 1} ({task_names[j]}): teacher heads '
                  f'loaded (frozen) for reward / termination anchors.')
    return aux_buffers, teacher_modules, aux_sizes


def _sample_merged_batch(main_buffer, main_task_id, main_size,
                         aux_buffers, aux_sizes, batch_size, device):
    """Assemble ONE within-batch task-mixed training batch.

    Draws `batch_size` rows split across the current-task buffer and the
    anchored ER buffers, with per-source counts ~ multinomial proportional to
    each source's transition count. Early in a task the current buffer is
    small, so most rows come from the ER (old) tasks; as it fills, the mix
    shifts toward the current task — the DreamerV3 merged-replay dynamic.

    Each row's task id is synthesized from WHICH buffer it came from (current
    -> main_task_id, ER buffer j -> j), so downstream per-sample MoE routing
    sends task-j rows through task-j's experts. Returns
    (obs, action, reward, terminated, task) ready for `agent.update_merged`.
    """
    sources = [(int(main_task_id), main_buffer, max(int(main_size), 1))]
    for j, buf in aux_buffers.items():
        sources.append((int(j), buf, max(int(aux_sizes.get(j, 1)), 1)))

    probs = torch.tensor([float(s) for (_, _, s) in sources])
    idx = torch.multinomial(probs, batch_size, replacement=True)
    counts = torch.bincount(idx, minlength=len(sources)).tolist()

    obs_p, act_p, rew_p, term_p, task_p = [], [], [], [], []
    for (task_id, buf, _), cnt in zip(sources, counts):
        if cnt <= 0:
            continue
        obs, action, reward, terminated, _ = buf.sample()
        obs_p.append(obs[:, :cnt])
        act_p.append(action[:, :cnt])
        rew_p.append(reward[:, :cnt])
        term_p.append(terminated[:, :cnt])
        task_p.append(torch.full(
            (cnt,), task_id, dtype=torch.long, device=obs.device))

    obs = torch.cat(obs_p, dim=1)
    action = torch.cat(act_p, dim=1)
    reward = torch.cat(rew_p, dim=1)
    terminated = torch.cat(term_p, dim=1)
    task = torch.cat(task_p, dim=0)
    return obs, action, reward, terminated, task


def run_continual_sequential(args, *, progressive,
                             er_buffer_ratio=0.0, er_seed=42,
                             per_task_tau=False,
                             reward_anchor_coef=0.0,
                             termination_anchor_coef=0.0,
                             merged=False,
                             sdp_freeze=False):
    """Shared sequential continual-learning driver.

    Both pipelines build a DreamerV3-style split agent — a shared
    `PrismBackbone` (encoder + dynamics MoE) plus a per-task `TaskModules`
    (reward / Q / termination / π / scale). The ONLY difference is the MoE
    expert-pool management:

      progressive=True  (progmoe): the `MaskedMoEBlock` grows its active
          window by `args.K_per_task` experts each task and freezes the
          experts from prior tasks. total_K = num_tasks * K_per_task.

      progressive=False (naive sequential): static full-K MoE — all
          `args.num_experts` experts are always active and trainable. Task
          boundaries only swap the per-task `TaskModules`.

    `ContinualTDMPC2.start_new_task` / `PrismBackbone` branch on the
    `progressive_moe` cfg flag, so this driver itself stays mode-agnostic
    apart from the bookkeeping of `total_K` and the cross-task eval
    `active_K`.

    Experience Replay (optional, `er_buffer_ratio > 0`):
      - During training each completed episode is fed to an online
        reservoir (Vitter's Algorithm R) sized to
        `K = (er_buffer_ratio · cfg.steps) / cfg.episode_length` episodes.
      - At task end the reservoir is dumped to
        `{task_logdir}/er_episodes.pt`.
      - At the start of task t > 0, every prior task's `er_episodes.pt` is
        loaded into a SEPARATE per-task auxiliary buffer (never FIFO-evicted
        by new-task data — *anchored ER*).
      - During training, each main `agent.update(buffer)` is followed by an
        `agent.aux_update(aux_buffers[j], j)` for a randomly chosen prior
        task j. The aux update routes via task-j's one-hot so the *frozen*
        task-j experts produce the features (*routing-aware ER*), and the
        backward only steps `backbone_optim` — `task_modules` (per-task
        reward / Q / π) are untouched.
      - `er_buffer_ratio = 0` (default) disables ER; no aux buffers, no
        reservoir dumps, no aux updates.
    """
    # Lazy import — keeps `import sequential_train` free of any progmoe
    # import-time coupling (helpers above are imported by other CL trainers).
    from progmoe_training.continual_tdmpc2 import ContinualTDMPC2
    from progmoe_training.backbone import TaskModules, EvalAgent

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

    # Mode-specific expert-count bookkeeping.
    if progressive:
        K_per_task = int(args.K_per_task)
        if K_per_task <= 0:
            raise ValueError('--K-per-task must be > 0.')
        total_K = num_tasks * K_per_task
    else:
        K_per_task = 0  # unused in static mode
        total_K = int(args.num_experts)
        if total_K <= 0:
            raise ValueError('--num-experts must be > 0.')

    method = 'progmoe' if progressive else 'naive'

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ---- shared cfg overrides (identical across tasks) --------------------
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
        enable_wandb=False,  # one wandb run is driven manually below
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        compile=False,  # MaskedMoEBlock state mutates; cudagraph capture invalid
        save_episodes=args.save_episodes,
        episode_dir=None,
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
        # PRISM-WM continual
        use_moe=True,
        num_tasks=num_tasks,
        progressive_moe=progressive,
        use_orthogonal=args.use_orthogonal,
        # Offline-pretrained encoder (loaded + optionally frozen in
        # PrismBackbone.__init__). null path -> fresh encoder. When
        # freeze_encoder=True the encoder is held fixed for ALL tasks.
        pretrained_encoder=getattr(args, 'pretrained_encoder', None),
        freeze_encoder=bool(getattr(args, 'freeze_encoder', False)),
        # rgb-branch encoder type: 'dino' swaps the CNN for a FROZEN DINOv2
        # backbone + trainable pooling head (see layers.DinoRGBEncoder).
        # The frozen backbone persists across all tasks; only the pooling/
        # projection/fusion train. dino_* knobs come from config.yaml.
        encoder_type=getattr(args, 'encoder_type', 'default'),
        # SDP-faithful (Plan A) freeze: after task 0 ContinualTDMPC2 freezes
        # the shared MoE head + gate τ-column + τ schedule (old experts already
        # progressively frozen; per-task heads already isolated). Prevents
        # forgetting while keeping new-task routing free to reuse old experts.
        sdp_freeze=sdp_freeze,
    )
    if progressive:
        shared['K_per_task'] = K_per_task

    # ---- per-task cfgs ----------------------------------------------------
    # `num_experts` drives PrismBackbone.total_K in static mode; in
    # progressive mode PrismBackbone derives total_K = num_tasks*K_per_task
    # itself, but we still set it so cfg / wandb logging stays consistent.
    task_cfgs = []
    for i in range(num_tasks):
        task_logdir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
        per_task = {
            **shared,
            'buffer_size': int(task_buffer_sizes[i]),
            'num_experts': total_K,
        }
        cfg = build_task_cfg(
            args.tasks[i], args.task_steps[i], task_logdir, per_task, args.seed,
        )
        task_cfgs.append(cfg)

    set_seed(args.seed)

    # ---- Resume detection -------------------------------------------------
    resume_from = 0
    global_step = 0
    progress = load_progress(base_logdir)
    if progress is not None:
        for i in range(num_tasks):
            entry = progress.get('tasks', {}).get(str(i))
            if entry and entry.get('completed'):
                resume_from = i + 1
                global_step = int(entry['global_step_at_end'])
                print(colored(
                    f'>>> RESUME: task {i + 1} ({args.tasks[i]}) '
                    f'already done (@g={global_step}).', 'cyan'))
            else:
                break
    if resume_from >= num_tasks:
        print(colored('>>> All tasks already completed. Nothing to do.',
                      'green', attrs=['bold']))
        return

    # Resume hygiene: prior-task cfgs were never threaded through make_env
    # (their training loop is skipped), so cfg.action_dim / cfg.obs_shape
    # remain YAML placeholders. eval_fn instantiates TaskModules from these
    # cfgs to load saved per-task heads — without this priming, that crashes
    # with `int + str` in backbone.TaskModules.__init__.
    for j in range(resume_from):
        _env = make_env(task_cfgs[j])
        try:
            _env.close()
        except Exception:
            pass
        del _env

    # ---- One wandb run across all tasks -----------------------------------
    stored_run_id = (progress or {}).get('wandb_run_id') if progress else None
    effective_run_id = args.wandb_run_id or stored_run_id

    wandb_run = None
    if args.logger == 'wandb':
        import wandb
        default_prefix = 'progmoe_tdmpc2' if progressive else 'seq_naive_tdmpc2'
        run_name = args.wandb_run_name or (
            f'{default_prefix}_{"-".join(args.tasks)}_s{args.seed}'
        )
        if effective_run_id is not None:
            print(colored(
                f'>>> RESUME: attaching to existing wandb run id='
                f'{effective_run_id}', 'cyan',
            ))
        # Launcher-provided tags / group take precedence over the
        # auto-generated ones, so the sweep-style naming convention
        # (run_name = "{taskset}_seed{seed}", group = "{taskset}", tags
        # carrying method/seed/sweep_id) reaches wandb verbatim.
        explicit_tags = list(getattr(args, 'wandb_tags', None) or [])
        if explicit_tags:
            tags = explicit_tags
        else:
            tags = ['sequential', 'tdmpc2', 'prism']
            tags += (['progmoe', 'masked'] if progressive
                     else ['naive', 'static'])
            if args.use_orthogonal:
                tags.append('orthogonal')
            if er_buffer_ratio > 0:
                tags.append('er')
            if sdp_freeze:
                tags.append('sdp_freeze')
            if per_task_tau:
                tags.append('per_task_tau')
            tags += list(args.tasks)
        wandb_group = getattr(args, 'wandb_group', None) or None
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=wandb_group,
            id=effective_run_id,
            resume='allow' if effective_run_id else None,
            dir=str(base_logdir),
            config={
                'tasks': args.tasks,
                'task_steps': args.task_steps,
                'seed': args.seed,
                'method': method,
                'num_tasks': num_tasks,
                'total_K': total_K,
                'K_per_task': K_per_task,
                'progressive_moe': progressive,
                'use_orthogonal': args.use_orthogonal,
                'er_buffer_ratio': er_buffer_ratio,
                'er_seed': er_seed,
                'sdp_freeze': sdp_freeze,
                'per_task_tau': per_task_tau,
                'reward_anchor_coef': reward_anchor_coef,
                'termination_anchor_coef': termination_anchor_coef,
                'merged': merged,
                **shared,
            },
            tags=tags,
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary -----------------------------------------------------
    print('=' * 64)
    if progressive:
        print(colored('>>> PROGRESSIVE-MoE TD-MPC2 (static-K masked)',
                      'cyan', attrs=['bold']))
    else:
        print(colored('>>> NAIVE SEQUENTIAL TD-MPC2 '
                      '(DreamerV3 split, static full-K MoE)',
                      'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        if progressive:
            print(f'  Task {i + 1}: {t}  steps={s}  '
                  f'(active_K at end = {(i + 1) * K_per_task}/{total_K}){mark}')
        else:
            print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir: {base_logdir}')
    print(f'  Seed: {args.seed}')
    print(f'  Wandb project: {args.wandb_project}')
    if progressive:
        print(colored(
            f'  K_per_task={K_per_task}, num_tasks={num_tasks}, '
            f'total_K={total_K}', 'cyan'))
    else:
        print(colored(
            f'  Static MoE: num_experts={total_K} '
            f'(all active, all trainable, no freezing)', 'cyan'))
    print(colored(
        f'  Backbone: encoder + MoE dynamics (gate=task_one_hot). '
        f'Per-task: reward, Q, termination, π, scale.   '
        f'use_orthogonal={args.use_orthogonal}', 'cyan'))
    if er_buffer_ratio > 0:
        print(colored(
            f'  ER: buffer_ratio={er_buffer_ratio}, seed={er_seed} '
            f'(anchored aux buffers + routing-aware aux update / step; '
            f'world-model only, task_modules untouched)', 'cyan'))
    if sdp_freeze:
        print(colored(
            '  SDP-FREEZE (Plan A): after task 0, shared MoE head + gate '
            'τ-column + τ schedule are frozen. Old experts progressively '
            'frozen; per-task heads isolated. New-task gate column stays free '
            'to reuse old experts (transfer). -> no forgetting on prior tasks.',
            'cyan'))
    if per_task_tau:
        print(colored(
            '  PER-TASK-τ: cross-task eval restores each prior task\'s '
            'converged τ from its backbone.pt snapshot (closes the τ-drift '
            'leak for non-freeze recipes at eval time).',
            'cyan'))
    if reward_anchor_coef > 0 or termination_anchor_coef > 0 and not merged:
        print(colored(
            f'  ANCHOR: reward_coef={reward_anchor_coef}, '
            f'termination_coef={termination_anchor_coef}. '
            f'Each aux step adds a soft-CE / BCE loss through the FROZEN '
            f'teacher heads of the sampled prior task, pinning the live '
            f'encoder + dynamics to old-task-head-compatible latents.',
            'cyan'))
    if merged:
        print(colored(
            '  MERGED (DreamerV3-style multitask ER): no separate aux step. '
            'Each update samples ONE within-batch task-mix (current buffer + '
            'anchored ER, proportional to size) and runs the FULL loss. Old '
            'rows route PER-SAMPLE through their frozen experts; the current '
            'heads train on the whole mix. Eval unchanged (per-task window + '
            'saved per-task heads).', 'cyan'))
    print('=' * 64)

    # Eval envs are built FRESH per evaluation and closed immediately after
    # (see eval_fn below). They are deliberately NOT cached: a long-lived
    # cached eval env, coexisting in-process with later tasks' train/eval
    # envs under MUJOCO_GL=osmesa, gets its MuJoCo render context corrupted
    # and silently returns garbage camera observations — which collapses the
    # *live* wandb eval curve of a previous task to random-policy levels even
    # though the model weights are perfectly fine. (Offline re-eval of the
    # same checkpoints reproduces good performance — that is the tell.)

    # Agent built lazily on the first loop iteration — make_env populates
    # `cfg.obs_shape` / `cfg.action_dim`, which size the encoder + MoE. After
    # the first build the agent persists in memory across all tasks.
    agent = None

    # ---- Per-task training loop -------------------------------------------
    for task_idx in range(resume_from, num_tasks):
        cfg = task_cfgs[task_idx]
        task_name = args.tasks[task_idx]
        task_logdir = Path(cfg.work_dir)
        task_logdir.mkdir(parents=True, exist_ok=True)

        print('=' * 64)
        print(colored(
            f'>>> TASK {task_idx + 1}/{num_tasks}: {task_name}',
            'cyan', attrs=['bold'],
        ))
        if agent is not None and progressive:
            print(f'    Backbone mask: active_K={agent.backbone.active_K}, '
                  f'frozen_K={agent.backbone.frozen_K}')
        print(f'    Internal steps: {cfg.steps}')
        print(f'    Global step start: {global_step}')
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

        # Fresh env + buffer for this task. make_env mutates `cfg` to set
        # `obs_shape` / `action_dim` from the env's observation space.
        train_env = make_env(cfg)
        buffer = Buffer(cfg)

        # ---- One-time agent construction --------------------------------
        # On the first iteration (or after a resume), build the agent from
        # the current task's cfg and replay through any already-completed
        # task boundaries to bring the backbone state up to `task_idx`.
        if agent is None:
            agent = ContinualTDMPC2(cfg)
            for i in range(task_idx):
                prev_dir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
                bb_path = prev_dir / 'backbone.pt'
                if not bb_path.exists():
                    raise FileNotFoundError(
                        f'expected completed-task backbone at {bb_path}; '
                        f'cannot resume from task {task_idx}.')
                agent.load_backbone(bb_path, load_optim=True)
                agent.start_new_task(i + 1)
            if task_idx == 0 and args.from_checkpoint is not None:
                ck = Path(args.from_checkpoint)
                if (ck / 'backbone.pt').exists():
                    print(colored(
                        f'>>> Loading external start checkpoint: {ck}',
                        'cyan'))
                    agent.load_backbone(ck / 'backbone.pt', load_optim=True)
                    if (ck / 'task_modules.pt').exists():
                        agent.load_task_modules(
                            ck / 'task_modules.pt', load_optim=True)
                else:
                    raise FileNotFoundError(
                        f'--from-checkpoint dir {ck} missing backbone.pt')

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

        # ---- Build per-prior-task aux buffers (anchored ER, never FIFO'd).
        # Replaces the older "load ER into the main buffer" prefill — that
        # leaked ER episodes via FIFO eviction once the main buffer filled
        # with new-task data. Each prior task's reservoir now lives in its
        # OWN Buffer; the trainer samples from one per update.
        aux_buffers = {}
        teacher_modules = {}
        aux_sizes = {}
        if er_buffer_ratio > 0 and task_idx > 0:
            print(colored(
                f'>>> ER: building per-task aux buffers '
                f'({task_idx} prior task(s))...', 'cyan'))
            load_teachers = (
                (reward_anchor_coef > 0 or termination_anchor_coef > 0)
                and not merged
            )
            aux_buffers, teacher_modules, aux_sizes = _build_aux_buffers(
                base_logdir, args.tasks, task_idx, cfg,
                task_cfgs=task_cfgs, device=agent.device,
                load_teacher_modules=load_teachers,
            )
            if merged:
                print(colored(
                    f'>>> MERGED: {len(aux_buffers)} ER source(s) ready; each '
                    f'step samples ONE within-batch task-mix (current + ER, '
                    f'proportional) and does a single full-loss update.',
                    'cyan'))
            else:
                print(colored(
                    f'>>> ER: {len(aux_buffers)} aux buffer(s) ready; '
                    f'each main update will be followed by one routing-aware '
                    f'anchored update on a random prior task.', 'cyan'))
                if load_teachers:
                    print(colored(
                        f'>>> ANCHOR: {len(teacher_modules)} teacher head(s) '
                        f'loaded; each aux step adds the supervised anchor '
                        f'loss(es) through them.', 'cyan'))

        logger = SequentialLogger(
            cfg, task_idx, num_tasks, task_name, wandb_run,
            save_video=args.save_video,
        )

        # Ensure the agent's routing one-hot matches the active training task.
        agent.set_current_task(task_idx)

        # ---- ER capture: build the online reservoir callback for THIS task.
        # `cfg.episode_length` was populated by make_env above, so we have a
        # real value for the K calculation. Reservoir state is closed-over by
        # the callback and read at task end below.
        if er_buffer_ratio > 0:
            er_budget_transitions = int(er_buffer_ratio * cfg.steps)
            ep_len = max(1, int(cfg.episode_length))
            reservoir_K = max(1, er_budget_transitions // ep_len)
            er_rng = random.Random(er_seed + task_idx)
            er_callback, er_reservoir = _make_er_reservoir(reservoir_K, er_rng)
            print(colored(
                f'>>> ER: reservoir K={reservoir_K} episodes '
                f'(budget={er_budget_transitions} transitions @ '
                f'episode_length={ep_len}).', 'cyan'))
        else:
            er_callback = None
            er_reservoir = None

        # ---- Cross-task eval closure -----------------------------------
        # j == task_idx : live agent
        # j <  task_idx : load saved task_modules_j into a fresh TaskModules,
        #                 pair with the LIVE backbone via EvalAgent.
        def eval_fn(agent, local_step, gstep,
                    _task_idx=task_idx, _logger=logger, _cfg=cfg):
            for j in range(_task_idx + 1):
                is_current = (j == _task_idx)
                video_rec = (
                    _logger.video if (is_current and _logger.video) else None
                )
                if video_rec is not None:
                    video_rec.set_prefix(
                        f'eval/task{j + 1}_{args.tasks[j]}/videos/eval_video'
                    )

                if is_current:
                    agent.set_current_task(j)
                    eval_subject = agent
                else:
                    prev_dir = base_logdir / f'task{j + 1}_{args.tasks[j]}'
                    tm_path = prev_dir / 'task_modules.pt'
                    if not tm_path.exists():
                        ptr = prev_dir / 'models' / 'final.pt'
                        if ptr.exists():
                            ck = torch.load(
                                ptr, map_location='cpu', weights_only=False)
                            tm_path = Path(ck['task_dir']) / 'task_modules.pt'
                        else:
                            print(colored(
                                f'[eval] no task_modules.pt for task {j + 1} '
                                f'({args.tasks[j]}); skipping eval.',
                                'yellow'))
                            continue
                    loaded_tm = TaskModules(task_cfgs[j]).to(agent.device)
                    ck_tm = torch.load(
                        tm_path, map_location=agent.device, weights_only=False)
                    loaded_tm.load_state_dict(ck_tm['model'])
                    loaded_tm.eval()
                    for p in loaded_tm.parameters():
                        p.requires_grad = False
                    # Static mode: all experts always active. Progressive
                    # mode: task j only ever used experts [0, (j+1)·K).
                    active_K_eval = (
                        (j + 1) * K_per_task if progressive else total_K
                    )
                    # Per-task τ snapshot: read task-j's converged τ from
                    # backbone.pt (`_dynamics._tau_buf` is a registered
                    # buffer, so it round-trips through the per-task save).
                    # Only loaded when `per_task_tau=True`; otherwise the
                    # eval uses whatever τ the live backbone currently
                    # holds, matching the pre-snapshot behavior.
                    tau_j = None
                    if per_task_tau:
                        bb_path_j = prev_dir / 'backbone.pt'
                        if bb_path_j.exists():
                            try:
                                ck_bb = torch.load(
                                    bb_path_j, map_location='cpu',
                                    weights_only=False)
                                tau_buf = ck_bb.get('model', {}).get(
                                    '_dynamics._tau_buf', None)
                                if tau_buf is not None:
                                    tau_j = float(tau_buf.item())
                            except Exception as e:
                                print(colored(
                                    f'[eval] failed to load τ for task {j + 1}'
                                    f' from {bb_path_j} ({e}); using live τ.',
                                    'yellow'))
                    eval_subject = EvalAgent(
                        agent_for_plan=agent,
                        backbone=agent.backbone,
                        task_modules=loaded_tm,
                        task_idx=j,
                        active_K=active_K_eval,
                        cfg=task_cfgs[j],
                        tau=tau_j,
                    )

                # Build the eval env FRESH for this evaluation and close it
                # immediately after. Never cache / reuse it across task
                # boundaries — a stale eval env's MuJoCo render context gets
                # corrupted once later tasks' envs are created in-process,
                # and it then silently feeds the agent garbage observations.
                env_j = make_env(task_cfgs[j])
                try:
                    metrics = eval_on_env(
                        eval_subject, env_j, _cfg.eval_episodes,
                        video_recorder=video_rec,
                        video_step=gstep if video_rec else None,
                    )
                finally:
                    try:
                        env_j.close()
                    except Exception:
                        pass
                    del env_j
                metrics['step'] = local_step
                _logger.log_eval(j, args.tasks[j], metrics, gstep)

                if not is_current:
                    del eval_subject, loaded_tm
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Restore training task one-hot.
            agent.set_current_task(_task_idx)

        # ---- Train this task -------------------------------------------
        aux_rng = random.Random(er_seed + 10_000 + task_idx)
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
            episode_callback=er_callback,
            aux_buffers=aux_buffers if aux_buffers else None,
            aux_rng=aux_rng,
            teacher_modules=teacher_modules if teacher_modules else None,
            reward_anchor_coef=reward_anchor_coef,
            termination_anchor_coef=termination_anchor_coef,
            merged=merged,
            merged_main_task_id=task_idx,
            aux_sizes=aux_sizes,
        )

        # ---- Final cross-task eval + save ------------------------------
        eval_fn(agent, cfg.steps, global_step + cfg.steps)
        agent.save_task(task_logdir)
        print(colored(
            f'>>> Saved task {task_idx + 1} checkpoints: '
            f'{task_logdir}/{{backbone,task_modules}}.pt',
            'green'))

        # ---- ER dump: save the online reservoir for this task ----------
        if er_buffer_ratio > 0 and er_reservoir is not None:
            er_out = task_logdir / 'er_episodes.pt'
            n_eps, n_trans = save_er_episodes(er_reservoir, er_out)
            print(colored(
                f'>>> ER: saved {n_eps} reservoir episodes '
                f'({n_trans} transitions) -> {er_out}', 'cyan'))

        global_step += cfg.steps
        save_progress(
            base_logdir, task_idx, task_name=task_name,
            completed=True, global_step_at_end=global_step,
        )

        # ---- Transition: prepare agent for task t+1 --------------------
        if task_idx + 1 < num_tasks:
            if progressive:
                print(colored(
                    f'>>> Transition: freezing through '
                    f'{(task_idx + 1) * K_per_task} experts, opening '
                    f'{(task_idx + 2) * K_per_task} slot.', 'cyan'))
            else:
                print(colored(
                    '>>> Transition: fresh TaskModules '
                    '(backbone MoE unchanged — static full-K).', 'cyan'))
            agent.start_new_task(task_idx + 1)

        try:
            train_env.close()
        except Exception:
            pass
        del train_env, buffer
        if teacher_modules:
            for _tm in list(teacher_modules.values()):
                del _tm
            teacher_modules.clear()
        if aux_buffers:
            aux_buffers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(colored(
            f'>>> TASK {task_idx + 1} ({task_name}) done @g={global_step}.',
            'green', attrs=['bold'],
        ))
        print()

    if wandb_run is not None:
        wandb_run.finish()

    print('=' * 64)
    print(colored('>>> ALL TASKS COMPLETED.', 'green', attrs=['bold']))
    print(f'>>> Final global step: {global_step}')
    print('=' * 64)


# ==========================================================================
# Entry point — naive static-K sequential
# ==========================================================================

def main(args):
    run_continual_sequential(args, progressive=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Naive sequential TD-MPC2 continual learning '
                    '(DreamerV3 split, static full-K PRISM-MoE).',
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
                   default='Metaworld_Tdmpc2_Sequential_PrismNaive')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None,
                   help="Existing wandb run id to resume with resume='allow'.")
    p.add_argument('--wandb-group', type=str, default=None,
                   help='Wandb group; sweep launcher passes the taskset name.')
    p.add_argument('--wandb-tags', nargs='*', default=None,
                   help='Wandb tags; sweep launcher fills with '
                        '[method, taskset, seed, date, sweep_id, variant…].')
    p.add_argument('--exp-name', type=str, default='seq_naive_tdmpc2')
    p.add_argument('--from-checkpoint', type=str, default=None,
                   help='Directory containing backbone.pt (+ optional '
                        'task_modules.pt) to seed task 0.')

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

    # PRISM-WM MoE — the naive baseline always uses the DreamerV3 split
    # (shared PrismBackbone + per-task TaskModules) with a STATIC full-K MoE:
    # all `--num-experts` experts are active + trainable for every task; task
    # boundaries only swap the per-task heads. This isolates progmoe's CL
    # machinery (progressive masking + freezing) from the shared split.
    p.add_argument('--num-experts', type=int, default=8,
                   help='Number of experts in the (static, full-K) dynamics '
                        'MoE. All experts are always active and trainable.')

    # Gram-Schmidt orthogonalization across the K experts (PRISM-WM's
    # `OrthogonalLayer1D`). Off by default — matches config.yaml.
    p.add_argument('--use-orthogonal', dest='use_orthogonal',
                   action='store_true',
                   help='Enable Gram-Schmidt orthogonalization of the expert '
                        'feature stack inside the dynamics MoE.')
    p.add_argument('--no-use-orthogonal', dest='use_orthogonal',
                   action='store_false')
    p.set_defaults(use_orthogonal=True)

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
    # --pretrained-encoder). dino_* knobs (pool, input size, amp, weights
    # path) come from config.yaml defaults.
    p.add_argument('--encoder-type', type=str, default='default',
                   choices=['default', 'dino'],
                   help="rgb encoder: 'default' CNN or frozen 'dino' "
                        '(DINOv2 + trainable head; forces compile=false).')

    main(p.parse_args())
