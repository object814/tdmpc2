"""Sequential TD-MPC2 training with EWC (Elastic Weight Consolidation).

Mirrors `sequential_train.py` but:
  1. Uses `EWCTDMPC2` (a `TDMPC2` subclass that adds an EWC penalty to both
     the world-model loss and the policy loss inside `_update`).
  2. After each task except the last, computes a diagonal Fisher matrix
     over **all trainable agent parameters** (encoder, dynamics, reward,
     termination, Q-ensemble, policy) and consolidates a θ* snapshot.
  3. Persists EWC state to `<task_logdir>/ewc_state_task{N}.pt` and
     reloads it on resume.

Compile must be off — the per-step EWC penalty is computed on the Python
side and adding it to a `torch.compile`'d step is not supported. The script
forces `cfg.compile = False` before constructing the agent.

Caveat (TD-MPC2-specific): TD-MPC2 has no task-agnostic shaping signal for
the encoder/dynamics (no decoder/reconstruction). Fisher therefore captures
parameters important for predicting the **previous task's reward and value**,
and protecting them via EWC directly conflicts with learning the new
task's reward and value. Expect EWC to be more constraining on TD-MPC2
than on Dreamer; tune `--ewc-lambda` accordingly.

Example:
    python sequential_train_ewc.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2_ewc/run0 \\
        --ewc-lambda 1000.0 \\
        --ewc-fisher-batches 50 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_EWC
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
import shutil
import sys
import time as _time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from termcolor import colored

# Add tdmpc2/ to sys.path to reuse helpers from the naive sequential script.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from common import math as tdmpc_math
from envs import make_env
from tdmpc2 import TDMPC2

from sequential_train import (
    build_task_cfg,
    save_progress,
    load_progress,
    save_wandb_run_id,
    SequentialLogger,
    eval_on_env,
    _to_td,
)

from ewc_training.ewc import EWCManager, _compute_world_model_loss, _compute_pi_loss


CONFIG_PATH = _PARENT / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# EWC-aware TD-MPC2 subclass
# ==========================================================================
# Override `_update` to add `λ · Σ F · (θ - θ*)²` to both the world-model
# loss and the policy loss before each backward pass. Everything else
# (target-Q polyak, scale.update, optimizer steps) is unchanged.
#
# We deliberately reproduce the loss math here rather than calling the
# helpers in `ewc.py`, because we want to keep the original control-flow
# (target-Q soft-update, scale.update inside update_pi) intact. The two
# code paths (Fisher-time helpers vs. training-time _update) compute the
# same loss expressions; if upstream `tdmpc2.py` changes its math, both
# need to be updated together.

class EWCTDMPC2(TDMPC2):
    """TD-MPC2 + EWC penalty injected into both backward passes.

    Attach an `EWCManager` via `agent.ewc_manager = manager` after
    instantiation (or pass through the constructor — both work).
    """

    def __init__(self, cfg, ewc_manager: EWCManager = None):
        # Force compile off — the EWC penalty is dynamic Python and is not
        # compatible with TD-MPC2's `torch.compile(_update, "reduce-overhead")`
        # cudagraph capture.
        if getattr(cfg, 'compile', False):
            print(colored(
                '>>> EWC: forcing cfg.compile=False (penalty is incompatible '
                'with cudagraph capture)', 'yellow',
            ))
            cfg.compile = False
        super().__init__(cfg)
        self.ewc_manager = ewc_manager

    # ------------------------------------------------------------------
    # Override _update with EWC penalty injection
    # ------------------------------------------------------------------
    def _update(self, obs, action, reward, terminated, task=None):
        # ---- world-model loss (mirrors parent _update body) --------------
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, terminated, task)

        self.model.train()

        zs = torch.empty(
            self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim,
            device=self.device,
        )
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action, task)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.cfg.rho ** t
            zs[t + 1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        if self.cfg.episodic:
            termination_pred = self.model.termination(zs[1:], task, unnormalized=True)

        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + tdmpc_math.soft_ce(
                rew_pred_unbind, rew_unbind, self.cfg
            ).mean() * self.cfg.rho ** t
            for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
                value_loss = value_loss + tdmpc_math.soft_ce(
                    qs_unbind_unbind, td_targets_unbind, self.cfg
                ).mean() * self.cfg.rho ** t

        consistency_loss = consistency_loss / self.cfg.horizon
        reward_loss = reward_loss / self.cfg.horizon
        if self.cfg.episodic:
            termination_loss = F.binary_cross_entropy_with_logits(
                termination_pred, terminated
            )
        else:
            termination_loss = 0.
        value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)

        total_loss = (
            self.cfg.consistency_coef * consistency_loss
            + self.cfg.reward_coef * reward_loss
            + self.cfg.termination_coef * termination_loss
            + self.cfg.value_coef * value_loss
        )

        # ---- EWC penalty (world-model backward) --------------------------
        ewc_penalty_world = torch.tensor(0.0, device=self.device)
        if self.ewc_manager is not None and self.ewc_manager.num_tasks_consolidated > 0:
            ewc_penalty_world = self.ewc_manager.penalty(self)
            total_loss = total_loss + ewc_penalty_world

        # ---- world-model backward + step ---------------------------------
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm,
        )
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # ---- policy update with EWC penalty ------------------------------
        pi_info = self._update_pi_with_ewc(zs.detach(), task)

        # ---- target-Q soft update (unchanged) ----------------------------
        self.model.soft_update_target_Q()

        self.model.eval()
        info = TensorDict({
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "termination_loss": termination_loss,
            "total_loss": total_loss,
            "grad_norm": grad_norm,
            "ewc_penalty_world": ewc_penalty_world.detach(),
        })
        if self.cfg.episodic:
            info.update(tdmpc_math.termination_statistics(
                torch.sigmoid(termination_pred[-1]), terminated[-1],
            ))
        info.update(pi_info)
        return info.detach().mean()

    def _update_pi_with_ewc(self, zs, task):
        """Mirror of `update_pi` with an EWC penalty added before backward."""
        action, info = self.model.pi(zs, task)
        qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(
            self.cfg.rho, torch.arange(len(qs), device=self.device),
        )
        pi_loss = (
            -((self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho)
        ).mean()

        # Add the same EWC penalty: only π parameters that were tracked in
        # the Fisher pass receive a non-zero gradient contribution from this
        # backward call (the penalty is the same scalar; backward routes it
        # through the params that participate in this subgraph).
        ewc_penalty_pi = torch.tensor(0.0, device=self.device)
        if self.ewc_manager is not None and self.ewc_manager.num_tasks_consolidated > 0:
            ewc_penalty_pi = self.ewc_manager.penalty(self)
            pi_loss = pi_loss + ewc_penalty_pi

        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm,
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        return TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
            "ewc_penalty_pi": ewc_penalty_pi.detach(),
        })


# ==========================================================================
# Per-task online training (mirrors sequential_train.train_one_task)
# ==========================================================================

def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   task_idx=0, num_tasks=1, task_name='',
                   episode_dir=None, save_episodes=True,
                   buffer_cap=None, prune_every_n_eps=5):
    from tqdm.auto import tqdm

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
    start_time = _time.time()
    train_metrics = {}
    done = True
    eval_next = False
    tds = []
    obs = None
    info = None

    def _common():
        elapsed = _time.time() - start_time
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
                            print(colored(f'[seq_train_ewc] save_episode failed: {e}', 'red'))

                obs = env.reset()
                tds = [_to_td(env, obs)]

            if local_step > cfg.seed_steps:
                action = agent.act(obs, t0=(len(tds) == 1))
            else:
                action = env.rand_act()
            obs, reward, done, info = env.step(action)
            tds.append(_to_td(env, obs, action, reward, info['terminated']))

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
# Main sequential EWC loop
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
    # Force compile=False — EWC penalty is incompatible with cudagraph capture.
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
        enable_wandb=False,
        save_video=args.save_video,
        save_agent=True,
        exp_name=args.exp_name,
        compile=False,
        # Replay-buffer persistence + storage knobs (read by Buffer / trainer).
        save_episodes=args.save_episodes,
        episode_dir=None,                     # per-task dir is built inside the task loop
        buffer_storage_device=args.buffer_storage_device,
        prune_every_n_episodes=args.prune_every_n_episodes,
    )

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

    # ---- EWC manager (single instance for entire run) ----------------------
    ewc_manager = EWCManager(lambda_ewc=args.ewc_lambda)

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

    # [EWC] Resume: reload latest consolidated EWC state, if any.
    if resume_from > 0:
        for j in range(resume_from - 1, -1, -1):
            ewc_file = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                        / f'ewc_state_task{j + 1}.pt')
            if ewc_file.exists():
                print(colored(f'>>> EWC: loading state from {ewc_file}', 'cyan'))
                ewc_state = torch.load(str(ewc_file), map_location='cpu',
                                       weights_only=False)
                ewc_manager.load_state_dict(ewc_state)
                print(colored(
                    f'>>> EWC: restored {ewc_manager.num_tasks_consolidated} '
                    f'consolidated task(s)', 'cyan',
                ))
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
            f'seq_tdmpc2_ewc_{"-".join(args.tasks)}_s{args.seed}'
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
                'ewc_lambda': args.ewc_lambda,
                'ewc_fisher_batches': args.ewc_fisher_batches,
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'ewc'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 EWC TRAINING', 'cyan', attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir:               {base_logdir}')
    print(f'  Seed:                 {args.seed}')
    print(f'  EWC lambda:           {args.ewc_lambda}')
    print(f'  EWC Fisher batches:   {args.ewc_fisher_batches}')
    print(f'  Compile:              FORCED OFF (EWC requirement)')
    print(f'  Wandb project:        {args.wandb_project}')
    print('=' * 64)

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
        print(f'    internal steps:   {cfg.steps}  '
              f'(action_repeat={cfg.action_repeat} -> '
              f'{cfg.steps * cfg.action_repeat} env frames)')
        print(f'    global step start:{global_step}')
        print(f'    EWC consolidated: {ewc_manager.num_tasks_consolidated} '
              f'task(s)')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        # Buffer isolation: EWC keeps replay in-memory only — no cross-task
        # data is intentional. Once a previous task is fully completed its
        # `train_eps/` is no longer needed for resume, so delete it on entry
        # to a later task. (A partially-trained earlier task is protected so
        # its resume path stays intact.)
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

        # Build the EWC-aware agent.
        agent = EWCTDMPC2(cfg, ewc_manager=ewc_manager)
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

        # [EWC] Move stored Fishers/θ* to device + rebuild penalty cache.
        if ewc_manager.num_tasks_consolidated > 0:
            ewc_manager.to_device(agent.device)
            ewc_manager._rebuild_penalty_cache(agent)
            n_tracked = len(ewc_manager._tracked_param_names)
            print(colored(
                f'>>> EWC: rebuilt penalty cache on {agent.device} '
                f'({n_tracked} tracked param groups, '
                f'{ewc_manager.num_tasks_consolidated} consolidated tasks)',
                'cyan',
            ))

        # Cross-task evaluation closure ------------------------------------
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

        # Final cross-task eval + checkpoint save ------------------------
        eval_fn(agent, cfg.steps, global_step + cfg.steps)
        logger.save_agent(agent, identifier='final')
        prev_ckpt = logger.model_dir / 'final.pt'

        # ============================================================
        # [EWC] Compute Fisher + consolidate (skip on the last task)
        # ============================================================
        if task_idx < num_tasks - 1:
            print(colored(
                f'>>> EWC: computing Fisher for task {task_idx + 1} '
                f'({task_name})...', 'cyan',
            ))
            t0 = _time.time()
            importance, task_param = ewc_manager.compute_fisher(
                agent, buffer,
                num_batches=args.ewc_fisher_batches,
                device=agent.device,
            )
            ewc_manager.consolidate(agent, importance, task_param, task_idx)

            elapsed = _time.time() - t0
            n_params = sum(v.numel() for v in importance.values())
            fisher_mean = (
                sum(v.mean().item() for v in importance.values())
                / max(len(importance), 1)
            )
            fisher_max = max(v.max().item() for v in importance.values())
            print(colored(
                f'>>> EWC: consolidated task {task_idx + 1} in {elapsed:.1f}s '
                f'({len(importance)} param groups, {n_params:,} params, '
                f'mean F={fisher_mean:.2e}, max F={fisher_max:.2e})',
                'cyan',
            ))

            # Log Fisher stats to wandb under a continuous global_step.
            if wandb_run is not None:
                wandb_run.log({
                    'ewc/fisher_mean': fisher_mean,
                    'ewc/fisher_max': fisher_max,
                    'ewc/fisher_compute_time_s': elapsed,
                    'ewc/num_tasks_consolidated': ewc_manager.num_tasks_consolidated,
                    'global_step': global_step + cfg.steps,
                }, step=global_step + cfg.steps)

            # Persist EWC state for resume.
            ewc_path = task_logdir / f'ewc_state_task{task_idx + 1}.pt'
            torch.save(ewc_manager.state_dict(), str(ewc_path))
            print(colored(f'>>> EWC: saved state -> {ewc_path}', 'cyan'))

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
        description='Sequential TD-MPC2 training with EWC.',
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
                   default='Metaworld_Tdmpc2_Sequential_EWC')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2_ewc')
    p.add_argument('--from-checkpoint', type=str, default=None)

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

    # EWC knobs
    p.add_argument(
        '--ewc-lambda', type=float, default=1000.0,
        help='EWC regularisation strength (default: 1000.0). On TD-MPC2 the '
             'penalty is more constraining than on Dreamer (no task-agnostic '
             'shaping signal); sweep [100, 500, 1000, 5000] to calibrate.',
    )
    p.add_argument(
        '--ewc-fisher-batches', type=int, default=50,
        help='Number of buffer mini-batches used to estimate the diagonal '
             'Fisher at each task boundary (default: 50).',
    )

    main(p.parse_args())
