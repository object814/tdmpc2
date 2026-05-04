"""Sequential TD-MPC2 training with PackNet (paper-faithful) and cross-task eval.

Mirrors `sequential_train.py` but:
  1. Uses `PackNetTDMPC2` (a `TDMPC2` subclass that masks gradients on
     frozen weights and re-zeros pruned weights after each step, in BOTH
     the world-model optimizer and the policy optimizer).
  2. After each task except the last, prunes low-magnitude free weights
     of all `ndim >= 2` agent params, retrains the surviving weights for
     a fraction of the original task budget, and freezes them. 1-D
     parameters (biases, norm) are frozen after task 1's retrain.
  3. Stores per-task eval masks so previous tasks can be evaluated using
     their own task subnet (other tasks' weights zeroed out at eval).
  4. Persists PackNet state to `<task_logdir>/packnet_state_task{N}.pt`.

Compile must be off: gradient masking is dynamic Python and incompatible
with `torch.compile(_update, "reduce-overhead")`'s cudagraph capture.

Caveat (TD-MPC2-specific): PackNet doesn't replay or penalise — it
explicitly carves capacity per task. This is the cleanest match for the
monolithic TD-MPC2 architecture; gradient flow on each new task is
restricted to free weights, so previous-task weights cannot drift.
Target-Q networks Polyak-track the (now partially frozen) online Q,
which is fine.

Example:
    python sequential_train_packnet.py \\
        --tasks metaworld_drawer-open-v3 metaworld_pick-place-v3 \\
        --task-steps 200000 500000 \\
        --logdir ../logdir/seq_tdmpc2_packnet/run0 \\
        --packnet-prune-ratio 0.75 \\
        --packnet-retrain-ratio 0.1 \\
        --wandb-project Metaworld_Tdmpc2_Sequential_PackNet
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
import sys
import time as _time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict
from termcolor import colored
from tqdm.auto import tqdm

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

from packnet_training.packnet import PackNetManager


CONFIG_PATH = _PARENT / 'config.yaml'

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


# ==========================================================================
# PackNet-aware TD-MPC2 subclass
# ==========================================================================
# Override `_update` and `update_pi` to inject PackNet's gradient/weight
# masking around BOTH `agent.optim.step()` and `agent.pi_optim.step()`.
# Everything else (target-Q polyak, scale.update, loss math) is unchanged.

class PackNetTDMPC2(TDMPC2):
    """TD-MPC2 + PackNet gradient/weight masking."""

    def __init__(self, cfg, packnet_manager: PackNetManager = None):
        if getattr(cfg, 'compile', False):
            print(colored(
                '>>> PackNet: forcing cfg.compile=False (gradient masking is '
                'incompatible with cudagraph capture)', 'yellow',
            ))
            cfg.compile = False
        super().__init__(cfg)
        self.packnet_manager = packnet_manager

    # ------------------------------------------------------------------
    # Override _update with PackNet hooks
    # ------------------------------------------------------------------
    def _update(self, obs, action, reward, terminated, task=None):
        # ---- world-model loss (same math as parent _update) --------------
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

        # ---- world-model backward + masked step --------------------------
        total_loss.backward()
        if self.packnet_manager is not None:
            self.packnet_manager.apply_gradient_mask(self)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.grad_clip_norm,
        )
        self.optim.step()
        if self.packnet_manager is not None:
            self.packnet_manager.apply_weight_mask(self)
        self.optim.zero_grad(set_to_none=True)

        # ---- policy update (with PackNet hooks) --------------------------
        pi_info = self._update_pi_with_packnet(zs.detach(), task)

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
        })
        if self.cfg.episodic:
            info.update(tdmpc_math.termination_statistics(
                torch.sigmoid(termination_pred[-1]), terminated[-1],
            ))
        info.update(pi_info)
        return info.detach().mean()

    def _update_pi_with_packnet(self, zs, task):
        """Mirror of `update_pi` with PackNet masking around the step."""
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

        pi_loss.backward()
        if self.packnet_manager is not None:
            self.packnet_manager.apply_gradient_mask(self)
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.cfg.grad_clip_norm,
        )
        self.pi_optim.step()
        if self.packnet_manager is not None:
            self.packnet_manager.apply_weight_mask(self)
        self.pi_optim.zero_grad(set_to_none=True)

        return TensorDict({
            "pi_loss": pi_loss,
            "pi_grad_norm": pi_grad_norm,
            "pi_entropy": info["entropy"],
            "pi_scaled_entropy": info["scaled_entropy"],
            "pi_scale": self.scale.value,
        })


# ==========================================================================
# Per-task online training (mirrors sequential_train.train_one_task)
# ==========================================================================

def train_one_task(cfg, agent, env, buffer, logger,
                   global_step_start, eval_fn, skip_pretrain,
                   task_idx=0, num_tasks=1, task_name='',
                   step_budget=None, run_eval_save=True,
                   pbar_desc=None):
    """TD-MPC2 online training with optional override of step budget.

    `step_budget`: when None, run for `cfg.steps`; otherwise run for the
                   given number of internal steps. Used by the retrain
                   phase to run a short additional training segment.
    `run_eval_save`: when False, suppress periodic eval and checkpointing
                     (used during retrain). The agent is still trained.
    `pbar_desc`: override progress-bar description.
    """
    logger.global_step_fn = lambda s: global_step_start + s
    total_steps = cfg.steps if step_budget is None else step_budget

    pbar = tqdm(
        total=total_steps,
        desc=pbar_desc or f'T{task_idx + 1}/{num_tasks} {task_name}',
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
        while local_step <= total_steps:
            if run_eval_save and local_step % cfg.eval_freq == 0:
                eval_next = True

            if (run_eval_save and cfg.save_freq > 0 and local_step > 0
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
                    ep_idx = buffer.add(torch.cat(tds))

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
            if local_step <= total_steps:
                pbar.update(1)
    finally:
        pbar.close()

    return local_step


# ==========================================================================
# Cross-task evaluation closure (with PackNet weight masking for past tasks)
# ==========================================================================

def make_eval_fn(agent, task_idx, tasks, task_cfgs, logger, packnet_manager,
                 get_eval_env, save_video):
    """Build an evaluation closure that, for each previous task `j`, masks
    the agent's weights with `task_masks[j]` before running rollout, then
    restores the live weights. The current task is evaluated unmasked
    (its mask hasn't been finalised yet)."""
    def eval_fn(agent_, local_step, gstep):
        for j in range(task_idx + 1):
            is_current = (j == task_idx)
            env_j = get_eval_env(j)
            video_rec = logger.video if (is_current and logger.video) else None
            if video_rec is not None:
                video_rec.set_prefix(
                    f'eval/task{j + 1}_{tasks[j]}/videos/eval_video'
                )

            apply_mask = (
                (not is_current)
                and packnet_manager is not None
                and j in packnet_manager.task_masks
            )
            saved = None
            if apply_mask:
                saved = packnet_manager.save_agent_weights(agent_)
                packnet_manager.apply_eval_mask(agent_, j)

            try:
                metrics = eval_on_env(
                    agent_, env_j, task_cfgs[task_idx].eval_episodes,
                    video_recorder=video_rec,
                    video_step=gstep if video_rec else None,
                )
                metrics['step'] = local_step
                logger.log_eval(j, tasks[j], metrics, gstep)
            finally:
                if saved is not None:
                    packnet_manager.restore_agent_weights(agent_, saved)
                    del saved
    return eval_fn


# ==========================================================================
# Main sequential PackNet loop
# ==========================================================================

def main(args):
    assert torch.cuda.is_available(), 'TD-MPC2 requires CUDA.'

    num_tasks = len(args.tasks)
    if len(args.task_steps) != num_tasks:
        raise ValueError('--task-steps must have one value per task.')

    base_logdir = Path(args.logdir).expanduser().resolve()
    base_logdir.mkdir(parents=True, exist_ok=True)

    # ---- shared overrides (force compile=False) ----------------------------
    shared = dict(
        model_size=args.model_size,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
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
    )

    task_cfgs = []
    for i in range(num_tasks):
        task_logdir = base_logdir / f'task{i + 1}_{args.tasks[i]}'
        cfg = build_task_cfg(
            args.tasks[i], args.task_steps[i], task_logdir,
            shared, args.seed,
        )
        task_cfgs.append(cfg)

    set_seed(args.seed)

    # ---- PackNet manager (single instance for entire run) ------------------
    packnet_manager = PackNetManager(prune_ratio=args.packnet_prune_ratio)

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

    # [PackNet] Resume: reload latest packnet state.
    if resume_from > 0:
        for j in range(resume_from - 1, -1, -1):
            pn_file = (base_logdir / f'task{j + 1}_{args.tasks[j]}'
                       / f'packnet_state_task{j + 1}.pt')
            if pn_file.exists():
                print(colored(f'>>> PackNet: loading state from {pn_file}',
                              'cyan'))
                pn_state = torch.load(str(pn_file), map_location='cpu',
                                      weights_only=False)
                packnet_manager.load_state_dict(pn_state)
                print(colored(
                    f'>>> PackNet: restored {packnet_manager.num_tasks_packed} '
                    f'packed task(s)', 'cyan',
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
            f'seq_tdmpc2_packnet_{"-".join(args.tasks)}_s{args.seed}'
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
                'packnet_prune_ratio': args.packnet_prune_ratio,
                'packnet_retrain_ratio': args.packnet_retrain_ratio,
                **shared,
            },
            tags=['sequential', 'tdmpc2', 'packnet'] + list(args.tasks),
        )
        try:
            save_wandb_run_id(base_logdir, wandb_run.id)
        except Exception as e:
            print(colored(f'[warn] could not persist wandb run id: {e}', 'red'))

    # ---- Plan summary ------------------------------------------------------
    print('=' * 64)
    print(colored('>>> SEQUENTIAL TD-MPC2 PackNet TRAINING', 'cyan',
                  attrs=['bold']))
    print('=' * 64)
    for i, (t, s) in enumerate(zip(args.tasks, args.task_steps)):
        mark = '  <-- resume here' if i == resume_from else ''
        print(f'  Task {i + 1}: {t}  steps={s}{mark}')
    print(f'  Logdir:                {base_logdir}')
    print(f'  Seed:                  {args.seed}')
    print(f'  PackNet prune_ratio:   {args.packnet_prune_ratio}')
    print(f'  PackNet retrain_ratio: {args.packnet_retrain_ratio} '
          f'(fraction of task steps)')
    print(f'  Compile:               FORCED OFF (PackNet requirement)')
    print(f'  Wandb project:         {args.wandb_project}')
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
        print(f'    internal steps:    {cfg.steps}  '
              f'(action_repeat={cfg.action_repeat} -> '
              f'{cfg.steps * cfg.action_repeat} env frames)')
        print(f'    global step start: {global_step}')
        print(f'    PackNet packed:    {packnet_manager.num_tasks_packed} '
              f'task(s)')
        print('=' * 64)

        save_progress(
            base_logdir, task_idx, task_name=task_name,
            global_step_at_start=global_step,
        )

        train_env = make_env(cfg)
        buffer = Buffer(cfg)

        logger = SequentialLogger(
            cfg, task_idx, num_tasks, task_name, wandb_run,
            save_video=args.save_video,
        )

        # Build the PackNet-aware agent.
        agent = PackNetTDMPC2(cfg, packnet_manager=packnet_manager)
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

        # [PackNet] Move masks to device + register live params.
        packnet_manager.to_device(agent.device)
        packnet_manager.register_agent_params(agent)
        if packnet_manager.num_tasks_packed > 0:
            n_frozen = sum(
                v.sum().item() for v in packnet_manager.frozen_mask.values()
            )
            n_total_w = sum(
                p.numel() for n, p in agent.named_parameters()
                if packnet_manager._is_prunable(n, p)
            )
            print(colored(
                f'>>> PackNet: {int(n_frozen):,}/{n_total_w:,} weight params '
                f'frozen from {packnet_manager.num_tasks_packed} previous '
                f'task(s); shared_frozen={packnet_manager._shared_params_frozen}',
                'cyan',
            ))

        eval_fn = make_eval_fn(
            agent, task_idx, args.tasks, task_cfgs, logger,
            packnet_manager, _get_eval_env, args.save_video,
        )

        # ---- Train -------------------------------------------------------
        train_one_task(
            cfg, agent, train_env, buffer, logger,
            global_step_start=global_step,
            eval_fn=eval_fn,
            skip_pretrain=args.skip_pretrain or (task_idx > 0),
            task_idx=task_idx,
            num_tasks=num_tasks,
            task_name=task_name,
        )

        # Final cross-task eval + checkpoint save (BEFORE pruning).
        eval_fn(agent, cfg.steps, global_step + cfg.steps)
        logger.save_agent(agent, identifier='final')

        # ============================================================
        # [PackNet] Prune -> retrain -> freeze (skip on the last task)
        # ============================================================
        if task_idx < num_tasks - 1:
            print(colored(
                f'>>> PackNet: pruning task {task_idx + 1} ({task_name})...',
                'cyan',
            ))
            t0 = _time.time()

            # Step 1: prune
            task_mask = packnet_manager.prune(agent, task_idx)

            # Step 2: retrain with frozen + pruned weights masked.
            retrain_steps = int(cfg.steps * args.packnet_retrain_ratio)
            if retrain_steps > 0:
                print(colored(
                    f'>>> PackNet: retraining for {retrain_steps} internal '
                    f'steps ({args.packnet_retrain_ratio*100:.0f}% of '
                    f'{cfg.steps})...',
                    'cyan',
                ))
                packnet_manager.start_retrain(task_mask)
                # Run a brief additional training segment with eval/save
                # disabled. Reuse the existing env + buffer.
                retrain_global_start = global_step + cfg.steps
                train_one_task(
                    cfg, agent, train_env, buffer, logger,
                    global_step_start=retrain_global_start,
                    eval_fn=eval_fn,  # unused with run_eval_save=False
                    skip_pretrain=True,
                    task_idx=task_idx,
                    num_tasks=num_tasks,
                    task_name=task_name,
                    step_budget=retrain_steps,
                    run_eval_save=False,
                    pbar_desc=f'T{task_idx + 1} PackNet retrain',
                )
                packnet_manager.end_retrain()

            # Step 3: freeze surviving weights + freeze 1-D shared params.
            packnet_manager.freeze_task(task_mask, task_idx)

            elapsed = _time.time() - t0
            print(colored(
                f'>>> PackNet: pruned + retrained + frozen task '
                f'{task_idx + 1} in {elapsed:.1f}s', 'cyan',
            ))

            if wandb_run is not None:
                n_frozen_w = sum(
                    v.sum().item() for v in packnet_manager.frozen_mask.values()
                )
                n_total_w = sum(
                    v.numel() for v in packnet_manager.frozen_mask.values()
                )
                wandb_run.log({
                    'packnet/frozen_ratio':
                        n_frozen_w / max(n_total_w, 1),
                    'packnet/num_tasks_packed':
                        packnet_manager.num_tasks_packed,
                    'packnet/prune_retrain_time_s': elapsed,
                    'global_step': global_step + cfg.steps,
                }, step=global_step + cfg.steps)

            # Persist PackNet state for resume.
            pn_path = (task_logdir
                       / f'packnet_state_task{task_idx + 1}.pt')
            torch.save(packnet_manager.state_dict(), str(pn_path))
            print(colored(f'>>> PackNet: saved state -> {pn_path}', 'cyan'))

            # Save agent again so the on-disk checkpoint reflects the
            # post-prune-retrain weights.
            logger.save_agent(agent, identifier='final')
        else:
            # Last task: store a "full active" mask for eval consistency.
            task_mask = {}
            for name, param in agent.named_parameters():
                if packnet_manager._is_prunable(name, param):
                    task_mask[name] = torch.ones_like(param.data)
            packnet_manager.task_masks[task_idx] = task_mask
            print(colored(
                f'>>> PackNet: last task {task_idx + 1} — stored full mask '
                f'(no pruning).', 'cyan',
            ))
            pn_path = (task_logdir
                       / f'packnet_state_task{task_idx + 1}.pt')
            torch.save(packnet_manager.state_dict(), str(pn_path))

        # The final checkpoint to hand off to the next task is the (possibly
        # post-retrain) agent on disk.
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
        description='Sequential TD-MPC2 training with PackNet.',
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
                   default='Metaworld_Tdmpc2_Sequential_PackNet')
    p.add_argument('--wandb-run-name', type=str, default=None)
    p.add_argument('--wandb-run-id', type=str, default=None)
    p.add_argument('--exp-name', type=str, default='seq_tdmpc2_packnet')
    p.add_argument('--from-checkpoint', type=str, default=None)

    p.add_argument('--seed', type=int, default=1)

    # TDMPC2 hyperparams
    p.add_argument('--model-size', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--buffer-size', type=int, default=50000)
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

    # PackNet knobs
    p.add_argument(
        '--packnet-prune-ratio', type=float, default=0.75,
        help='Fraction of free weights to prune after each task (default: 0.75). '
             'Higher = more aggressive pruning, more capacity for future tasks.',
    )
    p.add_argument(
        '--packnet-retrain-ratio', type=float, default=0.1,
        help='Fraction of cfg.steps used for the retrain phase after pruning '
             '(default: 0.1 = 10%%). Set to 0 to skip retraining.',
    )

    main(p.parse_args())
