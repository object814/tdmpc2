"""Rollout tier: does the routing mass on prior experts actually do work?

The router heatmap shows *intent*. It cannot show that the prior experts are
load-bearing: a large coefficient on an old expert is still compatible with
that expert contributing nothing useful. This module answers the causal
question by intervening on the routing and measuring what breaks.

Two levels, deliberately split by cost:

  open-loop (cheap)
      Collect episodes ONCE with the unablated final-task agent and cache the
      encoded latents. The encoder is untouched by any routing intervention,
      so the ground-truth latent sequence is shared across every condition --
      only the dynamics rollout is recomputed. That makes leave-one-expert-out
      over all K experts essentially free, which is what produces the causal
      per-expert heatmap that sits next to the router heatmap.

  closed-loop (expensive)
      Re-run the environment under MPC with the intervention active and
      measure success rate and return. This is the metric a reviewer will
      actually ask for, but it costs a full eval per condition, so only the
      headline conditions are run by default.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from common import math as tdmath
from .interventions import routing_intervention


# =======================================================================
# Agent construction from checkpoints
# =======================================================================

def load_final_agent(run, defaults: dict, geom: dict, *, task_idx=None,
                     verbose=True):
    """Rebuild the agent as it stood at the end of `task_idx` (default: the
    final task) and return `(agent, cfg, env)`.

    `make_env` must run before `ContinualTDMPC2(cfg)` because it fills in
    `cfg.obs_shape` / `cfg.action_dim` / `cfg.episode_length`, which size the
    encoder and the MoE.
    """
    from envs import make_env
    from progmoe_training.continual_tdmpc2 import ContinualTDMPC2

    t = run.final_task_idx if task_idx is None else int(task_idx)
    cfg = run.build_cfg(t, defaults=defaults, geom=geom)
    env = make_env(cfg)

    agent = ContinualTDMPC2(cfg)
    agent.load_backbone(run.backbone_path(t), load_optim=False)
    agent.load_task_modules(run.task_modules_path(t), load_optim=False)

    # `_current_task_idx` is a plain Python attribute and does not survive a
    # state_dict round-trip (the one-hot buffer does). Sync both explicitly.
    agent.set_current_task(t)
    agent.eval_mode()

    block = agent.backbone._dynamics
    expected_K = (t + 1) * geom['K_per_task']
    assert block._active_K == expected_K, (
        f'{run.name}: checkpoint active_K={block._active_K}, expected '
        f'{expected_K} for task {t}')
    if verbose:
        print(f'[{run.name}] loaded task {t + 1}/{run.num_tasks} '
              f'({run.short_tasks()[t]}): active_K={block._active_K}, '
              f'frozen_K={block._frozen_K}, tau={block.tau:.4f}')
    return agent, cfg, env


# =======================================================================
# Episode collection (encoded once, reused by every open-loop condition)
# =======================================================================

@torch.no_grad()
def collect_episodes(agent, env, episodes: int, *, seed=0, verbose=True):
    """Roll the unablated agent in the env and cache encoded latents.

    Returns a list of dicts with:
        z          [T+1, latent_dim]  encoded observations
        action     [T, action_dim]
        reward     [T]
        success    float
        ep_reward  float
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = []
    for i in range(episodes):
        t0 = time.time()
        obs = env.reset()
        done, t = False, 0
        obs_list, act_list, rew_list = [obs], [], []
        info = {}
        while not done:
            action = agent.act(obs, t0=(t == 0), eval_mode=True)
            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            act_list.append(action)
            rew_list.append(float(reward))
            t += 1

        z = torch.stack([_encode_one(agent, o) for o in obs_list])  # [T+1, D]
        ep = dict(
            z=z.cpu(),
            action=torch.stack(act_list).cpu(),
            reward=torch.tensor(rew_list),
            success=float(info.get('success', 0.0)),
            ep_reward=float(np.sum(rew_list)),
            length=t,
        )
        out.append(ep)
        if verbose:
            print(f'  episode {i + 1}/{episodes}: len={t} '
                  f'R={ep["ep_reward"]:.1f} success={ep["success"]:.0f} '
                  f'({time.time() - t0:.1f}s)')
    return out


@torch.no_grad()
def _encode_one(agent, obs):
    dev = agent.device
    if isinstance(obs, dict):
        obs = {k: v.to(dev).unsqueeze(0) for k, v in obs.items()}
    elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
        obs = obs.to(dev).unsqueeze(0)
    else:
        obs = obs.to(dev).unsqueeze(0)
    return agent.backbone.encode(obs)[0]


# =======================================================================
# Open-loop: multi-step latent prediction error under an intervention
# =======================================================================

@torch.no_grad()
def openloop_error(agent, episodes, horizon: int, *, weight_fn=None,
                   max_starts=None, reward_head=True):
    """Roll the dynamics `horizon` steps from every start index and compare
    against the cached ground-truth latents.

    Returns per-horizon MSE and cosine similarity, plus (optionally) the
    reward-prediction MAE through the final task's frozen reward head.
    All predictions come from the same cached `z`, so the only thing that
    varies across conditions is the MoE mixing.
    """
    block = agent.backbone._dynamics
    dev = agent.device
    ctx = (routing_intervention(block, weight_fn) if weight_fn is not None
           else _null_ctx())

    sq = np.zeros(horizon)
    cos = np.zeros(horizon)
    n = np.zeros(horizon)
    rew_abs, rew_n = 0.0, 0

    with ctx:
        for ep in episodes:
            z = ep['z'].to(dev)
            a = ep['action'].to(dev)
            r = ep['reward'].to(dev)
            T = a.shape[0]
            starts = np.arange(T)
            if max_starts is not None and len(starts) > max_starts:
                starts = np.unique(
                    np.linspace(0, T - 1, max_starts).astype(int))
            if len(starts) == 0:
                continue

            # `cur` is the index of the action about to be applied; after the
            # step it advances by one so horizon h predicts z[start + h + 1].
            cur = torch.as_tensor(starts, device=dev, dtype=torch.long)
            zc = z[cur]                                        # [S, D]
            for h in range(horizon):
                keep = cur < T
                if not bool(keep.any()):
                    break
                cur, zc = cur[keep], zc[keep]
                ac = a[cur]
                zc = agent.backbone.next(zc, ac)
                tgt = z[cur + 1]
                sq[h] += float(((zc - tgt) ** 2).sum(-1).sum())
                cos[h] += float(torch.nn.functional.cosine_similarity(
                    zc, tgt, dim=-1).sum())
                n[h] += int(cur.numel())

                if reward_head:
                    logits = agent.task_modules.reward(zc, ac)
                    pred = tdmath.two_hot_inv(logits, agent.cfg).squeeze(-1)
                    rew_abs += float((pred - r[cur]).abs().sum())
                    rew_n += int(cur.numel())
                cur = cur + 1

    n = np.maximum(n, 1)
    return dict(
        mse_per_h=(sq / n).tolist(),
        cos_per_h=(cos / n).tolist(),
        mse=float((sq / n).mean()),
        mse_h1=float(sq[0] / n[0]),
        mse_hlast=float(sq[horizon - 1] / n[horizon - 1]),
        cos=float((cos / n).mean()),
        reward_mae=(rew_abs / max(rew_n, 1)) if reward_head else float('nan'),
    )


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =======================================================================
# Closed-loop: MPC in the environment under an intervention
# =======================================================================

@torch.no_grad()
def closedloop_eval(agent, env, episodes: int, *, weight_fn=None, seed=0,
                    verbose=True, tag=''):
    """Run MPC in the env with the intervention active. Returns mean return,
    success rate and episode length."""
    block = agent.backbone._dynamics
    ctx = (routing_intervention(block, weight_fn) if weight_fn is not None
           else _null_ctx())
    torch.manual_seed(seed)
    np.random.seed(seed)
    rewards, successes, lengths = [], [], []
    with ctx:
        for i in range(episodes):
            obs, done, ep_r, t = env.reset(), False, 0.0, 0
            info = {}
            while not done:
                action = agent.act(obs, t0=(t == 0), eval_mode=True)
                obs, reward, done, info = env.step(action)
                ep_r += float(reward)
                t += 1
            rewards.append(ep_r)
            successes.append(float(info.get('success', 0.0)))
            lengths.append(t)
    res = dict(
        episode_reward=float(np.mean(rewards)),
        episode_success=float(np.mean(successes)),
        episode_length=float(np.mean(lengths)),
        episode_reward_std=float(np.std(rewards)),
        n_episodes=episodes,
    )
    if verbose:
        print(f'  [{tag}] R={res["episode_reward"]:.1f} '
              f'success={res["episode_success"]:.2f}')
    return res
