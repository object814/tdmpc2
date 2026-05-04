"""EWC (Elastic Weight Consolidation) manager for monolithic TD-MPC2.

Standard per-task EWC (Kirkpatrick et al. 2017):
  L_total = L_task(θ) + λ · Σ_t Σ_i F_{t,i} · (θ_i - θ*_{t,i})²

Unlike the Dreamer variant, TD-MPC2 has no shared/per-task split:
  - Encoder, dynamics, reward, termination, Q-ensemble, and policy all live
    in the single `TDMPC2` agent and share two optimizers (`agent.optim` for
    the world-model components, `agent.pi_optim` for the policy).
  - This manager therefore tracks **all trainable parameters** of the agent
    that participate in either optimizer's backward pass. The target-Q and
    detach-Q TensorDictParams clones are skipped — they are updated via
    `lerp_` / detach copy, never via gradient steps.

Fisher computation reproduces the loss math from `TDMPC2._update` and
`TDMPC2.update_pi` without stepping the optimizers. The duplication is
intentional: it keeps `tdmpc2.py` untouched so upstream changes can be
pulled in without merge conflicts in the EWC pipeline.

Penalty computation uses the same pre-flattened vectorised tensors as the
Dreamer EWC manager (fast O(N_params) per training step).
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from common import math as tdmpc_math


# ---------------------------------------------------------------------------
# Parameter selection — which params does EWC protect?
# ---------------------------------------------------------------------------

def _is_trainable_agent_param(name: str, param: torch.Tensor) -> bool:
    """Return True for every TDMPC2 parameter that gradient-trains.

    Excludes:
      * non-trainable buffers (handled automatically by `requires_grad`).
      * `_target_Qs` and `_detach_Qs` — TensorDictParams clones updated by
        Polyak averaging / detach copy, not by gradients.
    """
    if not param.requires_grad:
        return False
    # Normalise compile-wrapped names ("_orig_mod.") so the manager works
    # whether or not torch.compile is on. We disable compile in this script
    # but the helper is robust to either.
    norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
    if "_target_Qs" in norm or "_detach_Qs" in norm:
        return False
    return True


def collect_trainable_params(agent) -> List[Tuple[str, torch.Tensor]]:
    """Walk `agent.named_parameters()` and return EWC-trackable params."""
    return [
        (n, p) for n, p in agent.named_parameters()
        if _is_trainable_agent_param(n, p)
    ]


# ---------------------------------------------------------------------------
# Loss reproduction (mirrors TDMPC2._update / update_pi without stepping)
# ---------------------------------------------------------------------------

def _compute_world_model_loss(agent, obs, action, reward, terminated, task):
    """Re-run the world-model forward pass and return `total_loss` (and a
    detached `zs` for the policy pass).

    This mirrors the body of `TDMPC2._update` up to (but not including) the
    `total_loss.backward()` call. Keep this in sync with `tdmpc2.py` if the
    upstream loss formulation changes.
    """
    cfg = agent.cfg
    model = agent.model

    with torch.no_grad():
        next_z = model.encode(obs[1:], task)
        td_targets = agent._td_target(next_z, reward, terminated, task)

    model.train()

    zs = torch.empty(
        cfg.horizon + 1, cfg.batch_size, cfg.latent_dim, device=agent.device,
    )
    z = model.encode(obs[0], task)
    zs[0] = z
    consistency_loss = 0
    for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
        z = model.next(z, _action, task)
        consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * cfg.rho ** t
        zs[t + 1] = z

    _zs = zs[:-1]
    qs = model.Q(_zs, action, task, return_type='all')
    reward_preds = model.reward(_zs, action, task)
    if cfg.episodic:
        termination_pred = model.termination(zs[1:], task, unnormalized=True)

    reward_loss, value_loss = 0, 0
    for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
        zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
    ):
        reward_loss = reward_loss + tdmpc_math.soft_ce(
            rew_pred_unbind, rew_unbind, cfg
        ).mean() * cfg.rho ** t
        for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
            value_loss = value_loss + tdmpc_math.soft_ce(
                qs_unbind_unbind, td_targets_unbind, cfg
            ).mean() * cfg.rho ** t

    consistency_loss = consistency_loss / cfg.horizon
    reward_loss = reward_loss / cfg.horizon
    if cfg.episodic:
        termination_loss = F.binary_cross_entropy_with_logits(
            termination_pred, terminated
        )
    else:
        termination_loss = 0.
    value_loss = value_loss / (cfg.horizon * cfg.num_q)

    total_loss = (
        cfg.consistency_coef * consistency_loss
        + cfg.reward_coef * reward_loss
        + cfg.termination_coef * termination_loss
        + cfg.value_coef * value_loss
    )
    return total_loss, zs


def _compute_pi_loss(agent, zs_detached, task):
    """Re-run the policy forward pass and return `pi_loss`.

    Mirrors `TDMPC2.update_pi` up to (but not including) `pi_loss.backward()`.
    """
    cfg = agent.cfg
    model = agent.model
    action, info = model.pi(zs_detached, task)
    qs = model.Q(zs_detached, action, task, return_type='avg', detach=True)
    # Note: we deliberately do NOT call agent.scale.update(qs[0]) here — the
    # running scale is part of training state, not the loss-shape, and we
    # don't want a Fisher pass to perturb it.
    qs_scaled = agent.scale(qs)
    rho = torch.pow(cfg.rho, torch.arange(len(qs_scaled), device=agent.device))
    pi_loss = (
        -((cfg.entropy_coef * info["scaled_entropy"] + qs_scaled).mean(dim=(1, 2)) * rho)
    ).mean()
    return pi_loss


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class EWCManager:
    """EWC regularisation manager for monolithic TD-MPC2."""

    def __init__(self, lambda_ewc: float = 5000.0):
        self.lambda_ewc = lambda_ewc

        # {task_idx: {'importance': {name: F}, 'task_param': {name: θ*}}}
        self.regularization_terms: Dict[int, dict] = {}
        self.num_tasks_consolidated: int = 0

        # Pre-flattened cache for fast per-step penalty.
        self._penalty_cache_valid = False
        self._cached_fishers: List[torch.Tensor] = []
        self._cached_params: List[torch.Tensor] = []
        self._param_name_to_slice: Dict[str, Tuple[int, int]] = {}
        self._tracked_param_names: List[str] = []

    # ------------------------------------------------------------------
    # Fisher computation (called once per task boundary)
    # ------------------------------------------------------------------
    def compute_fisher(self, agent, buffer, num_batches: int = 50,
                       device: str = 'cuda'):
        """Compute diagonal Fisher across all trainable agent parameters.

        Each batch:
          1. Sample (obs, action, reward, terminated, task) from `buffer`.
          2. Run forward through both the world-model loss and the policy
             loss; sum them; backward.
          3. Accumulate squared per-parameter gradients (FP32) into `importance`.

        Both losses contribute jointly so that, e.g., the encoder's Fisher
        also captures its importance for the policy gradient (which only
        depends on `zs` indirectly via the encoder's outputs frozen via
        `zs.detach()` in the standard pipeline). To match the standard
        `_update` / `update_pi` separation we follow the same pattern:
          * world-model loss backward populates encoder/dynamics/reward/Q
            gradients (zs is in-graph here);
          * policy loss backward (on zs.detach()) populates only π gradients.
        We do them as **separate backward passes** so gradients are not
        double-counted, then square+accumulate after each pass.
        """
        importance = {}
        task_param = {}
        for name, param in collect_trainable_params(agent):
            importance[name] = torch.zeros_like(
                param.data, dtype=torch.float32, device=device,
            )
            task_param[name] = param.data.detach().to(
                device=device, dtype=torch.float32,
            ).clone()

        was_training = agent.model.training
        agent.model.train()
        # Make sure all relevant grads are enabled — TDMPC2 sets eval()
        # outside _update; we need grads for Fisher.
        for _, p in collect_trainable_params(agent):
            p.requires_grad_(True)

        for b in range(num_batches):
            obs, action, reward, terminated, task = buffer.sample()

            # ---- world-model loss ------------------------------------------
            agent.optim.zero_grad(set_to_none=True)
            agent.pi_optim.zero_grad(set_to_none=True)
            total_loss, zs = _compute_world_model_loss(
                agent, obs, action, reward, terminated, task,
            )
            total_loss.backward()
            for name, param in collect_trainable_params(agent):
                if param.grad is not None:
                    g = param.grad.detach().float()
                    importance[name] += (g ** 2) / num_batches

            # ---- policy loss (on zs.detach()) ------------------------------
            agent.optim.zero_grad(set_to_none=True)
            agent.pi_optim.zero_grad(set_to_none=True)
            pi_loss = _compute_pi_loss(agent, zs.detach(), task)
            pi_loss.backward()
            for name, param in collect_trainable_params(agent):
                if param.grad is not None:
                    g = param.grad.detach().float()
                    importance[name] += (g ** 2) / num_batches

            agent.optim.zero_grad(set_to_none=True)
            agent.pi_optim.zero_grad(set_to_none=True)

        if not was_training:
            agent.model.eval()

        return importance, task_param

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------
    def consolidate(self, agent, importance, task_param, task_idx: int):
        """Snapshot Fisher + θ* for `task_idx` and rebuild penalty cache."""
        self.regularization_terms[task_idx] = {
            'importance': {k: v.clone() for k, v in importance.items()},
            'task_param': {k: v.clone() for k, v in task_param.items()},
        }
        self.num_tasks_consolidated += 1
        self._rebuild_penalty_cache(agent)

    def _rebuild_penalty_cache(self, agent):
        """Pre-flatten Fisher diagonals and θ* for fast per-step penalty."""
        if self.num_tasks_consolidated == 0:
            self._penalty_cache_valid = False
            return

        device = next(iter(
            next(iter(self.regularization_terms.values()))['importance'].values()
        )).device

        first_reg = next(iter(self.regularization_terms.values()))
        self._tracked_param_names = []
        self._param_name_to_slice = {}
        offset = 0
        for name, param in collect_trainable_params(agent):
            if name in first_reg['importance']:
                n = param.numel()
                self._tracked_param_names.append(name)
                self._param_name_to_slice[name] = (offset, offset + n)
                offset += n

        total_params = offset
        self._cached_fishers = []
        self._cached_params = []
        for tid in sorted(self.regularization_terms.keys()):
            reg = self.regularization_terms[tid]
            fisher_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            params_flat = torch.zeros(total_params, dtype=torch.float32, device=device)
            for name in self._tracked_param_names:
                s, e = self._param_name_to_slice[name]
                fisher_flat[s:e] = reg['importance'][name].flatten()
                params_flat[s:e] = reg['task_param'][name].flatten()
            self._cached_fishers.append(fisher_flat)
            self._cached_params.append(params_flat)

        self._penalty_cache_valid = True

    # ------------------------------------------------------------------
    # Per-step penalty (called in the training loop)
    # ------------------------------------------------------------------
    def penalty(self, agent) -> torch.Tensor:
        """Compute λ · Σ_t Σ_i F · (θ - θ*)² over all tracked params.

        Returns a scalar tensor on the agent's device. Returns 0 if no
        tasks have been consolidated yet.
        """
        device = agent.device
        if not self._penalty_cache_valid or self.num_tasks_consolidated == 0:
            return torch.tensor(0.0, device=device)

        # Build name -> param lookup once per call. Names are normalised so
        # this works whether or not torch.compile is on.
        param_dict = {}
        for n, p in agent.named_parameters():
            norm = n.replace("._orig_mod.", ".").replace("_orig_mod.", "")
            param_dict[norm] = p

        reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for name in self._tracked_param_names:
            norm = name.replace("._orig_mod.", ".").replace("_orig_mod.", "")
            if norm not in param_dict:
                continue
            s, e = self._param_name_to_slice[name]
            param_flat = param_dict[norm].flatten().float()
            for fisher_flat, params_flat in zip(self._cached_fishers, self._cached_params):
                diff = param_flat - params_flat[s:e]
                reg_loss = reg_loss + (fisher_flat[s:e] * diff ** 2).sum()
        return self.lambda_ewc * reg_loss

    # ------------------------------------------------------------------
    # Save / Load (resume support)
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            'regularization_terms': {
                tid: {
                    'importance': {k: v.detach().cpu() for k, v in reg['importance'].items()},
                    'task_param': {k: v.detach().cpu() for k, v in reg['task_param'].items()},
                }
                for tid, reg in self.regularization_terms.items()
            },
            'num_tasks_consolidated': self.num_tasks_consolidated,
            'lambda_ewc': self.lambda_ewc,
        }

    def load_state_dict(self, state, agent=None):
        self.regularization_terms = state['regularization_terms']
        self.num_tasks_consolidated = state['num_tasks_consolidated']
        self.lambda_ewc = state.get('lambda_ewc', self.lambda_ewc)
        self._penalty_cache_valid = False
        if agent is not None and self.num_tasks_consolidated > 0:
            self._rebuild_penalty_cache(agent)

    def to_device(self, device):
        """Move all stored Fishers and θ* tensors to `device`. Penalty cache
        becomes stale; call `_rebuild_penalty_cache(agent)` afterwards."""
        for tid, reg in self.regularization_terms.items():
            for k in reg['importance']:
                reg['importance'][k] = reg['importance'][k].to(device)
                reg['task_param'][k] = reg['task_param'][k].to(device)
        self._penalty_cache_valid = False
