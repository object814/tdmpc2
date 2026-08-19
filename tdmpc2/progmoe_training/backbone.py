"""Shared backbone + per-task heads for PRISM-WM continual learning.

Architecture (DreamerV3-inspired three-way split):

  PrismBackbone  — task-agnostic, persists across all tasks
      ├── _encoder          multi-modal RGB + proprio fusion (unchanged)
      └── _dynamics         MaskedMoEBlock with total_K experts pre-allocated.

                            Routing is task-conditioned via the
                            `current_task_one_hot` buffer.

  TaskModules    — task-specific, FRESH at every task boundary
      ├── _reward           plain MLP (z, a) → num_bins
      ├── _termination      plain MLP z → 1   (only if cfg.episodic)
      ├── _pi               plain MLP z → 2 * action_dim
      ├── _Qs / _target_Qs  Ensemble of Q-networks + target-update machinery
      └── scale             RunningScale for π-loss Q normalization

Invariants:
  - TaskModules has NO reference to PrismBackbone (no `self.backbone`, no
    submodule registration). `task_modules.state_dict()` is therefore
    disjoint from `backbone.state_dict()` — load/save are clean.
  - All TaskModules forward methods take `z` (already-encoded latent) as
    explicit input. The agent encodes once via backbone and threads `z`
    through both sides.
  - Cross-task evaluation pairs the LIVE backbone with a loaded historical
    TaskModules through `EvalAgent`, with `backbone.scoped(...)` to
    temporarily restrict `_active_K` and `task_idx` without mutating the
    training state. No weight swapping.
"""

import contextlib
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from common import layers, math, init
from common.scale import RunningScale

from progmoe_training.masked_moe import MaskedMoEBlock


# ===========================================================================
#  Backbone (shared)
# ===========================================================================

class PrismBackbone(nn.Module):
	"""Encoder + masked dynamics MoE. Persists across all tasks."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		assert not getattr(cfg, 'multitask', False), (
			'PrismBackbone handles task identity via a one-hot; '
			'cfg.multitask=True is incompatible.'
		)

		self.num_tasks = int(getattr(cfg, 'num_tasks', 0))
		self.K_per_task = int(getattr(cfg, 'K_per_task', 4))
		assert self.num_tasks > 0, 'cfg.num_tasks must be > 0'
		assert self.K_per_task > 0, 'cfg.K_per_task must be > 0'

		# Expert-management mode.
		#   progressive=True  (progmoe): MaskedMoEBlock grows its active
		#       window by K_per_task each task and freezes prior experts.
		#       total_K = num_tasks * K_per_task.
		#   progressive=False (naive sequential): static full-K MoE — all
		#       `num_experts` experts always active + trainable; task
		#       boundaries only swap the per-task heads. total_K = num_experts.
		self.progressive = bool(getattr(cfg, 'progressive_moe', True))
		if self.progressive:
			self.total_K = self.num_tasks * self.K_per_task
		else:
			self.total_K = int(getattr(
				cfg, 'num_experts', self.num_tasks * self.K_per_task))
		assert self.total_K > 0, 'total_K must be > 0'

		# Gram-Schmidt orthogonalization can yield at most `mlp_dim` mutually
		# orthogonal expert feature vectors — guard against asking for more.
		use_orthogonal = bool(getattr(cfg, 'use_orthogonal', False))
		if use_orthogonal:
			assert self.total_K <= cfg.mlp_dim, (
				f'use_orthogonal=True requires total_K ({self.total_K}) <= '
				f'mlp_dim ({cfg.mlp_dim}): Gram-Schmidt produces at most '
				f'mlp_dim mutually orthogonal expert feature vectors.'
			)

		# Encoder — same multi-modal stack as common.world_model.WorldModel.
		self._encoder = layers.enc(cfg)

		# Per-task rgb projectors ------------------------------------------
		# With encoder_type=dino + dino_per_task_projector, the rgb branch
		# stops at pooled ViT features and the projection into encoder space
		# lives HERE, one module per task. Task t's projector is frozen at the
		# task boundary along with task t's experts, so the frozen experts of
		# earlier tasks keep seeing the exact input distribution they were
		# trained on. A single shared projector (the previous behaviour) keeps
		# training during later tasks and silently drifts that distribution —
		# freezing the DINO backbone does not help, because the drift is in
		# the trainable projector downstream of it.
		#
		# Pre-allocated for all `num_tasks` up front so the checkpoint shape is
		# constant across the whole CL run, matching the `total_K` experts
		# convention (load_backbone stays bit-exact, no shape negotiation).
		self._projectors = None
		# NB: nn.ModuleDict has no `.get()` — membership must be tested with
		# `in`, otherwise this silently skips and encode() returns raw pooled
		# features instead of a latent.
		rgb_branch = self._encoder['rgb'] if 'rgb' in self._encoder else None
		if rgb_branch is not None and getattr(rgb_branch, 'externalize_projector', False):
			multimodal = ('state' in cfg.obs_shape and 'rgb' in cfg.obs_shape)
			proj_out = cfg.enc_dim if multimodal else cfg.latent_dim
			proj_act = None if multimodal else layers.SimNorm(cfg)
			self._projectors = nn.ModuleList([
				layers.dino_projector(
					rgb_branch.feature_dim, proj_out, cfg, act=proj_act)
				for _ in range(self.num_tasks)
			])

		# Dynamics MoE — gate sees one-hot, experts see [z, a, one-hot].
		dyn_in = cfg.latent_dim + cfg.action_dim + self.num_tasks
		gate_dim = self.num_tasks
		moe_kwargs = dict(
			num_experts=self.total_K,
			gate_dim=gate_dim,
			use_orthogonal=use_orthogonal,
			tau_init=float(getattr(cfg, 'moe_tau_init', 1.8)),
			tau_min=float(getattr(cfg, 'moe_tau_min', 0.5)),
			tau_max=float(getattr(cfg, 'moe_tau_max', 2.0)),
			beta=float(getattr(cfg, 'moe_beta', 0.02)),
			freeze_frac=float(getattr(cfg, 'moe_freeze_frac', 0.05)),
			total_steps=int(getattr(cfg, 'steps', 200_000)),
		)
		self._dynamics = MaskedMoEBlock(
			dyn_in, cfg.mlp_dim, cfg.latent_dim,
			act=layers.SimNorm(cfg), **moe_kwargs,
		)

		# Project-wide init
		self.apply(init.weight_init)

		# Optionally overwrite the freshly-initialised encoder with offline-
		# pretrained weights and (optionally) freeze it. Must come AFTER
		# `self.apply(init.weight_init)` and BEFORE the optimizers are built
		# (ContinualTDMPC2._build_optims filters on requires_grad, so a frozen
		# encoder is automatically excluded from backbone_optim).
		self._pretrained_encoder_status = layers.maybe_load_pretrained_encoder(
			self._encoder, cfg)

		# Open the active expert window.
		#   progressive: only the task-0 slot [0, K_per_task).
		#   static:      all total_K experts active + trainable from the start.
		self._dynamics.set_active_K(
			self.K_per_task if self.progressive else self.total_K)

		# Task one-hot buffer (starts at task 0).
		one_hot = torch.zeros(self.num_tasks)
		one_hot[0] = 1.0
		self.register_buffer('current_task_one_hot', one_hot)
		self._current_task_idx = 0

	# ------------------------------------------------------------------
	# Task identity
	# ------------------------------------------------------------------
	def set_current_task(self, task_idx):
		idx = int(task_idx)
		assert 0 <= idx < self.num_tasks, (
			f'task_idx {idx} out of [0, {self.num_tasks})'
		)
		self._current_task_idx = idx
		self.current_task_one_hot.zero_()
		self.current_task_one_hot[idx] = 1.0

	def current_task(self):
		return self._current_task_idx

	# ------------------------------------------------------------------
	# Per-task projectors
	# ------------------------------------------------------------------
	def freeze_projectors_through(self, t):
		"""Freeze per-task projectors `[0, t)`; leave `[t, num_tasks)` trainable.

		The mirror of `MaskedMoEBlock.freeze_experts_through`, called from the
		same task-boundary transition so a task's projector and its experts are
		always frozen together. No-op unless per-task projectors are enabled.
		"""
		if self._projectors is None:
			return
		t = int(t)
		for i, proj in enumerate(self._projectors):
			trainable = i >= t
			for p in proj.parameters():
				p.requires_grad = trainable

	def _project_rgb(self, feat):
		"""Apply the CURRENT task's projector to pooled rgb features.

		Routing is by task index, not by the MoE gate: the projector is not a
		routed component, so each task simply selects its own module. Under
		`scoped(task_idx=j)` this automatically picks task j's projector, which
		is what makes cross-task evaluation see the representation task j was
		actually trained against.
		"""
		return self._projectors[self._current_task_idx](feat)

	def _task_one_hot_for(self, z):
		leading = z.shape[:-1]
		return self.current_task_one_hot.expand(*leading, self.num_tasks)

	def _task_one_hot_from_ids(self, task_ids, z):
		"""Build a PER-SAMPLE one-hot from explicit task ids.

		`task_ids` is a 1-D LongTensor of shape [B] (one id per batch sample);
		`z` is the latent it must align with ([B, latent_dim] in the per-step
		rollout). Returns [B, num_tasks] so the MoE gate + expert input route
		each row by its OWN task — used by the merged-buffer (multitask-style)
		update where a single batch mixes rows from several tasks."""
		oh = F.one_hot(task_ids.long().to(z.device), self.num_tasks)
		return oh.to(z.dtype)

	# ------------------------------------------------------------------
	# MoE state mutators (delegated to the masked block)
	# ------------------------------------------------------------------
	def set_active_K(self, K):
		self._dynamics.set_active_K(K)

	def freeze_experts_through(self, K):
		self._dynamics.freeze_experts_through(K)

	@property
	def active_K(self):
		return self._dynamics._active_K

	@property
	def frozen_K(self):
		return self._dynamics._frozen_K

	@contextlib.contextmanager
	def scoped(self, *, active_K=None, task_idx=None, tau=None):
		"""Temporarily change active_K, task_idx, and/or τ. Restores all on
		exit.

		Used by EvalAgent so we can run a cross-task eval pass against the
		live training backbone without mutating its training state.

		`tau` is the per-task τ snapshot path: in non-freeze recipes τ
		drifts during later tasks, so to evaluate task j at its trained
		operating point we restore τ_j here. None means "use whatever τ is
		live" (current behavior for callers that don't opt in)."""
		saved_active = self._dynamics._active_K
		saved_frozen = self._dynamics._frozen_K
		saved_idx = self._current_task_idx
		saved_one_hot = self.current_task_one_hot.clone()
		saved_tau = self._dynamics.tau
		try:
			if active_K is not None:
				self._dynamics._active_K = int(active_K)
			if task_idx is not None:
				self.set_current_task(int(task_idx))
			if tau is not None:
				self._dynamics.tau = float(tau)
			yield
		finally:
			self._dynamics._active_K = saved_active
			self._dynamics._frozen_K = saved_frozen
			self._current_task_idx = saved_idx
			self.current_task_one_hot.copy_(saved_one_hot)
			self._dynamics.tau = saved_tau

	# ------------------------------------------------------------------
	# Forward methods
	# ------------------------------------------------------------------
	def encode(self, obs):
		"""Multi-modal or single-modal observation encoding.

		Handles:
		  - TensorDict / dict obs with 'state' and 'rgb' (multi-modal fusion)
		  - TensorDict / dict obs with a single key
		  - Plain Tensor obs (`cfg.obs` resolves the encoder branch)
		  - (T, B, ...) leading time-batch dim on rgb observations
		"""
		# `_encode_rgb` folds in the per-task projector when one is configured;
		# otherwise it is exactly `self._encoder['rgb']`.
		if not isinstance(obs, torch.Tensor) and hasattr(obs, 'keys'):
			if 'fusion' in self._encoder:
				state = obs['state']
				rgb = obs['rgb']
				if rgb.ndim == 5:
					T = rgb.shape[0]
					outs = []
					for t in range(T):
						s_feat = self._encoder['state'](state[t])
						r_feat = self._encode_rgb(rgb[t])
						outs.append(self._encoder['fusion'](
							torch.cat([s_feat, r_feat], dim=-1)))
					return torch.stack(outs)
				s_feat = self._encoder['state'](state)
				r_feat = self._encode_rgb(rgb)
				return self._encoder['fusion'](
					torch.cat([s_feat, r_feat], dim=-1))
			k = next(iter(obs.keys()))
			v = obs[k]
			if k == 'rgb':
				if v.ndim == 5:
					return torch.stack([self._encode_rgb(o) for o in v])
				return self._encode_rgb(v)
			return self._encoder[k](v)

		if self.cfg.obs == 'rgb':
			if obs.ndim == 5:
				return torch.stack([self._encode_rgb(o) for o in obs])
			return self._encode_rgb(obs)
		return self._encoder[self.cfg.obs](obs)

	def _encode_rgb(self, rgb):
		"""rgb branch + (optional) current-task projector."""
		feat = self._encoder['rgb'](rgb)
		if self._projectors is None:
			# Fail loudly rather than silently returning unprojected features:
			# if the branch externalized its head, a projector MUST exist.
			assert not getattr(self._encoder['rgb'], 'externalize_projector', False), (
				'rgb branch externalized its projector but PrismBackbone built '
				'no projector bank — encode() would return raw pooled features.'
			)
			return feat
		return self._project_rgb(feat)

	def next(self, z, a, task_ids=None):
		"""z_{t+1} = MoE_dyn(z, a, task_one_hot).

		`task_ids=None` (default): route the whole batch by the live global
		`current_task_one_hot` — the original single-task behavior used by all
		existing callers (planning, eval, the standard `_update`, aux ER).

		`task_ids` provided ([B] LongTensor): route PER SAMPLE by each row's
		own task id. Used only by the merged-buffer (multitask-style) update so
		a single batch mixing tasks routes task-j rows through task-j's experts
		and task-t rows through task-t's."""
		if task_ids is None:
			e = self._task_one_hot_for(z)
		else:
			e = self._task_one_hot_from_ids(task_ids, z)
		return self._dynamics(z, a, e)

	def gate_weights(self, z, a):
		e = self._task_one_hot_for(z)
		_, w = self._dynamics.forward_with_gate(z, a, e)
		return w

	def gate_diagnostics(self, z, a):
		e = self._task_one_hot_for(z)
		_, w, f = self._dynamics.forward_with_diagnostics(z, a, e)
		return {'weights': w, 'features': f}


# ===========================================================================
#  Per-task modules (fresh per task)
# ===========================================================================

class TaskModules(nn.Module):
	"""Reward + Q + termination + π + scale for ONE task.

	NO reference to PrismBackbone — all forward methods take an already-
	encoded latent `z` as input. This keeps `task_modules.state_dict()`
	completely disjoint from `backbone.state_dict()`.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		rew_in = cfg.latent_dim + cfg.action_dim
		q_in = cfg.latent_dim + cfg.action_dim
		pi_in = cfg.latent_dim

		# Reward: plain TDMPC2-style MLP (same as the non-MoE PRISM-WM reward
		# baseline, just instantiated fresh per task).
		self._reward = layers.mlp(rew_in, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1))

		# Termination (only when episodic=True)
		if cfg.episodic:
			self._termination = layers.mlp(pi_in, 2 * [cfg.mlp_dim], 1)
		else:
			self._termination = None

		# Policy
		self._pi = layers.mlp(pi_in, 2 * [cfg.mlp_dim], 2 * cfg.action_dim)

		# Q-ensemble
		self._Qs = layers.Ensemble([
			layers.mlp(q_in, 2 * [cfg.mlp_dim], max(cfg.num_bins, 1),
			           dropout=cfg.dropout)
			for _ in range(cfg.num_q)
		])

		# Init
		self.apply(init.weight_init)
		# TDMPC2 zero-init for reward + Q final-layer (matches base WorldModel).
		init.zero_([self._reward[-1].weight, self._Qs.params['2', 'weight']])

		self.register_buffer('log_std_min', torch.tensor(cfg.log_std_min))
		self.register_buffer(
			'log_std_dif',
			torch.tensor(cfg.log_std_max) - self.log_std_min,
		)

		# RunningScale (per-task statistic for π loss normalization).
		self.scale = RunningScale(cfg)

		# Target / detach Q machinery — mirrors WorldModel.init().
		self._init_q_aux()

	def _init_q_aux(self):
		"""Initialize `_detach_Qs` / `_target_Qs` with TensorDictParams.
		Idempotent; re-runs on .to(device) to keep param storage consistent."""
		self._detach_Qs_params = TensorDictParams(
			self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(
			self._Qs.params.data.clone(), no_convert=True)
		with self._detach_Qs_params.data.to('meta').to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)
		delattr(self._detach_Qs, 'params')
		self._detach_Qs.__dict__['params'] = self._detach_Qs_params
		delattr(self._target_Qs, 'params')
		self._target_Qs.__dict__['params'] = self._target_Qs_params

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self._init_q_aux()
		return self

	def train(self, mode=True):
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	# ------------------------------------------------------------------
	# Forward methods — take encoded latent `z` as input.
	# ------------------------------------------------------------------
	def reward(self, z, a):
		return self._reward(torch.cat([z, a], dim=-1))

	def termination(self, z, unnormalized=False):
		assert self._termination is not None, (
			'termination head only built when cfg.episodic=True'
		)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))

	def pi(self, z):
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)
		log_prob = math.gaussian_logprob(eps, log_std)
		scaled_log_prob = log_prob * eps.shape[-1]
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)
		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			'mean': mean,
			'log_std': log_std,
			'action_prob': 1.,
			'entropy': -log_prob,
			'scaled_entropy': -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, return_type='min', target=False, detach=False):
		assert return_type in {'min', 'avg', 'all'}
		x = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(x)
		if return_type == 'all':
			return out
		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == 'min':
			return Q.min(0).values
		return Q.sum(0) / 2


# ===========================================================================
#  Cross-task eval wrapper
# ===========================================================================

class EvalAgent:
	"""Lightweight pairing of a shared backbone + a loaded TaskModules for
	one eval run.

	NOT an nn.Module — pure Python. So:
	  - it can't accidentally appear in any state_dict
	  - it doesn't try to .to() the (shared) backbone
	  - it can be safely discarded after eval

	Used by the trainer to evaluate task j ≤ current_task without disturbing
	the live training state.
	"""

	def __init__(self, agent_for_plan, backbone, task_modules, task_idx,
	             active_K, cfg, tau=None):
		self._agent = agent_for_plan      # `ContinualTDMPC2` — used for _plan
		self.backbone = backbone           # shared reference, not copied
		self.task_modules = task_modules   # loaded fresh, owned by this object
		self.task_idx = int(task_idx)
		self.active_K = int(active_K)
		self.cfg = cfg
		# Per-task τ snapshot to restore during this eval pass. None means
		# "use whatever τ the live backbone has right now" (backward-compat
		# with freeze recipes and pipelines that don't opt into per-task τ).
		self.tau = float(tau) if tau is not None else None
		self.device = next(backbone.parameters()).device
		self._prev_mean = torch.zeros(
			cfg.horizon, cfg.action_dim, device=self.device)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=True, task=None):
		# `task` arg ignored — task identity comes from our stored task_idx.
		if isinstance(obs, dict):
			obs = {k: v.to(self.device, non_blocking=True).unsqueeze(0)
			       for k, v in obs.items()}
		elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		else:
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)

		with self.backbone.scoped(
			active_K=self.active_K, task_idx=self.task_idx, tau=self.tau,
		):
			if self.cfg.mpc:
				return self._agent._plan(
					obs, t0=t0, eval_mode=eval_mode,
					backbone=self.backbone, task_modules=self.task_modules,
					prev_mean=self._prev_mean,
				).cpu()
			z = self.backbone.encode(obs)
			action, info = self.task_modules.pi(z)
			if eval_mode:
				action = info['mean']
			return action[0].cpu()
