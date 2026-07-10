from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.

		# `task_id_routing` is a single-task mode (cfg.multitask=False) that
		# injects a task one-hot into the MoE routing in the same shape as
		# the progmoe backbone. Used by the naive task-aware sequential
		# baseline to isolate the contribution of masking + freezing +
		# separation versus task-conditioned routing alone.
		self._task_id_routing = (
			bool(getattr(cfg, 'task_id_routing', False))
			and not cfg.multitask
		)
		if self._task_id_routing:
			assert bool(getattr(cfg, 'use_moe', False)), (
				'task_id_routing requires cfg.use_moe=True.'
			)
			self._num_tasks_for_routing = int(getattr(cfg, 'num_tasks', 0))
			assert self._num_tasks_for_routing > 0, (
				'task_id_routing requires cfg.num_tasks > 0.'
			)
			one_hot = torch.zeros(self._num_tasks_for_routing)
			one_hot[0] = 1.0
			self.register_buffer('current_task_one_hot', one_hot)
		else:
			self._num_tasks_for_routing = 0

		self._encoder = layers.enc(cfg)

		# Extra-dim contribution to dynamics + reward expert input.
		#  - multitask:        cfg.task_dim   (learnable nn.Embedding output)
		#  - task_id_routing:  num_tasks      (one-hot)
		#  - vanilla single:   0
		extra_dim = (cfg.task_dim if cfg.multitask
		             else (self._num_tasks_for_routing if self._task_id_routing
		                   else 0))
		dyn_in = cfg.latent_dim + cfg.action_dim + extra_dim
		rew_in = cfg.latent_dim + cfg.action_dim + extra_dim

		self._use_moe = bool(getattr(cfg, 'use_moe', False))
		if self._use_moe:
			K = int(getattr(cfg, 'num_experts', 4))
			# PRISM-WM gate convention:
			#  - multitask:        gate sees task_emb only
			#  - task_id_routing:  gate sees task one-hot only
			#  - vanilla single:   gate sees [z, a]
			# Experts always see [z, a, task_signal?]. A scalar τ context
			# column is appended inside MoEBlock as a phase indicator.
			if cfg.multitask:
				gate_dim = cfg.task_dim
			elif self._task_id_routing:
				gate_dim = self._num_tasks_for_routing
			else:
				gate_dim = cfg.latent_dim + cfg.action_dim
			moe_kwargs = dict(
				num_experts=K,
				gate_dim=gate_dim,
				use_orthogonal=bool(getattr(cfg, 'use_orthogonal', False)),
				tau_init=float(getattr(cfg, 'moe_tau_init', 1.8)),
				tau_min=float(getattr(cfg, 'moe_tau_min', 0.5)),
				tau_max=float(getattr(cfg, 'moe_tau_max', 2.0)),
				beta=float(getattr(cfg, 'moe_beta', 0.02)),
				freeze_frac=float(getattr(cfg, 'moe_freeze_frac', 0.05)),
				total_steps=int(getattr(cfg, 'steps', 200_000)),
			)
			self._dynamics = layers.MoEBlock(dyn_in, cfg.mlp_dim, cfg.latent_dim, act=layers.SimNorm(cfg), **moe_kwargs)
			self._reward = layers.MoEBlock(rew_in, cfg.mlp_dim, max(cfg.num_bins, 1), **moe_kwargs)
		else:
			self._dynamics = layers.mlp(dyn_in, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._reward = layers.mlp(rew_in, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		_reward_last = self._reward.head if self._use_moe else self._reward[-1]
		init.zero_([_reward_last.weight, self._Qs.params["2", "weight"]])

		# Optionally overwrite the freshly-initialised encoder with offline-
		# pretrained weights and (optionally) freeze it. Must come AFTER
		# `self.apply(init.weight_init)` (which would otherwise clobber the
		# loaded weights) and BEFORE the optimizer is built in TDMPC2.__init__.
		self._pretrained_encoder_status = layers.maybe_load_pretrained_encoder(
			self._encoder, cfg)

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		Supports single-tensor obs (state OR rgb) as well as multi-modal
		dict/TensorDict obs (keys 'state' and 'rgb'), with optional time
		leading dim handled by iterating.
		"""
		# Multi-modal dict / TensorDict obs
		if not isinstance(obs, torch.Tensor) and hasattr(obs, 'keys'):
			if 'fusion' in self._encoder:
				state = obs['state']
				rgb = obs['rgb']
				if rgb.ndim == 5:
					T = rgb.shape[0]
					outs = []
					for t in range(T):
						s_feat = self._encoder['state'](state[t])
						r_feat = self._encoder['rgb'](rgb[t])
						outs.append(self._encoder['fusion'](torch.cat([s_feat, r_feat], dim=-1)))
					return torch.stack(outs)
				s_feat = self._encoder['state'](state)
				r_feat = self._encoder['rgb'](rgb)
				return self._encoder['fusion'](torch.cat([s_feat, r_feat], dim=-1))
			# Single-key dict fallback
			k = next(iter(obs.keys()))
			v = obs[k]
			if k == 'rgb' and v.ndim == 5:
				return torch.stack([self._encoder[k](o) for o in v])
			return self._encoder[k](v)

		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def _task_emb_tensor(self, z, task):
		"""Return the task-side tensor to feed into the MoE block.

		Modes:
		  - cfg.multitask:        look up learnable `_task_emb(task)` and
		                          broadcast to z's leading dims.
		  - task_id_routing:      return the registered `current_task_one_hot`
		                          buffer broadcast to z's leading dims.
		                          `task` arg is ignored — the active task is
		                          set externally via `set_current_task`.
		  - vanilla single-task:  return None (no task signal in MoE).
		"""
		if self.cfg.multitask:
			if isinstance(task, int):
				task = torch.tensor([task], device=z.device)
			emb = self._task_emb(task.long())
			if z.ndim == 3:
				emb = emb.unsqueeze(0).expand(z.shape[0], z.shape[1], -1)
			elif emb.shape[0] == 1:
				emb = emb.expand(z.shape[0], -1)
			return emb
		if self._task_id_routing:
			leading = z.shape[:-1]
			return self.current_task_one_hot.expand(
				*leading, self._num_tasks_for_routing,
			)
		return None

	def set_current_task(self, task_idx):
		"""Flip the routing task one-hot. No-op unless task_id_routing=True.
		Called by the naive task-aware sequential trainer at task boundaries
		and inside the cross-task eval loop."""
		if not self._task_id_routing:
			return
		idx = int(task_idx)
		assert 0 <= idx < self._num_tasks_for_routing, (
			f'task_idx {idx} out of [0, {self._num_tasks_for_routing})'
		)
		self.current_task_one_hot.zero_()
		self.current_task_one_hot[idx] = 1.0

	def current_task(self):
		"""Return the index of the active routing task, or None if
		task_id_routing is disabled.

		Derived from the `current_task_one_hot` buffer (via argmax) rather
		than a separate Python attribute, so the task identity survives a
		state_dict save/load round-trip (which carries the buffer but not
		Python attributes)."""
		if not self._task_id_routing:
			return None
		return int(self.current_task_one_hot.argmax().item())

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.

		Under `cfg.use_moe=True`, the MoE block directly emits the next latent
		on the SimNorm simplex (matching PRISM-WM's official `core/common/
		world_model.py:next`). No residual.
		"""
		if self._use_moe:
			task_emb = self._task_emb_tensor(z, task)
			return self._dynamics(z, a, task_emb)
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self._use_moe:
			task_emb = self._task_emb_tensor(z, task)
			return self._reward(z, a, task_emb)
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)

	def gate_weights(self, z, a, task):
		"""
		Returns per-step softmax gate weights of the dynamics and reward MoE
		blocks for the given (z, a). Only valid when `cfg.use_moe=True`.
		Returns a dict with keys 'dynamics' and 'reward', each a tensor of
		shape [..., num_experts].
		"""
		assert self._use_moe, "gate_weights() only available when cfg.use_moe=True"
		task_emb = self._task_emb_tensor(z, task)
		_, w_dyn = self._dynamics.forward_with_gate(z, a, task_emb)
		_, w_rew = self._reward.forward_with_gate(z, a, task_emb)
		return {"dynamics": w_dyn, "reward": w_rew}

	def gate_diagnostics(self, z, a, task):
		"""
		Like `gate_weights`, but also returns the pre-aggregation per-expert
		feature stack of both blocks. The feature stack lets the evaluator
		measure *functional* expert collapse (the failure mode Gram-Schmidt
		is supposed to prevent) — two experts can be functionally identical
		even when the gate distributes evenly between them.

		Returns: {'dynamics': {'weights': [..., K], 'features': [..., K, H]},
		          'reward':   {'weights': [..., K], 'features': [..., K, H]}}
		"""
		assert self._use_moe, "gate_diagnostics() only available when cfg.use_moe=True"
		task_emb = self._task_emb_tensor(z, task)
		_, w_dyn, f_dyn = self._dynamics.forward_with_diagnostics(z, a, task_emb)
		_, w_rew, f_rew = self._reward.forward_with_diagnostics(z, a, task_emb)
		return {
			"dynamics": {"weights": w_dyn, "features": f_dyn},
			"reward":   {"weights": w_rew, "features": f_rew},
		}
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
