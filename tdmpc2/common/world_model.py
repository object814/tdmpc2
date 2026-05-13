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
		self._encoder = layers.enc(cfg)
		dyn_in = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		rew_in = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		self._use_moe = bool(getattr(cfg, 'use_moe', False))
		if self._use_moe:
			K = int(getattr(cfg, 'num_experts', 4))
			self._dynamics = layers.MoEBlock(dyn_in, cfg.mlp_dim, cfg.latent_dim, num_experts=K, act=layers.SimNorm(cfg))
			self._reward = layers.MoEBlock(rew_in, cfg.mlp_dim, max(cfg.num_bins, 1), num_experts=K)
		else:
			self._dynamics = layers.mlp(dyn_in, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
			self._reward = layers.mlp(rew_in, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		self.apply(init.weight_init)
		_reward_last = self._reward.head if self._use_moe else self._reward[-1]
		init.zero_([_reward_last.weight, self._Qs.params["2", "weight"]])

		# Test: re-normalise after residual
		# Stateless SimNorm reused inside `next()` to reproject (z + ∆z) onto
		# tdmpc2's simplex latent (groups of 8 summing to 1). Without this
		# the residual sum doubles per rollout step and the consistency loss
		# can't fit a SimNorm-encoded target.
		self._post_residual_simnorm = layers.SimNorm(cfg) if self._use_moe else None
		# Test: re-normalise after residual

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

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.

		When `cfg.use_moe` is set, the dynamics MoE predicts a residual ∆z and
		the next latent is `z + ∆z` (PRISM-WM, Algorithm 1, line 13).
		Otherwise the original monolithic dynamics model returns the next
		latent directly.
		"""
		z_res = z
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		out = self._dynamics(z)
		if self._use_moe and getattr(self.cfg, 'moe_residual_dynamics', True):
			# Test: re-normalise after residual
			return self._post_residual_simnorm(z_res + out)
			# Test: re-normalise after residual
		return out

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
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
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		x = torch.cat([z, a], dim=-1)
		_, w_dyn = self._dynamics.forward_with_gate(x)
		_, w_rew = self._reward.forward_with_gate(x)
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
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		x = torch.cat([z, a], dim=-1)
		_, w_dyn, f_dyn = self._dynamics.forward_with_diagnostics(x)
		_, w_rew, f_rew = self._reward.forward_with_diagnostics(x)
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
