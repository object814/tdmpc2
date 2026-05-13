"""WorldModel variant for progressive-MoE continual learning.

Differences vs `common.world_model.WorldModel`:

  - `_dynamics`, `_reward` are `GrowingMoEBlock`s. Their input dim is
    `latent_dim + action_dim + num_tasks`, where the trailing block is a
    one-hot of the current task (paper's c_t = [z, a, e_task]).

  - `set_current_task(t)` flips the one-hot. Called before training
    each task AND before every cross-task eval episode.

  - `expand_for_new_task(k_new)` appends `k_new` experts to both blocks
    and freezes all experts that existed prior to the call.

  - Encoder, Q-heads, policy, termination are unchanged from the base
    WorldModel single-task path; task identity is injected ONLY on the
    MoE input (paper's "context-aware gating").

Task identity is handled entirely through the one-hot; we deliberately
keep `cfg.multitask=False` so tdmpc2's `task_emb` machinery (action
masks, multi-task discount tensor, etc.) doesn't activate.
"""

import torch
import torch.nn as nn

from common import layers, init
from common.world_model import WorldModel

from progmoe_training.growing_moe import GrowingMoEBlock


class ContinualPrismWorldModel(WorldModel):

	def __init__(self, cfg):
		# Bypass WorldModel.__init__ — we need to swap MoEBlock for
		# GrowingMoEBlock and extend the MoE input with a task one-hot.
		# Build the rest of the components by hand to mirror the base.
		torch.nn.Module.__init__(self)
		self.cfg = cfg

		assert not cfg.multitask, (
			'ContinualPrismWorldModel handles task identity via a one-hot; '
			'incompatible with tdmpc2 multitask=True.'
		)
		assert getattr(cfg, 'use_moe', False), (
			'ContinualPrismWorldModel requires cfg.use_moe=True.'
		)
		num_tasks = int(getattr(cfg, 'num_tasks', 0))
		assert num_tasks > 0, 'cfg.num_tasks must equal len(--tasks).'
		self.num_tasks = num_tasks
		self._use_moe = True

		# ---- Encoder ---------------------------------------------------
		self._encoder = layers.enc(cfg)

		# ---- Dynamics + reward: GrowingMoE with task-one-hot input -----
		K = int(getattr(cfg, 'num_experts', 4))
		dyn_in = cfg.latent_dim + cfg.action_dim + self.num_tasks
		rew_in = cfg.latent_dim + cfg.action_dim + self.num_tasks
		self._dynamics = GrowingMoEBlock(
			dyn_in, cfg.mlp_dim, cfg.latent_dim,
			num_experts=K, act=layers.SimNorm(cfg),
		)
		self._reward = GrowingMoEBlock(
			rew_in, cfg.mlp_dim, max(cfg.num_bins, 1),
			num_experts=K,
		)

		# ---- Termination / policy / Q-heads: same as base single-task --
		# (Task identity is NOT injected into these — paper's spec only
		# context-aware gates the MoE router.)
		if cfg.episodic:
			self._termination = layers.mlp(
				cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)
		else:
			self._termination = None
		self._pi = layers.mlp(
			cfg.latent_dim + cfg.task_dim,
			2 * [cfg.mlp_dim], 2 * cfg.action_dim,
		)
		self._Qs = layers.Ensemble([
			layers.mlp(
				cfg.latent_dim + cfg.action_dim + cfg.task_dim,
				2 * [cfg.mlp_dim], max(cfg.num_bins, 1),
				dropout=cfg.dropout,
			)
			for _ in range(cfg.num_q)
		])

		# ---- Project-wide init + zero-init the reward / Q final layers
		self.apply(init.weight_init)
		init.zero_([
			self._reward.head.weight,
			self._Qs.params["2", "weight"],
		])

		# Post-residual SimNorm (only used when moe_residual_dynamics=True).
		self._post_residual_simnorm = layers.SimNorm(cfg)

		self.register_buffer(
			'log_std_min', torch.tensor(cfg.log_std_min))
		self.register_buffer(
			'log_std_dif',
			torch.tensor(cfg.log_std_max) - self.log_std_min)

		# ---- Task one-hot: identifies the active task to the MoE input.
		one_hot = torch.zeros(self.num_tasks)
		one_hot[0] = 1.0
		self.register_buffer('current_task_one_hot', one_hot)
		self._current_task_idx = 0

		# Set up detach / target Q machinery (same as WorldModel.init()).
		self.init()

	# ------------------------------------------------------------------
	# Task identity API
	# ------------------------------------------------------------------
	def set_current_task(self, task_idx):
		"""Set the active task one-hot. Call before training each task
		AND before every cross-task eval episode."""
		idx = int(task_idx)
		assert 0 <= idx < self.num_tasks, \
			f'task_idx {idx} out of [0, {self.num_tasks})'
		self._current_task_idx = idx
		self.current_task_one_hot.zero_()
		self.current_task_one_hot[idx] = 1.0

	def current_task(self):
		return self._current_task_idx

	def expand_for_new_task(self, k_new=4):
		"""Append `k_new` experts to BOTH MoE blocks; freeze all experts
		that existed prior to this call. The gate is widened in lockstep,
		with old rows preserved."""
		old_K_dyn = self._dynamics.num_experts
		old_K_rew = self._reward.num_experts

		self._dynamics.add_experts(k_new)
		self._dynamics.freeze_experts(list(range(old_K_dyn)))

		self._reward.add_experts(k_new)
		self._reward.freeze_experts(list(range(old_K_rew)))

	# ------------------------------------------------------------------
	# Forward paths: append task one-hot to (z, a) before the MoE blocks.
	# All other paths (encode, Q, pi, termination) are inherited unchanged.
	# ------------------------------------------------------------------
	def _task_one_hot_for(self, z):
		"""Broadcast `current_task_one_hot` to match `z`'s leading dims."""
		leading = z.shape[:-1]
		return self.current_task_one_hot.expand(*leading, self.num_tasks)

	def next(self, z, a, task):
		"""z_{t+1} = MoE_dyn([z, a, e_task]). Optional residual + SimNorm
		reproject when `cfg.moe_residual_dynamics=True`."""
		z_res = z
		e = self._task_one_hot_for(z)
		x = torch.cat([z, a, e], dim=-1)
		out = self._dynamics(x)
		if getattr(self.cfg, 'moe_residual_dynamics', True):
			# Test: re-normalise after residual
			return self._post_residual_simnorm(z_res + out)
			# Test: re-normalise after residual
		return out

	def reward(self, z, a, task):
		e = self._task_one_hot_for(z)
		x = torch.cat([z, a, e], dim=-1)
		return self._reward(x)

	def gate_weights(self, z, a, task):
		e = self._task_one_hot_for(z)
		x = torch.cat([z, a, e], dim=-1)
		_, w_dyn = self._dynamics.forward_with_gate(x)
		_, w_rew = self._reward.forward_with_gate(x)
		return {'dynamics': w_dyn, 'reward': w_rew}

	def gate_diagnostics(self, z, a, task):
		e = self._task_one_hot_for(z)
		x = torch.cat([z, a, e], dim=-1)
		_, w_dyn, f_dyn = self._dynamics.forward_with_diagnostics(x)
		_, w_rew, f_rew = self._reward.forward_with_diagnostics(x)
		return {
			'dynamics': {'weights': w_dyn, 'features': f_dyn},
			'reward':   {'weights': w_rew, 'features': f_rew},
		}
