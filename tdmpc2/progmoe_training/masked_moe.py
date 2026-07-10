"""Static-K masked Mixture-of-Experts block for progressive continual learning.

Replaces the earlier `GrowingMoEBlock`. Instead of growing the expert pool at
each task boundary, we pre-allocate `total_K = num_tasks * K_per_task` experts
up front and mutate an `_active_K` attribute to restrict which experts are
routable / runnable at any moment.

Properties:
  - At task t the block has `_active_K = (t + 1) * K_per_task`.
  - Experts `[0, frozen_K)` have `requires_grad=False` (frozen prior tasks).
  - Experts `[frozen_K, active_K)` are trainable (current task slot).
  - Experts `[active_K, total_K)` are completely inactive:
      * never forwarded (no compute)
      * never reached by autograd (no gradient)
      * `requires_grad=False` (no Adam state pressure)
  - The gate's first `active_K` rows are sliced from `gate.weight` and used
    via `F.linear`; rows `[active_K, total_K)` are not touched in either
    forward or backward, so their weights remain at init until activated.
  - `H_target` (entropy target for the τ schedule) is recomputed as
    `0.75 · log(active_K)` whenever `_active_K` changes.

Checkpoint shape is constant across the entire CL run (`total_K` experts
plus a gate of width `total_K`), so loading prev-task backbones is trivially
bit-exact and no API ever has to negotiate shape changes mid-run.
"""

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import MoEBlock


class MaskedMoEBlock(MoEBlock):
	"""MoEBlock with an `_active_K` mask. Subclasses MoEBlock for parameter
	layout compatibility (gate, experts list, head, τ buffers); overrides
	the forward path to route only over the first `_active_K` experts."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		# Mask state lives in buffers so it survives save_state_dict /
		# load_state_dict round-trips. Use Python @property accessors below
		# to keep call sites scalar-typed.
		self.register_buffer(
			'_active_K_buf',
			torch.tensor(self.num_experts, dtype=torch.long),
		)
		self.register_buffer(
			'_frozen_K_buf', torch.tensor(0, dtype=torch.long),
		)
		self._sync_grad_flags()

	# Scalar accessors so existing code reads `self._active_K` as an int.
	@property
	def _active_K(self):
		return int(self._active_K_buf.item())

	@_active_K.setter
	def _active_K(self, v):
		self._active_K_buf.fill_(int(v))

	@property
	def _frozen_K(self):
		return int(self._frozen_K_buf.item())

	@_frozen_K.setter
	def _frozen_K(self, v):
		self._frozen_K_buf.fill_(int(v))

	# ------------------------------------------------------------------
	# Public mutators
	# ------------------------------------------------------------------
	def set_active_K(self, K):
		"""Open up routing to the first K experts. Resets the τ schedule
		(re-enters the freeze phase) with `H_target` updated for the new K."""
		K = int(K)
		assert 1 <= K <= self.num_experts, (
			f'active_K must be in [1, {self.num_experts}]; got {K}'
		)
		self._active_K = K
		self._reset_tau_schedule()
		self.H_target = 0.75 * math.log(K)
		self._sync_grad_flags()

	def freeze_experts_through(self, K):
		"""Set `requires_grad=False` on experts `[0, K)`. Frozen experts may
		still execute in forward (if K ≤ active_K) and contribute features
		— they just don't receive gradient. Adam state is dropped on optimizer
		rebuild."""
		K = int(K)
		assert 0 <= K <= self.num_experts, (
			f'frozen_K must be in [0, {self.num_experts}]; got {K}'
		)
		self._frozen_K = K
		self._sync_grad_flags()

	def _sync_grad_flags(self):
		"""trainable iff i ∈ [frozen_K, active_K); else requires_grad=False."""
		for i, expert in enumerate(self.experts):
			trainable = (i >= self._frozen_K) and (i < self._active_K)
			for p in expert.parameters():
				p.requires_grad = trainable

	@contextlib.contextmanager
	def scoped_active_K(self, K):
		"""Temporarily change active_K (e.g. for cross-task eval). Saves and
		restores `_active_K` + `_frozen_K` but does NOT touch the τ schedule
		(would corrupt training state on resume). Intended for use in no_grad
		eval contexts only."""
		saved_active = self._active_K
		saved_frozen = self._frozen_K
		try:
			self._active_K = int(K)
			yield
		finally:
			self._active_K = saved_active
			self._frozen_K = saved_frozen

	# ------------------------------------------------------------------
	# Forward path (overrides parent)
	# ------------------------------------------------------------------
	def _route_masked(self, expert_in, gate_base):
		"""Routing over the first `_active_K` experts.

		Crucially:
		  - `gate.weight[:_active_K]` is sliced BEFORE `F.linear`, so the
		    matmul only does K_active rows of work AND backward only
		    populates `gate.weight.grad[:_active_K]`.
		  - The expert stack iterates `range(_active_K)`, so no forward
		    or backward is run on masked experts.
		"""
		K_active = self._active_K
		tau_col = torch.full_like(gate_base[..., :1], self.tau)
		gate_in = torch.cat([gate_base, tau_col], dim=-1)

		W_active = self.gate.weight[:K_active]                              # [K_active, gate_dim+1]
		logits = F.linear(gate_in, W_active) / max(self.tau, 1e-6)          # [N, K_active]
		weights = F.softmax(logits, dim=-1)

		feats = torch.stack(
			[self.experts[k](expert_in) for k in range(K_active)],
			dim=-2,
		)                                                                    # [N, K_active, H]
		if self._ortho is not None:
			feats = self._ortho(feats)

		if self.training:
			with torch.no_grad():
				ent = -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
			self.last_entropy = float(ent.item())
			self._update_tau(self.last_entropy)
			self._global_step += 1

		combined = (weights.unsqueeze(-1) * feats).sum(dim=-2)               # [N, H]
		return combined, weights, feats

	def forward(self, z, a, task_emb=None):
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, _, _ = self._route_masked(expert_in, gate_base)
		return restore(self.head(combined))

	def forward_with_gate(self, z, a, task_emb=None):
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, weights, _ = self._route_masked(expert_in, gate_base)
		if z.ndim == 3:
			T, B, _ = z.shape
			weights = weights.view(T, B, -1)
		return restore(self.head(combined)), weights

	def forward_with_diagnostics(self, z, a, task_emb=None):
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, weights, feats = self._route_masked(expert_in, gate_base)
		if z.ndim == 3:
			T, B, _ = z.shape
			weights = weights.view(T, B, -1)
			feats = feats.view(T, B, *feats.shape[1:])
		return restore(self.head(combined)), weights, feats

	def __repr__(self):
		return (
			f'MaskedMoEBlock(num_experts={self.num_experts}, '
			f'active_K={self._active_K}, frozen_K={self._frozen_K}, '
			f'in={self.in_dim}, gate={self.gate_dim}, '
			f'hidden={self.hidden_dim}, ortho={self.use_orthogonal})'
		)
