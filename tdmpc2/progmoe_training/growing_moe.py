"""Growing Mixture-of-Experts block.

Extends `common.layers.MoEBlock` with two task-boundary operations used
by progressive-MoE continual learning:

  - `add_experts(k_new)`: append `k_new` fresh experts. The gate
    (`nn.Linear`) is replaced by a wider one whose first `K_old` output
    rows are copied from the old gate, so routing on previously-trained
    inputs is bit-identical at the instant of expansion.

  - `freeze_experts(indices)`: set `requires_grad=False` on every
    parameter of the listed experts. Frozen experts still execute in
    forward and are still routable by the gate — they simply don't
    receive gradient updates.
"""

import torch
import torch.nn as nn

from common import init
from common.layers import MoEBlock, NormedLinear


class GrowingMoEBlock(MoEBlock):
	"""MoEBlock that can grow its expert pool and freeze old experts."""

	def add_experts(self, k_new):
		"""Append `k_new` new experts and widen the gate accordingly.
		Old gate rows are preserved so routing on previously-trained
		inputs is unchanged at the moment of expansion."""
		k_new = int(k_new)
		if k_new <= 0:
			return
		old_K = self.num_experts
		new_K = old_K + k_new

		device = self.gate.weight.device
		dtype = self.gate.weight.dtype

		# 1) Append new experts with the same 2-layer NormedLinear stack
		#    used at construction time, init'd with the project-wide init.
		for _ in range(k_new):
			expert = nn.Sequential(
				NormedLinear(self.in_dim, self.hidden_dim),
				NormedLinear(self.hidden_dim, self.hidden_dim),
			)
			expert.apply(init.weight_init)
			expert.to(device=device, dtype=dtype)
			self.experts.append(expert)

		# 2) Replace the gate with a wider Linear; copy old rows so old
		#    routing is preserved.
		old_gate = self.gate
		new_gate = nn.Linear(self.in_dim, new_K, bias=False)
		new_gate.apply(init.weight_init)
		with torch.no_grad():
			new_gate.weight[:old_K].copy_(old_gate.weight)
		new_gate.to(device=device, dtype=dtype)
		self.gate = new_gate

		self.num_experts = new_K

	def freeze_experts(self, indices):
		"""Disable gradient updates for the experts at `indices`.
		They still produce outputs during forward (and are still routable
		by the gate); they simply don't update."""
		for i in indices:
			for p in self.experts[int(i)].parameters():
				p.requires_grad = False

	def unfreeze_all_experts(self):
		"""Mark every expert trainable again (mostly for debugging)."""
		for E in self.experts:
			for p in E.parameters():
				p.requires_grad = True
