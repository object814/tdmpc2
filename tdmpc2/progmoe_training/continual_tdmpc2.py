"""TDMPC2 variant for progressive-MoE continual learning.

Builds a `ContinualPrismWorldModel`, exposes `expand_for_new_task` and
`set_current_task` for the training script, and rebuilds the Adam
optimizers from `requires_grad=True` parameters after each expansion
(so frozen experts don't accumulate stale momentum / variance state).

`cfg.compile=False` is forced — `torch.compile(_update, mode=
'reduce-overhead')` captures a CUDA graph that becomes invalid the
moment we mutate `self.model._dynamics.experts` (ModuleList growth).
"""

import torch
from termcolor import colored

from tdmpc2 import TDMPC2
from common.scale import RunningScale

from progmoe_training.continual_world_model import ContinualPrismWorldModel


class ContinualTDMPC2(TDMPC2):

	def __init__(self, cfg):
		# 1) Compile would freeze the cudagraph against the current MoE
		# expert count. Mutating the ModuleList at a task boundary breaks
		# capture. Force-off, with a single console warning.
		if getattr(cfg, 'compile', False):
			print(colored(
				'>>> ContinualTDMPC2: forcing cfg.compile=False '
				'(MoE growth incompatible with cudagraph capture).',
				'yellow',
			))
			cfg.compile = False

		# 2) Skip TDMPC2.__init__; re-implement minimally with our world
		# model. Everything else (plan, _update, target-Q polyak, scale)
		# is inherited as-is.
		torch.nn.Module.__init__(self)
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = ContinualPrismWorldModel(cfg).to(self.device)
		self._build_optims()
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2 * int(cfg.action_dim >= 20)
		self.discount = self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(
			torch.zeros(self.cfg.horizon, self.cfg.action_dim,
			            device=self.device))
		# (Intentionally NOT calling torch.compile on _update or _plan.)

	# ------------------------------------------------------------------
	# Optimizer (re)build. Filters by `requires_grad` so frozen experts
	# don't allocate Adam state. Called once at __init__ and once after
	# each `expand_for_new_task`.
	# ------------------------------------------------------------------
	def _build_optims(self):
		def trainable(params):
			return [p for p in params if p.requires_grad]

		groups = []
		enc = trainable(self.model._encoder.parameters())
		if enc:
			groups.append({
				'params': enc,
				'lr': self.cfg.lr * self.cfg.enc_lr_scale,
			})
		for name in ('_dynamics', '_reward'):
			p = trainable(getattr(self.model, name).parameters())
			if p:
				groups.append({'params': p})
		if self.cfg.episodic and getattr(self.model, '_termination', None):
			p = trainable(self.model._termination.parameters())
			if p:
				groups.append({'params': p})
		qp = trainable(self.model._Qs.parameters())
		if qp:
			groups.append({'params': qp})

		self.optim = torch.optim.Adam(
			groups, lr=self.cfg.lr, capturable=True)

		pi_p = trainable(self.model._pi.parameters())
		self.pi_optim = torch.optim.Adam(
			pi_p, lr=self.cfg.lr, eps=1e-5, capturable=True)

	# ------------------------------------------------------------------
	# Task-boundary API used by the continual training script.
	# ------------------------------------------------------------------
	def expand_for_new_task(self, k_new=4):
		"""Add `k_new` experts to both MoE blocks, freeze the experts
		that existed prior to this call, move new params onto the agent's
		device, then rebuild the optimizers (so frozen experts are
		dropped from the param groups)."""
		self.model.expand_for_new_task(k_new)
		# add_experts' inner `.to(device)` already handles the new gate
		# + new experts; this is a belt-and-braces idempotent move.
		self.model.to(self.device)
		self._build_optims()

	def set_current_task(self, task_idx):
		self.model.set_current_task(task_idx)

	def current_task(self):
		return self.model.current_task()
