"""Training agent for PRISM-WM continual learning.

Owns two independent nn.Module attributes:
    self.backbone       — PrismBackbone (shared, persists across tasks)
    self.task_modules   — TaskModules (replaced at every task boundary)

Three optimizers:
    self.backbone_optim   — trains encoder + dynamics MoE (unfrozen experts).
                            Persists across task boundaries; Adam state for
                            encoder / gate / dynamics-head carries over.
                            Frozen experts are filtered out on rebuild.
    self.task_optim       — trains reward / Q / termination of the current
                            TaskModules. Fresh per task.
    self.pi_optim         — trains the current task's policy. Fresh per task.

Task-boundary API:
    agent.start_new_task(t)   — freeze through t·K, open (t+1)·K slot,
                                build fresh TaskModules, rebuild task/pi optims.
    agent.save_task(dir)      — writes backbone.pt and task_modules.pt
                                INTO `dir` as two independent checkpoints.
    agent.load_backbone(p)    — load backbone weights + optim state from p.
    agent.load_task_modules(p)— load task_modules weights + optim state from p.

Cross-task eval:
    Build an EvalAgent (from `backbone.py`) with (backbone, loaded_task_modules,
    task_idx, active_K). NO weight swapping in the live agent.

Note: `torch.compile=False` is forced — `MaskedMoEBlock` mutates `_active_K`
across tasks and CUDA-graph capture would invalidate.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from termcolor import colored

from common import math

from progmoe_training.backbone import PrismBackbone, TaskModules


class ContinualTDMPC2(nn.Module):

	def __init__(self, cfg):
		super().__init__()

		if getattr(cfg, 'compile', False):
			print(colored(
				'>>> ContinualTDMPC2: forcing cfg.compile=False '
				'(MaskedMoEBlock._active_K mutates across tasks; '
				'cudagraph capture would invalidate).',
				'yellow',
			))
			cfg.compile = False
		self.cfg = cfg
		self.device = torch.device('cuda:0')

		# Shared backbone — built ONCE with total_K = num_tasks * K_per_task,
		# active_K starts at K_per_task (task-0 slot only).
		self.backbone = PrismBackbone(cfg).to(self.device)

		# Per-task heads — replaced at every task boundary.
		self.task_modules = TaskModules(cfg).to(self.device)

		# SDP-faithful (Plan A) freeze. It does NOT freeze the encoder (handled
		# by freeze_encoder) and does NOT freeze old experts' gate rows (so a
		# new task can still route to — reuse — old frozen experts, SDP's
		# transfer property). After task 0 it
		# freezes only the shared parts that would otherwise leak into old
		# tasks: the MoE output head, the gate's τ-context column, and the τ
		# schedule. Old-task routing columns are preserved for free by the
		# one-hot gate. Enabled via cfg.sdp_freeze.
		self.sdp_freeze = bool(getattr(cfg, 'sdp_freeze', False))
		# Set True at the task-1+ boundary under sdp_freeze; tells `_update` to
		# zero the gradient on gate.weight[:, -1] (the τ column) each step.
		self._freeze_gate_tau_col = False

		self._build_optims()

		# Debug from here --------------------------------------------------
		self._dump_trainable('init')
		# Debug ends here --------------------------------------------------

		# MPC bookkeeping.
		self.cfg.iterations += 2 * int(cfg.action_dim >= 20)
		self.discount = self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = nn.Buffer(torch.zeros(
			self.cfg.horizon, self.cfg.action_dim, device=self.device))

		self.eval_mode()

	# ------------------------------------------------------------------
	# Optimizers
	# ------------------------------------------------------------------
	def _build_optims(self):
		"""Build (or rebuild) the three optimizers from current trainable
		params. Frozen experts and masked experts are filtered out.

		Note: `_build_optims` is also called after task boundaries, which
		causes `backbone_optim` Adam state to be reset. The trainer reloads
		`backbone_optim.state_dict()` from disk right after rebuilding to
		preserve encoder / gate / dynamics-head momentum across tasks.
		"""
		def trainable(params):
			return [p for p in params if p.requires_grad]

		bb_groups = []
		# Per-task projectors belong to the encoder group: they are part of the
		# observation->latent path and want the same enc_lr_scale. Frozen ones
		# (completed tasks) drop out here via the requires_grad filter, exactly
		# as frozen experts do.
		enc_params = list(self.backbone._encoder.parameters())
		if self.backbone._projectors is not None:
			enc_params += list(self.backbone._projectors.parameters())
		enc = trainable(enc_params)
		if enc:
			bb_groups.append({
				'params': enc,
				'lr': self.cfg.lr * self.cfg.enc_lr_scale,
			})
		dyn = trainable(self.backbone._dynamics.parameters())
		if dyn:
			bb_groups.append({'params': dyn})
		self.backbone_optim = torch.optim.Adam(
			bb_groups, lr=self.cfg.lr, capturable=True)

		task_groups = []
		rew_p = trainable(self.task_modules._reward.parameters())
		if rew_p:
			task_groups.append({'params': rew_p})
		q_p = trainable(self.task_modules._Qs.parameters())
		if q_p:
			task_groups.append({'params': q_p})
		if self.cfg.episodic and self.task_modules._termination is not None:
			term_p = trainable(self.task_modules._termination.parameters())
			if term_p:
				task_groups.append({'params': term_p})
		self.task_optim = torch.optim.Adam(
			task_groups, lr=self.cfg.lr, capturable=True)

		pi_p = trainable(self.task_modules._pi.parameters())
		self.pi_optim = torch.optim.Adam(
			pi_p, lr=self.cfg.lr, eps=1e-5, capturable=True)

	# Debug from here ------------------------------------------------------
	def _dump_trainable(self, tag):
		"""Print which backbone params are trainable + which are in the
		backbone optimizer. Used to verify sdp_freeze /
		progmoe expert freezing actually took effect at each boundary."""
		total, train = 0, 0
		by_mod = {}
		for name, p in self.backbone.named_parameters():
			total += p.numel()
			if p.requires_grad:
				train += p.numel()
				key = name.split('.')[0]
				by_mod[key] = by_mod.get(key, 0) + p.numel()
		print(
			f'[debug:{tag}] backbone trainable {train}/{total} :: {by_mod} '
			f':: active_K={self.backbone.active_K} '
			f'frozen_K={self.backbone.frozen_K} '
			f'task_idx={self.backbone.current_task()}',
			flush=True,
		)
		opt_ids = {
			id(p) for g in self.backbone_optim.param_groups for p in g['params']
		}
		leaked = [
			n for n, p in self.backbone.named_parameters()
			if (not p.requires_grad) and id(p) in opt_ids
		]
		if leaked:
			print(
				f'[debug:{tag}] WARN: frozen-but-in-backbone_optim: '
				f'{leaked[:8]}',
				flush=True,
			)
	# Debug ends here ------------------------------------------------------

	# ------------------------------------------------------------------
	# Train / eval mode helpers — toggles both submodules together.
	# ------------------------------------------------------------------
	def eval_mode(self):
		self.backbone.eval()
		self.task_modules.eval()

	def train_mode(self):
		self.backbone.train()
		self.task_modules.train()

	@property
	def plan(self):
		return self._plan

	def _get_discount(self, episode_length):
		frac = episode_length / self.cfg.discount_denom
		return min(max((frac - 1) / frac, self.cfg.discount_min),
		           self.cfg.discount_max)

	# ------------------------------------------------------------------
	# Task-boundary API
	# ------------------------------------------------------------------
	def set_current_task(self, task_idx):
		self.backbone.set_current_task(task_idx)

	def current_task(self):
		return self.backbone.current_task()

	def start_new_task(self, task_idx):
		"""Task-boundary transition for task t (t ≥ 1).

		progressive backbone (progmoe):
		  1. Freeze experts through `t · K_per_task`.
		  2. Open `_active_K = (t + 1) · K_per_task` slot.
		static backbone (naive sequential):
		  - The MoE expert window is left untouched (all `total_K` experts
		    stay active + trainable). No freezing, no masking.

		Both modes then:
		  3. Switch the routing one-hot to task t.
		  4. Drop the old TaskModules; build a fresh one.
		  5. Rebuild backbone / task / π optimizers (frozen experts excluded).
		  6. Zero MPC prev-mean buffer.

		The trainer separately restores `backbone_optim.state_dict()` from
		the previous task's checkpoint right after this call, to preserve
		Adam momentum on the shared params (encoder / gate / dynamics-head)."""
		if self.backbone.progressive:
			K = self.backbone.K_per_task
			# 1) freeze prior, then 2) open the new slot.
			# Order matters: freeze first to keep [0, t·K) frozen even though
			# set_active_K below also touches grad flags.
			self.backbone.freeze_experts_through(task_idx * K)
			# 1.5) When sdp_freeze is enabled, lock τ BEFORE
			# set_active_K so its _reset_tau_schedule call becomes a no-op
			# (otherwise τ resets to tau_init each boundary and task-0
			# routing silently drifts; see debug_drift findings).
			if self.sdp_freeze and task_idx >= 1:
				self.backbone._dynamics._tau_frozen = True
			self.backbone.set_active_K((task_idx + 1) * K)
		# 2.5) Freeze the per-task rgb projectors of every completed task, so
		# the frozen experts of tasks [0, task_idx) keep receiving exactly the
		# representation they were trained against. Done in BOTH progressive
		# and static modes: the drift it prevents is a property of the shared
		# projector, not of expert masking.
		self.backbone.freeze_projectors_through(task_idx)
		# 3) Switch routing one-hot.
		self.backbone.set_current_task(task_idx)
		# 3.6) SDP-faithful (Plan A) freeze: after task 0 freeze the shared MoE
		# output head and flag the gate's τ-context column for per-step grad
		# zeroing. We deliberately do NOT touch the encoder (frozen separately
		# via freeze_encoder) or old experts' gate rows (leaving them trainable
		# lets the new task's own gate column learn to route to old frozen
		# experts — SDP reusability). The τ scalar is locked above.
		self._freeze_gate_tau_col = False
		if self.sdp_freeze and task_idx >= 1:
			for p in self.backbone._dynamics.head.parameters():
				p.requires_grad = False
			self._freeze_gate_tau_col = True
		# 4) Fresh TaskModules.
		self.task_modules = TaskModules(self.cfg).to(self.device)
		# 5) Rebuild optims (filter dropps frozen encoder/head/experts).
		self._build_optims()
		# 6) Reset MPC state.
		self._prev_mean.zero_()
		self.eval_mode()

		# Debug from here --------------------------------------------------
		self._dump_trainable(f'start_new_task(t={task_idx})')
		# Debug ends here --------------------------------------------------

	# ------------------------------------------------------------------
	# Save / load — two independent files per task
	# ------------------------------------------------------------------
	def save_task(self, task_logdir):
		"""Write `backbone.pt` and `task_modules.pt` separately into
		`task_logdir`. State_dicts are DISJOINT (TaskModules has no
		backbone reference), so loading one never overwrites the other."""
		task_logdir = Path(task_logdir)
		task_logdir.mkdir(parents=True, exist_ok=True)
		torch.save({
			'model': self.backbone.state_dict(),
			'optim': self.backbone_optim.state_dict(),
		}, task_logdir / 'backbone.pt')
		torch.save({
			'model': self.task_modules.state_dict(),
			'task_optim': self.task_optim.state_dict(),
			'pi_optim': self.pi_optim.state_dict(),
			'prev_mean': self._prev_mean.detach().cpu(),
		}, task_logdir / 'task_modules.pt')

	def load_backbone(self, backbone_path, load_optim=True):
		ck = torch.load(
			backbone_path, map_location=self.device, weights_only=False)
		# Backward-compat: pre-fix checkpoints predate `_dynamics._tau_buf`
		# being registered as a buffer (τ was a plain float). For such
		# checkpoints, the saved state_dict is missing that key; substitute
		# the freshly-initialized value (tau_init) so strict loading still
		# works and τ effectively defaults to its init for legacy runs.
		sd = ck['model']
		tau_key = '_dynamics._tau_buf'
		if tau_key not in sd:
			sd[tau_key] = self.backbone._dynamics._tau_buf.detach().clone()
		self.backbone.load_state_dict(sd)
		if load_optim and 'optim' in ck:
			try:
				self._load_optim_checked(
					self.backbone_optim, ck['optim'], 'backbone_optim')
			except Exception as e:
				print(colored(
					f'[warn] backbone_optim.load_state_dict failed ({e}); '
					f'continuing with fresh optim state.', 'yellow'))

	@staticmethod
	def _load_optim_checked(optim, state, name):
		"""Restore optimizer state ONLY if it matches the live parameters.

		`Optimizer.load_state_dict` matches saved state to parameters purely by
		positional index within each param_group -- it does NOT verify shapes.
		A checkpoint written when the group had a different composition (e.g.
		a different number of unfrozen per-task projectors, or a different
		expert freeze boundary) therefore loads "successfully" and then blows
		up much later inside `adam.step()` with a shape mismatch that points at
		the optimizer internals rather than at the real cause.

		Validate up front and fall back to fresh moments, which costs a little
		Adam warm-up but never corrupts training.
		"""
		live = [p for g in optim.param_groups for p in g['params']]
		saved_groups = state.get('param_groups', [])
		saved_ids = [i for g in saved_groups for i in g['params']]
		if len(live) != len(saved_ids):
			print(colored(
				f'[{name}] checkpoint has {len(saved_ids)} params but the live '
				f'optimizer has {len(live)} — the trainable set changed since '
				f'the snapshot. Starting with fresh optimizer state.', 'yellow'))
			return False
		saved_state = state.get('state', {})
		for p, sid in zip(live, saved_ids):
			st = saved_state.get(sid) or saved_state.get(str(sid))
			if not st:
				continue
			for key in ('exp_avg', 'exp_avg_sq'):
				buf = st.get(key)
				if buf is not None and tuple(buf.shape) != tuple(p.shape):
					print(colored(
						f'[{name}] shape mismatch on {key}: checkpoint '
						f'{tuple(buf.shape)} vs live {tuple(p.shape)}. '
						f'Starting with fresh optimizer state.', 'yellow'))
					return False
		optim.load_state_dict(state)
		return True

	def load_task_modules(self, task_modules_path, load_optim=True):
		ck = torch.load(
			task_modules_path, map_location=self.device, weights_only=False)
		self.task_modules.load_state_dict(ck['model'])
		if load_optim:
			if 'task_optim' in ck:
				try:
					self._load_optim_checked(
						self.task_optim, ck['task_optim'], 'task_optim')
				except Exception as e:
					print(colored(
						f'[warn] task_optim.load_state_dict failed ({e}).',
						'yellow'))
			if 'pi_optim' in ck:
				try:
					self._load_optim_checked(
						self.pi_optim, ck['pi_optim'], 'pi_optim')
				except Exception as e:
					print(colored(
						f'[warn] pi_optim.load_state_dict failed ({e}).',
						'yellow'))
		if 'prev_mean' in ck:
			self._prev_mean.copy_(ck['prev_mean'].to(self.device))

	# Compatibility shim — sequential_train.SequentialLogger.save_agent
	# calls `agent.save(fp)` with a single .pt path. We rewrite it to dump
	# the same content into a sibling directory.
	def save(self, fp):
		fp = Path(fp)
		# Use the file's stem as a sibling dir: `models/final.pt`
		# → `models/final/`.
		task_dir = fp.with_suffix('')
		self.save_task(task_dir)
		# Drop a tiny pointer file at `fp` so downstream "checkpoint exists"
		# probes succeed.
		torch.save({'task_dir': str(task_dir)}, fp)

	def load(self, fp):
		"""Symmetric counterpart to `save(fp)`. Reads the pointer at `fp`
		and loads backbone + task_modules from the sibling directory."""
		ck = torch.load(fp, map_location='cpu', weights_only=False)
		task_dir = Path(ck['task_dir']) if 'task_dir' in ck else Path(fp).with_suffix('')
		self.load_backbone(task_dir / 'backbone.pt', load_optim=True)
		self.load_task_modules(task_dir / 'task_modules.pt', load_optim=True)

	# ------------------------------------------------------------------
	# Inference: act + MPC
	# ------------------------------------------------------------------
	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""Match TDMPC2.act signature. `task` arg is ignored — task identity
		comes from `self.backbone.current_task_one_hot`."""
		if isinstance(obs, dict):
			obs = {k: v.to(self.device, non_blocking=True).unsqueeze(0)
			       for k, v in obs.items()}
		elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		else:
			obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if self.cfg.mpc:
			return self._plan(obs, t0=t0, eval_mode=eval_mode).cpu()
		z = self.backbone.encode(obs)
		action, info = self.task_modules.pi(z)
		if eval_mode:
			action = info['mean']
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, *, backbone=None, task_modules=None):
		backbone = backbone if backbone is not None else self.backbone
		task_modules = task_modules if task_modules is not None else self.task_modules
		G, discount = 0, 1
		termination = torch.zeros(
			z.shape[0], 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(
				task_modules.reward(z, actions[t]), self.cfg)
			z = backbone.next(z, actions[t])
			G = G + discount * (1 - termination) * reward
			discount = discount * self.discount
			if self.cfg.episodic:
				termination = torch.clip(
					termination
					+ (task_modules.termination(z) > 0.5).float(),
					max=1.,
				)
		action, _ = task_modules.pi(z)
		return G + discount * (1 - termination) * task_modules.Q(
			z, action, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False,
	          *, backbone=None, task_modules=None, prev_mean=None):
		backbone = backbone if backbone is not None else self.backbone
		task_modules = task_modules if task_modules is not None else self.task_modules
		prev_mean = prev_mean if prev_mean is not None else self._prev_mean

		z = backbone.encode(obs)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(
				self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim,
				device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon - 1):
				pi_actions[t], _ = task_modules.pi(_z)
				_z = backbone.next(_z, pi_actions[t])
			pi_actions[-1], _ = task_modules.pi(_z)

		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(
			self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full(
			(self.cfg.horizon, self.cfg.action_dim),
			self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = prev_mean[1:]
		actions = torch.empty(
			self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim,
			device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		for _ in range(self.cfg.iterations):
			r = torch.randn(
				self.cfg.horizon,
				self.cfg.num_samples - self.cfg.num_pi_trajs,
				self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample

			value = self._estimate_value(
				z, actions, backbone=backbone, task_modules=task_modules,
			).nan_to_num(0)
			elite_idxs = torch.topk(
				value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value = value[elite_idxs]
			elite_actions = actions[:, elite_idxs]

			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature * (elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) \
				/ (score.sum(0) + 1e-9)
			std = (
				(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2)
				.sum(dim=1) / (score.sum(0) + 1e-9)
			).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)

		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	# ------------------------------------------------------------------
	# Training update
	# ------------------------------------------------------------------
	def update_pi(self, zs):
		action, info = self.task_modules.pi(zs)
		qs = self.task_modules.Q(zs, action, return_type='avg', detach=True)
		self.task_modules.scale.update(qs[0])
		qs = self.task_modules.scale(qs)

		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (
			-(self.cfg.entropy_coef * info['scaled_entropy'] + qs).mean(dim=(1, 2))
			* rho
		).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(
			self.task_modules._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		return TensorDict({
			'pi_loss': pi_loss,
			'pi_grad_norm': pi_grad_norm,
			'pi_entropy': info['entropy'],
			'pi_scaled_entropy': info['scaled_entropy'],
			'pi_scale': self.task_modules.scale.value,
		})

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated):
		action, _ = self.task_modules.pi(next_z)
		return reward + self.discount * (1 - terminated) * self.task_modules.Q(
			next_z, action, return_type='min', target=True)

	def _update(self, obs, action, reward, terminated, task=None):
		# `task` (optional, [B] LongTensor): per-sample task ids for the
		# merged-buffer (multitask-style) update. When None the dynamics
		# routes via the global one-hot (original behavior). The encoder and
		# the per-task heads are task-agnostic in Setup A, so `task` only
		# affects the MoE dynamics routing in the rollout below.
		# Compute targets
		with torch.no_grad():
			next_z = self.backbone.encode(obs[1:])
			td_targets = self._td_target(next_z, reward, terminated)

		self.train_mode()

		# Latent rollout
		zs = torch.empty(
			self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim,
			device=self.device)
		z = self.backbone.encode(obs[0])
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(
			zip(action.unbind(0), next_z.unbind(0))
		):
			z = self.backbone.next(z, _action, task_ids=task)
			consistency_loss = consistency_loss \
				+ F.mse_loss(z, _next_z) * self.cfg.rho ** t
			zs[t + 1] = z

		_zs = zs[:-1]
		qs = self.task_modules.Q(_zs, action, return_type='all')
		reward_preds = self.task_modules.reward(_zs, action)
		if self.cfg.episodic:
			termination_pred = self.task_modules.termination(zs[1:], unnormalized=True)

		reward_loss, value_loss = 0, 0
		for t, (rew_pred, rew, td, qss) in enumerate(zip(
			reward_preds.unbind(0), reward.unbind(0),
			td_targets.unbind(0), qs.unbind(1),
		)):
			reward_loss = reward_loss \
				+ math.soft_ce(rew_pred, rew, self.cfg).mean() * self.cfg.rho ** t
			for q in qss.unbind(0):
				value_loss = value_loss \
					+ math.soft_ce(q, td, self.cfg).mean() * self.cfg.rho ** t

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(
				termination_pred, terminated)
		else:
			termination_loss = 0.
		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss
			+ self.cfg.reward_coef * reward_loss
			+ self.cfg.termination_coef * termination_loss
			+ self.cfg.value_coef * value_loss
		)

		total_loss.backward()

		# SDP Plan A: freeze the gate's shared τ-context column (last input
		# feature) so it can't drift old-task routing. Each task's own column
		# still trains (the one-hot zeroes every other column's gradient), so a
		# new task remains free to route to old + new experts.
		if self._freeze_gate_tau_col:
			gate_w = self.backbone._dynamics.gate.weight
			if gate_w.grad is not None:
				gate_w.grad[:, -1].zero_()

		# Single global grad-clip across all model params that contributed
		# to total_loss (i.e. union of backbone_optim and task_optim param
		# groups). π parameters are clipped separately in update_pi.
		all_params = []
		for g in self.backbone_optim.param_groups:
			all_params.extend(g['params'])
		for g in self.task_optim.param_groups:
			all_params.extend(g['params'])
		grad_norm = torch.nn.utils.clip_grad_norm_(
			all_params, self.cfg.grad_clip_norm)

		self.backbone_optim.step()
		self.task_optim.step()
		self.backbone_optim.zero_grad(set_to_none=True)
		self.task_optim.zero_grad(set_to_none=True)

		pi_info = self.update_pi(zs.detach())
		self.task_modules.soft_update_target_Q()

		self.eval_mode()
		info = TensorDict({
			'consistency_loss': consistency_loss,
			'reward_loss': reward_loss,
			'value_loss': value_loss,
			'termination_loss': termination_loss,
			'total_loss': total_loss,
			'grad_norm': grad_norm,
		})
		if self.cfg.episodic:
			info.update(math.termination_statistics(
				torch.sigmoid(termination_pred[-1]), terminated[-1]))
		info.update(pi_info)
		return info.detach().mean()

	def update(self, buffer):
		obs, action, reward, terminated, task = buffer.sample()
		torch.compiler.cudagraph_mark_step_begin()
		info = self._update(obs, action, reward, terminated, task=None)
		return info.clone()

	def update_merged(self, obs, action, reward, terminated, task_ids):
		"""Single full-loss update on a pre-sampled MERGED batch whose rows mix
		several tasks (each tagged by `task_ids`, an [B] LongTensor).

		Unlike `update`, the dynamics rollout routes PER SAMPLE via `task_ids`
		(so task-j rows flow through task-j's frozen experts and task-t rows
		through the live task-t experts), while the per-task heads (the current
		`task_modules`) are applied to every row — exactly DreamerV3's recipe of
		training the current heads on the task-mixed batch with the full loss.
		The merged batch is assembled by the driver from the current-task
		buffer + the anchored ER buffers; this method just runs the update."""
		torch.compiler.cudagraph_mark_step_begin()
		info = self._update(obs, action, reward, terminated, task=task_ids)
		return info.clone()

	# ------------------------------------------------------------------
	# Anchored + routing-aware ER update (world-model only)
	# ------------------------------------------------------------------
	def _aux_update(self, obs, action, reward, terminated, task_idx,
	                teacher_modules=None,
	                reward_anchor_coef=0.0,
	                termination_anchor_coef=0.0):
		"""World-model-only update on an old-task ER batch.

		- Routes via task `task_idx`'s one-hot (so the FROZEN task-`task_idx`
		  experts produce the features that flow into the dynamics head). In
		  progressive mode the active expert window is also temporarily
		  scoped to `[0, (task_idx+1)·K_per_task)` so future / new-task
		  experts neither produce features nor receive gradient.
		- Backprops ONLY into the backbone optimizer (encoder + dynamics
		  MoE). The agent's per-task heads (`task_modules`: reward / Q / π)
		  are NOT updated.

		Optional supervised anchors (when `teacher_modules` is provided and
		the corresponding coef > 0):
		  - `reward_anchor_coef`: soft-CE between the *frozen* teacher's
		    reward head applied to the LIVE rollout latent and the true
		    reward labels stored in the ER batch. Gradient flows THROUGH the
		    frozen teacher weights into the encoder + dynamics, pinning the
		    live latent to remain interpretable by task-j's original reward
		    head. Teacher weights themselves never update (requires_grad=False).
		  - `termination_anchor_coef`: same idea for the termination head
		    (BCE-with-logits). Only active when cfg.episodic and the teacher
		    has a termination head.
		"""
		if self.backbone.progressive:
			active_K_j = (task_idx + 1) * self.backbone.K_per_task
		else:
			active_K_j = self.backbone.total_K

		with self.backbone.scoped(active_K=active_K_j, task_idx=task_idx):
			with torch.no_grad():
				next_z = self.backbone.encode(obs[1:])

			self.train_mode()

			zs = torch.empty(
				self.cfg.horizon + 1, self.cfg.batch_size, self.cfg.latent_dim,
				device=self.device,
			)
			z = self.backbone.encode(obs[0])
			zs[0] = z
			consistency_loss = 0
			for t, (_action, _next_z) in enumerate(
				zip(action.unbind(0), next_z.unbind(0))
			):
				z = self.backbone.next(z, _action)
				consistency_loss = consistency_loss \
					+ F.mse_loss(z, _next_z) * self.cfg.rho ** t
				zs[t + 1] = z

			consistency_loss = consistency_loss / self.cfg.horizon
			loss = self.cfg.consistency_coef * consistency_loss

			# Supervised anchors via frozen teacher heads (optional).
			reward_anchor_loss = torch.zeros((), device=self.device)
			termination_anchor_loss = torch.zeros((), device=self.device)
			use_reward_anchor = (
				teacher_modules is not None and reward_anchor_coef > 0
			)
			use_term_anchor = (
				teacher_modules is not None
				and termination_anchor_coef > 0
				and self.cfg.episodic
				and getattr(teacher_modules, '_termination', None) is not None
			)
			if use_reward_anchor:
				_zs = zs[:-1]
				rew_pred = teacher_modules.reward(_zs, action)
				ra = 0
				for t, (rp, r) in enumerate(zip(
					rew_pred.unbind(0), reward.unbind(0),
				)):
					ra = ra + math.soft_ce(rp, r, self.cfg).mean() \
						* self.cfg.rho ** t
				reward_anchor_loss = ra / self.cfg.horizon
				loss = loss + reward_anchor_coef * reward_anchor_loss
			if use_term_anchor:
				term_pred = teacher_modules.termination(
					zs[1:], unnormalized=True)
				termination_anchor_loss = F.binary_cross_entropy_with_logits(
					term_pred, terminated,
				)
				loss = loss + termination_anchor_coef * termination_anchor_loss

			loss.backward()
			# Only step the backbone optimizer — task_modules stays clean.
			bb_params = [
				p for g in self.backbone_optim.param_groups for p in g['params']
			]
			grad_norm = torch.nn.utils.clip_grad_norm_(
				bb_params, self.cfg.grad_clip_norm,
			)
			self.backbone_optim.step()
			# Zero gradients on every optimizer so leftover grads on
			# task_optim / pi_optim don't leak into the next main step.
			self.backbone_optim.zero_grad(set_to_none=True)
			self.task_optim.zero_grad(set_to_none=True)
			self.pi_optim.zero_grad(set_to_none=True)

			self.eval_mode()

		return TensorDict({
			'consistency_loss': consistency_loss.detach(),
			'reward_anchor_loss': reward_anchor_loss.detach(),
			'termination_anchor_loss': termination_anchor_loss.detach(),
			'grad_norm': grad_norm.detach(),
			'task_idx': torch.tensor(
				float(task_idx), device=self.device),
		})

	def aux_update(self, buffer, task_idx, teacher_modules=None,
	               reward_anchor_coef=0.0, termination_anchor_coef=0.0):
		"""Sample one batch from `buffer` (a task-`task_idx` aux buffer) and
		do one anchored + routing-aware backbone update.

		`teacher_modules` (optional): a frozen `TaskModules` for task
		`task_idx`, used as a teacher for the optional reward / termination
		anchors. Pass None (or leave coefs at 0) to retain the original
		consistency-only behavior."""
		obs, action, reward, terminated, _task = buffer.sample()
		torch.compiler.cudagraph_mark_step_begin()
		info = self._aux_update(
			obs, action, reward, terminated, task_idx,
			teacher_modules=teacher_modules,
			reward_anchor_coef=reward_anchor_coef,
			termination_anchor_coef=termination_anchor_coef,
		)
		return info.detach().mean().clone()
