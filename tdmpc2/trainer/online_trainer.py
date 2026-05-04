import json
import os
from pathlib import Path
from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		# Episode persistence settings (default-on, opt-out via cfg).
		self._save_episodes = bool(self.cfg.get('save_episodes', True))
		ep_dir = self.cfg.get('episode_dir', None)
		if ep_dir is None or ep_dir == '':
			ep_dir = Path(self.cfg.work_dir) / 'train_eps'
		self._episode_dir = Path(ep_dir)
		self._manifest_path = Path(self.cfg.work_dir) / 'manifest.json'
		# Disk cap mirrors the in-memory ring cap so footprint tracks RAM.
		self._buffer_cap = int(min(self.cfg.buffer_size, self.cfg.steps))
		# Throttle pruning: doing it every episode is fine but a bit wasteful.
		self._prune_every_n_eps = max(1, int(getattr(self.cfg, 'prune_every_n_episodes', 5)))

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(
				{k: v.unsqueeze(0).cpu() for k, v in obs.items()},
				batch_size=(1,),
				device='cpu',
			)
		elif hasattr(obs, 'keys') and not isinstance(obs, torch.Tensor):
			# Already a TensorDict
			obs = obs.unsqueeze(0).cpu() if obs.batch_size == () else obs.cpu()
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	# ------------------------------------------------------------------
	#  Resume / manifest helpers
	# ------------------------------------------------------------------
	def _save_manifest(self):
		"""Persist enough state alongside the latest ckpt so the next launch
		can pick up where this one left off (model + episodes already on disk
		are restored separately)."""
		manifest = {
			'step': int(self._step),
			'ep_idx': int(self._ep_idx),
			'wandb_run_id': getattr(getattr(self.logger, '_wandb', None), 'run', None) and self.logger._wandb.run.id,
		}
		try:
			self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
			tmp = self._manifest_path.with_suffix('.json.tmp')
			tmp.write_text(json.dumps(manifest, indent=2))
			tmp.replace(self._manifest_path)
		except Exception as e:
			print(f'[OnlineTrainer] manifest save failed: {e}')

	def _load_manifest(self):
		if not self._manifest_path.exists():
			return None
		try:
			return json.loads(self._manifest_path.read_text())
		except Exception as e:
			print(f'[OnlineTrainer] manifest load failed: {e}')
			return None

	@staticmethod
	def _latest_ckpt(model_dir):
		"""Return (path, step) of the highest-step `<step>.pt` in model_dir, or None."""
		model_dir = Path(model_dir)
		if not model_dir.is_dir():
			return None
		best = None
		for fname in os.listdir(model_dir):
			if not fname.endswith('.pt'):
				continue
			stem = fname[:-3]
			if not stem.isdigit():
				continue
			step = int(stem)
			if best is None or step > best[1]:
				best = (model_dir / fname, step)
		return best

	def resume_from(self, work_dir):
		"""If a previous run left a model checkpoint + manifest under ``work_dir``,
		load them and replay any saved episodes back into the buffer. Safe to
		call when nothing is there — it just no-ops.
		"""
		work_dir = Path(work_dir)
		manifest = self._load_manifest()
		latest = self._latest_ckpt(work_dir / 'models')
		if manifest is None or latest is None:
			print('[OnlineTrainer] no resume state found, starting fresh.')
			return False

		ckpt_path, ckpt_step = latest
		print(f'[OnlineTrainer] resuming: agent ckpt step={ckpt_step} ({ckpt_path.name}), '
		      f'manifest step={manifest.get("step")}, ep_idx={manifest.get("ep_idx")}')
		self.agent.load(ckpt_path)
		self._step = int(manifest.get('step', ckpt_step))
		self._ep_idx = int(manifest.get('ep_idx', 0))

		if self._save_episodes:
			stats = self.buffer.load_from_directory(
				self._episode_dir, max_total_steps=self._buffer_cap)
			print(f'[OnlineTrainer] replayed {stats["transitions_restored"]:,} transitions '
			      f'from {stats["episodes_restored"]} episodes (cap {self._buffer_cap:,}).')
			# Keep disk in sync with what made it back into the in-memory ring.
			if stats['kept_files']:
				dropped = self.buffer.erase_over_episode_files(
					self._episode_dir, stats['kept_files'])
				if dropped:
					print(f'[OnlineTrainer] pruned {dropped} stale episode files post-load.')
		return True

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Checkpoint agent periodically (world model + actor + critics live in a single state_dict)
			if self.cfg.save_freq > 0 and self._step > 0 and self._step % self.cfg.save_freq == 0:
				self.logger.save_agent(self.agent, identifier=self._step)
				self._save_manifest()

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					ep_td = torch.cat(self._tds)
					self._ep_idx = self.buffer.add(ep_td)
					# Persist the episode and FIFO-prune so on-disk episodes
					# stay ≤ buffer_size transitions. Same contract as STORM.
					if self._save_episodes:
						try:
							self.buffer.save_episode(ep_td, self._episode_dir, self._ep_idx)
							if self._ep_idx % self._prune_every_n_eps == 0:
								self.buffer.prune_episode_dir_to_cap(
									self._episode_dir, self._buffer_cap)
						except Exception as e:
							print(f'[OnlineTrainer] save_episode failed: {e}')

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += 1

		self.logger.finish(self.agent)
