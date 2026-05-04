import os
import re
import uuid
from pathlib import Path

import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# `cfg.buffer_storage_device` overrides the auto-heuristic. Useful when
		# you want to force GPU regardless of free-memory estimates (and accept
		# an OOM rather than silent fallback to CPU on a slow re-launch).
		desired = str(self.cfg.get('buffer_storage_device', 'auto') or 'auto').lower()
		if desired in ('cuda', 'cuda:0', 'gpu'):
			storage_device = 'cuda:0'
			print(f'Using CUDA memory for storage (forced via buffer_storage_device={desired!r}).')
		elif desired == 'cpu':
			storage_device = 'cpu'
			print(f'Using CPU memory for storage (forced via buffer_storage_device={desired!r}).')
		else:
			storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
			print(f'Using {storage_device.upper()} memory for storage (auto).')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		return self._prepare_batch(td)

	# ------------------------------------------------------------------
	#  On-disk episode persistence (mirrors STORM's train_eps/ scheme).
	#
	#  File layout:
	#    {episode_dir}/{episode_idx:09d}_{length}_{uuid}.pt
	#
	#  - episode_idx prefix lets us load in chronological order with a sort.
	#  - length is encoded so prune_episode_dir_to_cap can compute total
	#    transitions without opening every file.
	#  - uuid keeps filenames unique even on rapid same-step writes.
	# ------------------------------------------------------------------
	_EP_FILENAME_RE = re.compile(r'^(\d{9})_(\d+)_[0-9a-f]{12}\.pt$')

	@staticmethod
	def _episode_filename(episode_idx, length):
		uid = uuid.uuid4().hex[:12]
		return f'{int(episode_idx):09d}_{int(length)}_{uid}.pt'

	@staticmethod
	def _parse_episode_filename(name):
		m = Buffer._EP_FILENAME_RE.match(name)
		if not m:
			return None
		return int(m.group(1)), int(m.group(2))   # (ep_idx, length)

	@staticmethod
	def save_episode(td, episode_dir, episode_idx):
		"""Write one completed episode to ``episode_dir/{idx}_{len}_{uuid}.pt``.

		Atomic via tmp-file + rename so partial writes never get loaded back.
		``td`` should be a 1-D TensorDict on CPU (the trainer constructs CPU
		TensorDicts already; calling .cpu() here is cheap if not).
		"""
		episode_dir = Path(episode_dir)
		episode_dir.mkdir(parents=True, exist_ok=True)
		length = int(td.batch_size[0]) if td.batch_size else int(td['reward'].shape[0])
		fname = Buffer._episode_filename(episode_idx, length)
		path = episode_dir / fname
		tmp = episode_dir / (fname + '.tmp')
		torch.save(td.cpu(), tmp)
		os.replace(tmp, path)
		return path

	def load_from_directory(self, episode_dir, max_total_steps=None):
		"""Reload episodes from ``episode_dir`` back into the buffer.

		Newest episodes win when total transitions would exceed
		``max_total_steps`` — matches STORM's "keep most-recent N transitions"
		semantic. Returns a stats dict with keys ``transitions_restored``,
		``episodes_restored``, and ``kept_files`` (the filenames kept).
		"""
		episode_dir = Path(episode_dir)
		if not episode_dir.is_dir():
			return {'transitions_restored': 0, 'episodes_restored': 0, 'kept_files': set()}
		entries = []
		for fname in os.listdir(episode_dir):
			parsed = self._parse_episode_filename(fname)
			if parsed is None:
				continue
			ep_idx, length = parsed
			entries.append((ep_idx, length, fname))
		if not entries:
			return {'transitions_restored': 0, 'episodes_restored': 0, 'kept_files': set()}

		# Newest-first by ep_idx — drop oldest until we fit under the cap.
		entries.sort(key=lambda x: x[0], reverse=True)
		cap = max_total_steps if max_total_steps is not None else float('inf')
		kept = []
		running = 0
		for ep_idx, length, fname in entries:
			if running + length > cap:
				continue
			kept.append((ep_idx, length, fname))
			running += length

		# Replay in chronological order so internal episode counters advance
		# the way they would have during live training.
		kept.sort(key=lambda x: x[0])
		transitions = 0
		for _ep_idx, length, fname in kept:
			td = torch.load(episode_dir / fname, map_location='cpu', weights_only=False)
			# `add()` re-stamps the episode field with the current _num_eps,
			# so internal sampling stays consistent regardless of the
			# original ep_idx.
			self.add(td)
			transitions += length

		return {
			'transitions_restored': transitions,
			'episodes_restored': len(kept),
			'kept_files': {x[2] for x in kept},
		}

	@staticmethod
	def prune_episode_dir_to_cap(episode_dir, max_total_steps):
		"""FIFO-prune oldest episodes (by ep_idx) until total transitions ≤ cap.

		Mirrors STORM's prune_episode_dir_to_cap: disk-side bound matches the
		in-memory ring-buffer cap so disk usage tracks RAM usage. Returns the
		number of files removed.
		"""
		episode_dir = Path(episode_dir)
		if not episode_dir.is_dir():
			return 0
		entries = []
		for fname in os.listdir(episode_dir):
			parsed = Buffer._parse_episode_filename(fname)
			if parsed is None:
				continue
			ep_idx, length = parsed
			entries.append((ep_idx, length, fname))
		if not entries:
			return 0
		entries.sort(key=lambda x: x[0])  # oldest first
		total = sum(length for _, length, _ in entries)
		removed = 0
		i = 0
		while total > max_total_steps and i < len(entries):
			_idx, length, fname = entries[i]
			try:
				os.remove(episode_dir / fname)
				removed += 1
				total -= length
			except FileNotFoundError:
				pass
			i += 1
		return removed

	@staticmethod
	def erase_over_episode_files(episode_dir, kept_filenames):
		"""Delete files in ``episode_dir`` whose name is NOT in ``kept_filenames``.

		Used after a load to drop episodes that didn't make it back into the
		(bounded) in-memory ring. Returns the number of files removed.
		"""
		episode_dir = Path(episode_dir)
		if not episode_dir.is_dir():
			return 0
		removed = 0
		for fname in os.listdir(episode_dir):
			if Buffer._parse_episode_filename(fname) is None:
				continue
			if fname in kept_filenames:
				continue
			try:
				os.remove(episode_dir / fname)
				removed += 1
			except FileNotFoundError:
				pass
		return removed
