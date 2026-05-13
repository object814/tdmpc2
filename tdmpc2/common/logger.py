import dataclasses
import os
import datetime
import re

import numpy as np
import pandas as pd
import torch
from termcolor import colored

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def _to_scalar(self, value):
		"""Convert logged values to JSON-safe Python scalars."""
		if isinstance(value, torch.Tensor):
			if value.numel() == 1:
				return value.detach().cpu().item()
			return value.detach().cpu().mean().item()
		if isinstance(value, np.generic):
			return value.item()
		return value

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		self.wandb_run_name = cfg.get("wandb_run_name", f"{cfg.task}-seed{cfg.seed}")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._wandb = None
			self._video = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		# Optional sweep-harness overrides. cfg.get(..., None) tolerates missing
		# keys for backward compat with hand-written scripts.
		pinned_id = cfg.get("wandb_run_id", None)
		group_override = cfg.get("wandb_group", None)
		tags_override = cfg.get("wandb_tags", None)

		# Resolve the wandb run id with logdir-based persistence:
		#   1. explicit cfg.wandb_run_id wins (sweep harness / manual override)
		#   2. else: re-use {work_dir}/wandb_run_id.txt if present (resume)
		#   3. else: let wandb generate a fresh id, then persist it after init
		# so subsequent re-launches against the same logdir auto-attach.
		run_id_file = self._log_dir / "wandb_run_id.txt"
		if not pinned_id and run_id_file.exists():
			try:
				pinned_id = run_id_file.read_text().strip() or None
				if pinned_id:
					print(colored(
						f"Resuming wandb run id from {run_id_file}: {pinned_id}",
						"blue", attrs=["bold"]))
			except Exception as e:
				print(colored(f"Failed to read {run_id_file}: {e}", "red"))
				pinned_id = None

		init_kwargs = dict(
			project=self.project,
			entity=self.entity,
			name=self.wandb_run_name,
			group=group_override or self._group,
			tags=list(tags_override) if tags_override else (cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"]),
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		if pinned_id:
			init_kwargs["id"] = pinned_id
			# `allow` covers first launch (creates run with our id) and resume
			# (re-attaches). `must` would reject the first-launch case.
			init_kwargs["resume"] = "allow"
		wandb.init(**init_kwargs)
		# Persist the wandb-assigned run id so the next launch with the same
		# logdir resumes against the same wandb run. No-op when the file
		# already exists (resume case) or wandb didn't yield an id.
		try:
			wid = getattr(getattr(wandb, "run", None), "id", None)
			if wid and not run_id_file.exists():
				run_id_file.write_text(str(wid))
				print(colored(f"Saved wandb run id to {run_id_file}: {wid}",
				              "blue"))
		except Exception as e:
			print(colored(f"Failed to persist wandb run id: {e}", "red"))
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise ValueError(f"invalid log format type: {ty}")

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		print(colored(f'Evaluated agent on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
		dmcontrol_reward = []
		metaworld_reward = []
		metaworld_success = []
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if task in TASK_SET['mt30'] and k.startswith('episode_reward'): # DMControl
				dmcontrol_reward.append(v)
				print(colored(f'  {task:<22}\tR: {v:.01f}', 'yellow'))
			elif task in TASK_SET['mt80'] and task not in TASK_SET['mt30']: # Meta-World
				if k.startswith('episode_reward'):
					metaworld_reward.append(v)
				elif k.startswith('episode_success'):
					metaworld_success.append(v)
					print(colored(f'  {task:<22}\tS: {v:.02f}', 'yellow'))
		dmcontrol_reward = np.nanmean(dmcontrol_reward)
		d['episode_reward+avg_dmcontrol'] = dmcontrol_reward
		print(colored(f'  {"dmcontrol":<22}\tR: {dmcontrol_reward:.01f}', 'yellow', attrs=['bold']))
		if cfg.task == 'mt80':
			metaworld_reward = np.nanmean(metaworld_reward)
			metaworld_success = np.nanmean(metaworld_success)
			d['episode_reward+avg_metaworld'] = metaworld_reward
			d['episode_success+avg_metaworld'] = metaworld_success
			print(colored(f'  {"metaworld":<22}\tR: {metaworld_reward:.01f}', 'yellow', attrs=['bold']))
			print(colored(f'  {"metaworld":<22}\tS: {metaworld_success:.02f}', 'yellow', attrs=['bold']))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = self._to_scalar(v)
			self._wandb.log(_d, step=self._to_scalar(d[xkey]))
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		self._print(d, category)
