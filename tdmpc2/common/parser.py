import dataclasses
import re
from pathlib import Path
from typing import Any

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Converts an OmegaConf config to a dataclass object.
	This prevents graph breaks when used with torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass_name = "Config"
	dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	if OmegaConf.is_missing(cfg, "work_dir"):
		cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	else:
		wd = Path(cfg.work_dir)
		if not wd.is_absolute():
			wd = Path(hydra.utils.get_original_cwd()) / wd
		cfg.work_dir = wd
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v
		if cfg.task == 'mt30' and cfg.model_size == 19:
			cfg.latent_dim = 512 # This checkpoint is slightly smaller

	# DINO encoder: the frozen ViT dominates compute and blows up
	# torch.compile time (many guard variants x a 12-block ViT through
	# Inductor, re-paid per node since the Inductor cache is node-local),
	# while eager timm already uses fused SDPA attention. Force compile off.
	if str(cfg.get('encoder_type', 'default')) == 'dino' and cfg.get('compile', False):
		print('[dino] encoder_type=dino -> forcing compile=false '
		      '(ViT-dominated compute; compile costs >> gains here)')
		cfg.compile = False

	# Pretrained encoder: with freeze_encoder=true the encoder param group is
	# empty (requires_grad filter in TDMPC2.__init__), which knocks Dynamo off
	# its static-address fast path for Adam and crashes guard construction on
	# the Parameter-keyed optimizer state dict once it fills after the first
	# step (InternalTorchDynamoError: SyntaxError). Force compile off for all
	# pretrained-encoder runs so preEnc/preEncFr stay comparable to each other.
	if cfg.get('pretrained_encoder', None) and cfg.get('compile', False):
		print('[pretrain] pretrained_encoder set -> forcing compile=false '
		      '(Dynamo cannot guard the optimizer state dict with a frozen/'
		      'loaded encoder param group)')
		cfg.compile = False

	# Multi-task
	cfg.multitask = cfg.task in TASK_SET.keys()
	if cfg.multitask:
		cfg.task_title = cfg.task.upper()
		# Account for slight inconsistency in task_dim for the mt30 experiments
		cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
	else:
		cfg.task_dim = 0
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

	return cfg_to_dataclass(cfg)
