import torch.nn as nn


def weight_init(m):
	"""Custom weight initialization for TD-MPC2.

	Modules flagged with `_no_reinit = True` are skipped — used by frozen
	pretrained backbones (e.g. the DINOv2 encoder) whose weights are loaded
	at construction time and must survive `WorldModel.apply(weight_init)`.
	"""
	if getattr(m, '_no_reinit', False):
		return
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Embedding):
		nn.init.uniform_(m.weight, -0.02, 0.02)
	elif isinstance(m, nn.ParameterList):
		for i,p in enumerate(m):
			if p.dim() == 3: # Linear
				nn.init.trunc_normal_(p, std=0.02) # Weight
				nn.init.constant_(m[i+1], 0) # Bias


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)
