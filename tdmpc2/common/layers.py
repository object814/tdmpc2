import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules
from copy import deepcopy


class Ensemble(nn.Module):
	"""
	Vectorized ensemble of modules.
	"""

	def __init__(self, modules, **kwargs):
		super().__init__()
		# combine_state_for_ensemble causes graph breaks
		self.params = from_modules(*modules, as_module=True)
		with self.params[0].data.to("meta").to_module(modules[0]):
			self.module = deepcopy(modules[0])
		self._repr = str(modules[0])
		self._n = len(modules)

	def __len__(self):
		return self._n

	def _call(self, params, *args, **kwargs):
		with params.to_module(self.module):
			return self.module(*args, **kwargs)

	def forward(self, *args, **kwargs):
		return torch.vmap(self._call, (0, None), randomness="different")(self.params, *args, **kwargs)

	def __repr__(self):
		return f'Vectorized {len(self)}x ' + self._repr


class ShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad
		self.padding = tuple([self.pad] * 4)

	def forward(self, x):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		x = F.pad(x, self.padding, 'replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
		shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class PixelPreprocess(nn.Module):
	"""
	Normalizes pixel observations to [-0.5, 0.5].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x.div(255.).sub(0.5)


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation, and optionally dropout.
	"""

	def __init__(self, *args, dropout=0., act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act
		self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

	def forward(self, x):
		x = super().forward(x)
		if self.dropout:
			x = self.dropout(x)
		return self.act(self.ln(x))

	def __repr__(self):
		repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}{repr_dropout}, "\
			f"act={self.act.__class__.__name__})"


class _GramSchmidt1D(nn.Module):
	"""Gram-Schmidt orthogonalization across the K expert dimension.

	Input  shape: [..., K, H]. Each of the N rows is treated as K vectors
	in ℝ^H; we orthonormalize the K vectors against each other along the
	feature axis. Used optionally inside `MoEBlock` to encourage functional
	independence between experts (PRISM-WM, `OrthogonalLayer1D`).
	"""

	def forward(self, x):
		# x: [..., K, H]. Iterate along K; project each new vector onto the
		# orthonormal span of the previous basis vectors and renormalize.
		K = x.shape[-2]
		v0 = x[..., 0:1, :]
		basis = v0 / (v0.norm(dim=-1, keepdim=True) + 1e-8)             # [..., 1, H]
		for i in range(1, K):
			v = x[..., i:i+1, :]                                         # [..., 1, H]
			# coeffs[..., 0, j] = <v, basis_j>;  proj = Σ_j coeffs_j basis_j
			coeffs = torch.matmul(v, basis.transpose(-1, -2))            # [..., 1, J]
			proj = torch.matmul(coeffs, basis)                           # [..., 1, H]
			w = v - proj
			w = w / (w.norm(dim=-1, keepdim=True) + 1e-8)
			basis = torch.cat([basis, w], dim=-2)                        # [..., J+1, H]
		return basis


class MoEBlock(nn.Module):
	"""
	Mixture-of-Experts block, faithful to PRISM-WM's official implementation
	(arXiv:2512.08411, `core/common/layers.py:MoEBlock`).

	Layout:
	    expert_input = [z, a, task_emb?]                          (per-expert)
	    gate_input   = [task_emb if multitask else [z, a],  τ ]   (router)
	    feats        = stack_K( expert_k(expert_input) )           [..., K, H]
	    feats        = GramSchmidt(feats)                          (if use_orthogonal)
	    weights      = softmax( gate(gate_input) / τ )             [..., K]
	    out          = head( Σ_k weights_k · feats_k )

	τ is an entropy-feedback scheduled scalar (not learned by SGD):
	  - held at `tau_init` for the first `freeze_frac` of training
	  - then `τ ← τ + β · tanh(H_target − H(weights))`, clipped to
	    [tau_min, tau_max]. `H_target` defaults to 0.75·log(K).
	  - also injected as a scalar column into the gate input as a phase
	    indicator (per the official code).

	The router input dim is configured via `gate_dim` independent of
	`in_dim`, so the caller can implement "gate sees task only / experts
	see state+action+task" — matching the official multitask wiring.
	"""

	def __init__(
		self,
		in_dim,
		hidden_dim,
		out_dim,
		num_experts=4,
		gate_dim=None,
		act=None,
		dropout=0.,
		use_orthogonal=False,
		tau_init=1.8,
		tau_min=0.5,
		tau_max=2.0,
		beta=0.02,
		H_target=None,
		freeze_frac=0.05,
		total_steps=200_000,
	):
		super().__init__()
		self.num_experts = int(num_experts)
		self.in_dim = int(in_dim)
		self.gate_dim = int(gate_dim) if gate_dim is not None else int(in_dim)
		self.hidden_dim = int(hidden_dim)
		self.out_dim = int(out_dim)
		self.use_orthogonal = bool(use_orthogonal)

		# Experts (parameters live here; gate is separate below).
		self.experts = nn.ModuleList([
			nn.Sequential(
				NormedLinear(self.in_dim, hidden_dim, dropout=dropout),
				NormedLinear(hidden_dim, hidden_dim),
			) for _ in range(self.num_experts)
		])
		self._ortho = _GramSchmidt1D() if self.use_orthogonal else None

		# Gate takes (gate_dim + 1) features: gate_base ⊕ scalar τ context.
		self.gate = nn.Linear(self.gate_dim + 1, self.num_experts, bias=False)
		with torch.no_grad():
			self.gate.weight += 1e-3 * torch.randn_like(self.gate.weight)

		if act is not None:
			self.head = NormedLinear(hidden_dim, out_dim, act=act)
		else:
			self.head = nn.Linear(hidden_dim, out_dim)

		# τ schedule state.
		self.tau_init = float(tau_init)
		self.tau_min = float(tau_min)
		self.tau_max = float(tau_max)
		self.beta = float(beta)
		# τ as a buffer so it round-trips through state_dict (load/save now
		# carry task-0's converged τ across task boundaries). Read/written
		# as a Python float via the `tau` @property below — all existing
		# call sites (forward path, _update_tau, _reset_tau_schedule)
		# stay unchanged.
		self.register_buffer('_tau_buf', torch.tensor(float(tau_init)))
		# When True, _update_tau and _reset_tau_schedule become no-ops.
		# Set by ContinualTDMPC2 at the task-1+ boundary when
		# sdp_freeze=True so the gate's τ context stays locked
		# at task-0's value (otherwise the entropy-feedback schedule plus
		# set_active_K's reset would silently shift task-0 routing).
		self._tau_frozen = False
		self.H_target = float(H_target) if H_target is not None \
			else 0.75 * math.log(self.num_experts)
		self.freeze_steps = max(5_000, int(freeze_frac * total_steps))
		self.register_buffer('_global_step', torch.tensor(0, dtype=torch.long))
		# Last-observed entropy, populated during training forwards. Public
		# attribute (no underscore) for diagnostics.
		self.last_entropy = None

	# Scalar accessors so existing `self.tau` / `self.tau = ...` /
	# `self.tau += ...` call sites keep working unchanged.
	@property
	def tau(self):
		return float(self._tau_buf.item())

	@tau.setter
	def tau(self, v):
		self._tau_buf.fill_(float(v))

	# --- τ schedule helpers ------------------------------------------------
	def _update_tau(self, entropy_val: float):
		"""Entropy-feedback step: push τ toward the value that yields
		entropy ≈ H_target. Skipped during the freeze phase, and skipped
		entirely when _tau_frozen is set."""
		if self._tau_frozen:
			return
		if int(self._global_step) < self.freeze_steps:
			return
		self.tau += self.beta * math.tanh(self.H_target - entropy_val)
		self.tau = float(max(self.tau_min, min(self.tau_max, self.tau)))

	def _reset_tau_schedule(self):
		"""Reset τ, step counter, and H_target. Called by set_active_K when
		experts are activated so the schedule re-enters its soft phase
		with the new K-aware target entropy. No-op when _tau_frozen is set
		(sdp_freeze boundaries preserve task-0's converged τ)."""
		if self._tau_frozen:
			return
		self.tau = self.tau_init
		self._global_step.zero_()
		self.H_target = 0.75 * math.log(self.num_experts)
		self.last_entropy = None

	# --- core routing ------------------------------------------------------
	def _prep_inputs(self, z, a, task_emb):
		"""Flatten optional (T, B, D) leading dims and broadcast task_emb if
		needed. Returns (expert_in, gate_base, restore_fn) where restore_fn
		reshapes a flat [N, ·] tensor back to [T, B, ·] if necessary."""
		is_seq = (z.ndim == 3)
		if is_seq:
			T, B, _ = z.shape
			if task_emb is not None and task_emb.ndim == 2:
				task_emb = task_emb.unsqueeze(0).expand(T, B, -1)
			z = z.reshape(T * B, -1)
			a = a.reshape(T * B, -1)
			if task_emb is not None:
				task_emb = task_emb.reshape(T * B, -1)
			restore = lambda y: y.view(T, B, -1)
		else:
			if task_emb is not None and task_emb.ndim == 2 and task_emb.shape[0] == 1:
				task_emb = task_emb.expand(z.shape[0], -1)
			restore = lambda y: y

		# Expert input: [z, a, task_emb?]
		expert_in = torch.cat([z, a] + ([task_emb] if task_emb is not None else []), dim=-1)
		# Gate base: task_emb (if multitask) else [z, a].
		gate_base = task_emb if task_emb is not None else torch.cat([z, a], dim=-1)
		return expert_in, gate_base, restore

	def _route(self, expert_in, gate_base):
		# Gate input = [gate_base, τ_scalar tiled per row].
		tau_col = torch.full_like(gate_base[..., :1], self.tau)
		gate_in = torch.cat([gate_base, tau_col], dim=-1)
		logits = self.gate(gate_in) / max(self.tau, 1e-6)            # [N, K]
		weights = F.softmax(logits, dim=-1)

		feats = torch.stack([E(expert_in) for E in self.experts], dim=-2)  # [N, K, H]
		if self._ortho is not None:
			feats = self._ortho(feats)

		# τ entropy-feedback update (training mode only).
		if self.training:
			with torch.no_grad():
				ent = -(weights * weights.clamp_min(1e-9).log()).sum(-1).mean()
			self.last_entropy = float(ent.item())
			self._update_tau(self.last_entropy)
			self._global_step += 1

		combined = (weights.unsqueeze(-1) * feats).sum(dim=-2)             # [N, H]
		return combined, weights, feats

	def forward(self, z, a, task_emb=None):
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, _, _ = self._route(expert_in, gate_base)
		return restore(self.head(combined))

	def forward_with_gate(self, z, a, task_emb=None):
		"""Returns (output, gate_weights). Weights have shape matching the
		output's leading dims, with an extra trailing K dimension."""
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, weights, _ = self._route(expert_in, gate_base)
		# Restore weights to the original leading-dim layout too.
		if z.ndim == 3:
			T, B, _ = z.shape
			weights = weights.view(T, B, -1)
		return restore(self.head(combined)), weights

	def forward_with_diagnostics(self, z, a, task_emb=None):
		"""Returns (output, gate_weights, expert_features).

		`expert_features` is the pre-aggregation per-expert feature stack
		([..., K, H]) — used by the evaluator to detect functional collapse
		(two experts producing nearly identical features even when the gate
		distributes evenly between them)."""
		expert_in, gate_base, restore = self._prep_inputs(z, a, task_emb)
		combined, weights, feats = self._route(expert_in, gate_base)
		if z.ndim == 3:
			T, B, _ = z.shape
			weights = weights.view(T, B, -1)
			feats = feats.view(T, B, *feats.shape[1:])
		return restore(self.head(combined)), weights, feats

	def __repr__(self):
		head_out = getattr(self.head, 'out_features', self.out_dim)
		return (
			f"MoEBlock(num_experts={self.num_experts}, in={self.in_dim}, "
			f"gate={self.gate_dim}, hidden={self.hidden_dim}, out={head_out}, "
			f"ortho={self.use_orthogonal})"
		)


def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations, and optionally dropout.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def conv(in_shape, num_channels, act=None):
	"""
	Basic convolutional encoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
	layers = [
		ShiftAug(), PixelPreprocess(),
		nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=False),
		nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()]
	if act:
		layers.append(act)
	return nn.Sequential(*layers)


def conv_deep(in_shape, num_channels, latent_dim, kernel_size=4, minres=4, act=None):
	"""
	DreamerV3-style convolutional encoder for larger images.
	Halves spatial dim per layer until it hits `minres`, doubling channels
	(capped at 8x base) along the way, then projects the flattened feature
	to `latent_dim` via a LayerNorm'd linear. Used for 128x128 RGB inputs
	and/or multi-modal observations where the latent must match a fixed
	`latent_dim` regardless of CNN output size.
	"""
	c, h, w = in_shape
	assert h == w, "conv_deep assumes square RGB inputs"
	assert h >= minres and (h // minres) > 0
	num_layers = int(math.log2(h // minres))
	assert 2 ** num_layers == (h // minres), \
		f"spatial {h} must downsample to minres={minres} by stride-2 convs"
	pad = max((kernel_size - 2) // 2, 0)
	layers_ = [ShiftAug(), PixelPreprocess()]
	in_c, out_c = c, num_channels
	for _ in range(num_layers):
		layers_.append(nn.Conv2d(in_c, out_c, kernel_size, stride=2, padding=pad))
		layers_.append(nn.GroupNorm(1, out_c))
		layers_.append(nn.SiLU(inplace=False))
		in_c = out_c
		out_c = min(out_c * 2, num_channels * 8)
	layers_.append(nn.Flatten())
	flat_dim = in_c * minres * minres
	layers_.append(nn.Linear(flat_dim, latent_dim))
	layers_.append(nn.LayerNorm(latent_dim))
	if act is not None:
		layers_.append(act)
	return nn.Sequential(*layers_)


class AttnPool(nn.Module):
	"""
	Learned attention pooling over a token set. `num_queries` learnable query
	vectors cross-attend over the input tokens; outputs are concatenated into
	a single flat vector per sample. Used to compress DINOv2 patch tokens
	into a compact per-camera feature while retaining spatial selectivity
	(DINO-WM shows global CLS-style features underperform patch features on
	precise manipulation).
	"""

	def __init__(self, dim, num_queries=4, num_heads=6):
		super().__init__()
		self.query = nn.Parameter(torch.randn(num_queries, dim) * dim ** -0.5)
		self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
		self.out_dim = num_queries * dim

	def forward(self, tokens):
		# tokens: [B, N, D] -> [B, num_queries * D]
		q = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
		out, _ = self.attn(q, tokens, tokens, need_weights=False)
		return out.flatten(1)


def dino_projector(in_dim, out_dim, cfg, act=None):
	"""Build the DINO rgb projector: pooled ViT features -> encoder out_dim.

	`cfg.dino_head_hidden > 0` (default) gives an MLP with one hidden layer
	(Linear -> LayerNorm -> Mish -> Linear -> LayerNorm [-> act]); the 4608->256
	single Linear it replaces was an 18x compression in one affine map, which
	bottlenecks everything the frozen backbone provides. Setting
	`dino_head_hidden: 0` restores that original single-Linear head so the two
	can be compared directly.

	`act` is the branch-dependent tail activation supplied by `enc()`:
	SimNorm for an rgb-only encoder (whose output IS the latent), None for the
	multimodal case (where the fusion layer applies SimNorm instead).
	"""
	hidden = int(getattr(cfg, 'dino_head_hidden', 0) or 0)
	if hidden > 0:
		layers_ = [
			nn.Linear(in_dim, hidden),
			nn.LayerNorm(hidden),
			nn.Mish(inplace=False),
			nn.Linear(hidden, out_dim),
			nn.LayerNorm(out_dim),
		]
	else:
		layers_ = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)]
	if act is not None:
		layers_.append(act)
	return nn.Sequential(*layers_)


def _default_dino_weights(model_name):
	"""Default local checkpoint path: <tdmpc2 repo root>/pretrain_data/dino/.

	Resolved relative to this file so it works both on the host and inside
	the apptainer container (the repo is bind-mounted wholesale)."""
	from pathlib import Path
	return Path(__file__).resolve().parents[2] / 'pretrain_data' / 'dino' / f'{model_name}.safetensors'


class DinoRGBEncoder(nn.Module):
	"""
	Frozen DINOv2 rgb branch (drop-in replacement for the `conv`/`conv_deep`
	branch built by `enc()`), plus a small trainable head.

	Pipeline per forward, for rgb input [B, 3*num_cameras, H, W] in [0, 255]:
	  1. ShiftAug on the raw camera stack (parity with the CNN branches).
	  2. Split cameras into the batch dim -> [B*cams, 3, H, W].
	  3. /255, bicubic resize to `cfg.dino_input_size`, ImageNet normalize.
	  4. Frozen DINOv2 forward under no_grad (optionally bf16 autocast)
	     -> patch tokens [B*cams, N, D].
	  5. Trainable pooling (`cfg.dino_pool`: attn | mean | cls) per camera.
	  6. Concat cameras -> Linear + LayerNorm (+ optional act) -> `out_dim`.

	Only the pooling + head parameters are trainable; the backbone is loaded
	from a local checkpoint, frozen (requires_grad=False), pinned to eval
	mode, and flagged `_no_reinit` so `WorldModel.apply(init.weight_init)`
	cannot clobber its weights.
	"""

	def __init__(self, cfg, out_dim, act=None):
		super().__init__()
		import timm

		in_shape = cfg.obs_shape['rgb']
		c, h, w = in_shape
		assert h == w, 'DinoRGBEncoder assumes square rgb inputs'
		assert c % 3 == 0, f'rgb channels must be a multiple of 3, got {c}'
		self.num_cameras = c // 3
		self.img_hw = h
		self.input_size = int(getattr(cfg, 'dino_input_size', 224))
		self.pool_type = str(getattr(cfg, 'dino_pool', 'attn'))
		assert self.pool_type in ('attn', 'mean', 'cls'), self.pool_type
		self.use_amp = bool(getattr(cfg, 'dino_amp', True))

		model_name = str(getattr(cfg, 'dino_model', 'vit_small_patch14_dinov2'))
		weights = getattr(cfg, 'dino_weights', None)
		if weights is None or str(weights).strip() in ('', 'null', 'None'):
			weights = _default_dino_weights(model_name)
		if not os.path.isfile(str(weights)):
			raise FileNotFoundError(
				f'DINOv2 checkpoint not found at {weights}. Download it once '
				f'(e.g. https://huggingface.co/timm/{model_name}.lvd142m) and '
				f'set cfg.dino_weights, or place it at the default path.')

		# timm resamples the (518px) pos_embed to the requested img_size grid
		# when loading the checkpoint, so smaller inputs (e.g. 224) just work.
		backbone = timm.create_model(
			model_name, pretrained=True, num_classes=0,
			img_size=self.input_size,
			pretrained_cfg_overlay=dict(file=str(weights)),
		)
		backbone.requires_grad_(False)
		backbone.eval()
		for m in backbone.modules():
			m._no_reinit = True   # survive WorldModel.apply(init.weight_init)
		self.backbone = backbone
		embed_dim = backbone.embed_dim
		self.num_prefix_tokens = backbone.num_prefix_tokens

		self.aug = ShiftAug()
		# ImageNet normalization — what DINOv2 was trained with.
		self.register_buffer('_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
		self.register_buffer('_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

		# Trainable head: pool per camera, concat cameras, project to out_dim.
		if self.pool_type == 'attn':
			num_queries = int(getattr(cfg, 'dino_pool_queries', 4))
			self.pool = AttnPool(embed_dim, num_queries=num_queries,
			                     num_heads=backbone.blocks[0].attn.num_heads)
			per_cam_dim = self.pool.out_dim
		else:
			self.pool = None
			per_cam_dim = embed_dim
		self.feature_dim = self.num_cameras * per_cam_dim
		self.out_dim = out_dim

		# When the backbone owns per-task projectors, this branch stops at the
		# pooled features and `PrismBackbone.encode` applies the projector for
		# the current task. Keeping the projector out here is what lets it be
		# frozen per task alongside that task's MoE experts.
		self.externalize_projector = bool(
			getattr(cfg, 'dino_per_task_projector', False))
		if self.externalize_projector:
			self.head = None
		else:
			self.head = dino_projector(self.feature_dim, out_dim, cfg, act=act)

	def train(self, mode=True):
		"""Keep the frozen backbone in eval mode regardless of train()."""
		super().train(mode)
		self.backbone.eval()
		return self

	def _backbone_tokens(self, x):
		"""Frozen DINOv2 forward. x: [N, 3, S, S] normalized. Returns the
		full token sequence [N, prefix+patches, D] in float32."""
		with torch.no_grad():
			if self.use_amp and x.is_cuda:
				with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
					feats = self.backbone.forward_features(x)
				feats = feats.float()
			else:
				feats = self.backbone.forward_features(x)
		return feats

	def forward(self, x):
		B = x.shape[0]
		x = self.aug(x)                                       # [B, 3*cams, H, W]
		x = x.view(B * self.num_cameras, 3, self.img_hw, self.img_hw)
		x = x.div(255.)
		if self.img_hw != self.input_size:
			x = F.interpolate(x, size=(self.input_size, self.input_size),
			                  mode='bicubic', align_corners=False)
		x = (x - self._mean) / self._std
		feats = self._backbone_tokens(x)                      # [B*cams, 1+N, D]
		if self.pool_type == 'cls':
			pooled = feats[:, 0]
		else:
			tokens = feats[:, self.num_prefix_tokens:]
			if self.pool_type == 'mean':
				pooled = tokens.mean(dim=1)
			else:
				pooled = self.pool(tokens)                    # [B*cams, Q*D]
		pooled = pooled.view(B, -1)                           # [B, cams*P]
		if self.head is None:
			# Per-task projector mode: return pooled features; the caller
			# (PrismBackbone.encode) applies the current task's projector.
			return pooled
		return self.head(pooled)

	def __repr__(self):
		n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
		n_frozen = sum(p.numel() for p in self.backbone.parameters())
		proj = 'per-task (in backbone)' if self.head is None else 'internal'
		return (f'DinoRGBEncoder(cameras={self.num_cameras}, '
		        f'input={self.img_hw}->{self.input_size}, pool={self.pool_type}, '
		        f'feat_dim={self.feature_dim}, projector={proj}, '
		        f'trainable={n_trainable/1e6:.2f}M, frozen={n_frozen/1e6:.2f}M)')


def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	If the observation is multi-modal (contains both 'state' and 'rgb'),
	adds a 'fusion' branch that concatenates per-modality features and
	projects to cfg.latent_dim with SimNorm.
	"""
	keys = list(cfg.obs_shape.keys())
	multimodal = ('state' in keys and 'rgb' in keys)
	kernel_size = int(getattr(cfg, 'cnn_kernel_size', 4))
	minres = int(getattr(cfg, 'cnn_minres', 4))

	for k in keys:
		if k == 'state':
			in_dim = cfg.obs_shape[k][0] + cfg.task_dim
			if multimodal:
				out[k] = mlp(in_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.enc_dim)
			else:
				out[k] = mlp(in_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
		elif k == 'rgb':
			c_shape = cfg.obs_shape[k]
			h = c_shape[-1]
			use_deep = multimodal or (h != 64)
			if multimodal:
				branch_act = None
				target_dim = cfg.enc_dim
			else:
				branch_act = SimNorm(cfg)
				target_dim = cfg.latent_dim
			if getattr(cfg, 'encoder_type', 'default') == 'dino':
				out[k] = DinoRGBEncoder(cfg, out_dim=target_dim, act=branch_act)
			elif use_deep:
				out[k] = conv_deep(
					c_shape,
					num_channels=cfg.num_channels,
					latent_dim=target_dim,
					kernel_size=kernel_size,
					minres=minres,
					act=branch_act,
				)
			else:
				out[k] = conv(c_shape, cfg.num_channels, act=branch_act)
		else:
			raise NotImplementedError(f"Encoder for observation type {k} not implemented.")

	if multimodal:
		out['fusion'] = NormedLinear(2 * cfg.enc_dim, cfg.latent_dim, act=SimNorm(cfg))
	return nn.ModuleDict(out)


def maybe_load_pretrained_encoder(encoder, cfg):
	"""Optionally initialise `encoder` from a pretrained checkpoint and freeze it.

	Reads two cfg keys (both default to off):
	  - cfg.pretrained_encoder: path to a checkpoint saved by
	    `pretrain/pretrain_encoder.py`. Accepts either a raw `enc()`
	    state_dict or a dict wrapping it under the key 'encoder'.
	  - cfg.freeze_encoder: when truthy, sets requires_grad=False on every
	    encoder parameter so it is excluded from the optimizer (both the
	    single-task and the sequential param-group builders filter on
	    requires_grad / no-grad params).

	Call this immediately after `encoder = layers.enc(cfg)`, BEFORE the model
	is moved to its device or any optimizer is built. Weights are loaded onto
	CPU; the caller's later `.to(device)` moves them. Returns a small status
	dict for logging. No-op (returns {'loaded': False, 'frozen': False}) when
	cfg.pretrained_encoder is unset.
	"""
	import torch

	path = getattr(cfg, 'pretrained_encoder', None)
	if path is None or str(path).strip() in ('', 'null', 'None'):
		return {'loaded': False, 'frozen': False, 'path': None}

	if getattr(cfg, 'encoder_type', 'default') == 'dino':
		raise ValueError(
			'cfg.pretrained_encoder cannot be combined with encoder_type=dino: '
			'the DINOv2 rgb branch loads its own frozen backbone weights and '
			'has a different architecture than the AE-pretrained encoder. '
			'Unset one of the two.')

	ckpt = torch.load(str(path), map_location='cpu')
	state = ckpt.get('encoder', ckpt) if isinstance(ckpt, dict) else ckpt
	missing, unexpected = encoder.load_state_dict(state, strict=False)
	if missing or unexpected:
		raise RuntimeError(
			f'Pretrained encoder at {path} does not match the current '
			f'encoder architecture.\n  missing keys:    {list(missing)}\n'
			f'  unexpected keys: {list(unexpected)}\n'
			'Verify image_size / cameras / num_channels / latent_dim / '
			'enc_dim match the values used during pretraining.')

	frozen = bool(getattr(cfg, 'freeze_encoder', False))
	if frozen:
		for p in encoder.parameters():
			p.requires_grad_(False)

	print(f'[encoder] loaded pretrained weights from {path} '
	      f'(frozen={frozen}).')
	return {'loaded': True, 'frozen': frozen, 'path': str(path)}


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Converts a checkpoint from our old API to the new torch.compile compatible API.
	"""
	# check whether checkpoint is already in the new format
	if "_detach_Qs_params.0.weight" in source_state_dict:
		return source_state_dict

	name_map = ['weight', 'bias', 'ln.weight', 'ln.bias']
	new_state_dict = dict()

	# rename keys
	for key, val in list(source_state_dict.items()):
		if key.startswith('_Qs.'):
			num = key[len('_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_Qs.params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val
			new_total_key = "_detach_Qs_params." + new_key
			new_state_dict[new_total_key] = val
		elif key.startswith('_target_Qs.'):
			num = key[len('_target_Qs.params.'):]
			new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
			new_total_key = "_target_Qs_params." + new_key
			del source_state_dict[key]
			new_state_dict[new_total_key] = val

	# add batch_size and device from target_state_dict to new_state_dict
	for prefix in ('_Qs.', '_detach_Qs_', '_target_Qs_'):
		for key in ('__batch_size', '__device'):
			new_key = prefix + 'params.' + key
			new_state_dict[new_key] = target_state_dict[new_key]

	# check that every key in new_state_dict is in target_state_dict
	for key in new_state_dict.keys():
		assert key in target_state_dict, f"key {key} not in target_state_dict"
	# check that all Qs keys in target_state_dict are in new_state_dict
	for key in target_state_dict.keys():
		if 'Qs' in key:
			assert key in new_state_dict, f"key {key} not in new_state_dict"
	# check that source_state_dict contains no Qs keys
	for key in source_state_dict.keys():
		assert 'Qs' not in key, f"key {key} contains 'Qs'"

	# copy log_std_min and log_std_max from target_state_dict to new_state_dict
	new_state_dict['log_std_min'] = target_state_dict['log_std_min']
	new_state_dict['log_std_dif'] = target_state_dict['log_std_dif']
	if '_action_masks' in target_state_dict:
		new_state_dict['_action_masks'] = target_state_dict['_action_masks']

	# copy new_state_dict to source_state_dict
	source_state_dict.update(new_state_dict)

	return source_state_dict
