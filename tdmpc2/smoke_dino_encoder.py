"""Smoke test for the DINO+PRISM hybrid encoder (encoder_type=dino).

Checks, without touching any env or training loop:
  1. WorldModel builds with encoder_type=dino (MoE dynamics/reward intact).
  2. The frozen DINOv2 backbone weights SURVIVE `WorldModel.apply(weight_init)`
     (compared tensor-by-tensor against the raw safetensors checkpoint).
  3. Trainable vs frozen parameter split is correct (backbone frozen, pooling
     + head + fusion + state branch trainable).
  4. encode() shapes: [B] and [T, B] multimodal dict obs -> latent on the
     SimNorm simplex.
  5. Gradients flow into the pooling/head/fusion but NOT into the backbone.
  6. (CUDA only, --timing) per-update encode throughput at the real training
     volume: (horizon+1) x batch_size frames x num_cameras ViT forwards.

Run inside the container (CPU login node is fine, GPU node for --timing):
    cd /Metaworld/third_party/tdmpc2/tdmpc2
    python smoke_dino_encoder.py
    python smoke_dino_encoder.py --timing        # on a GPU node
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from omegaconf import OmegaConf
from common import MODEL_SIZE, layers  # noqa: E402

CONFIG_PATH = _HERE / "config.yaml"


def build_cfg(args):
    """Minimal cfg reproducing a single-task multimodal PRISM run with the
    DINO encoder (mirrors pretrain/pretrain_encoder.py:build_cfg + the
    WorldModel-required fields)."""
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.model_size = int(args.model_size)
    for k, v in MODEL_SIZE[int(args.model_size)].items():
        cfg[k] = v
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.obs = "rgb"
    num_cameras = 3
    cfg.obs_shape = {
        "state": [7],
        "rgb": [3 * num_cameras, int(cfg.image_size), int(cfg.image_size)],
    }
    cfg.action_dim = 4
    cfg.use_moe = True
    cfg.num_experts = 3
    cfg.use_orthogonal = True
    cfg.encoder_type = "dino"
    cfg.dino_pool = args.pool
    cfg.dino_amp = not args.no_amp
    cfg.steps = 200_000
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", type=int, default=19)
    p.add_argument("--pool", type=str, default="attn", choices=["attn", "mean", "cls"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--timing", action="store_true",
                   help="Time the per-update encode volume (CUDA only).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--horizon", type=int, default=3)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    cfg = build_cfg(args)
    print(f">>> device={device} model_size={cfg.model_size} "
          f"latent_dim={cfg.latent_dim} enc_dim={cfg.enc_dim} pool={args.pool}")

    # --- 1. Build the full world model (runs apply(weight_init) internally) --
    from common.world_model import WorldModel
    torch.manual_seed(0)
    wm = WorldModel(cfg).to(device)
    dino_enc = wm._encoder["rgb"]
    print(f">>> encoder rgb branch: {dino_enc}")
    assert isinstance(dino_enc, layers.DinoRGBEncoder), type(dino_enc)

    # --- 2. Backbone weights survived apply(weight_init) --------------------
    from safetensors.torch import load_file
    weights_path = layers._default_dino_weights(str(cfg.dino_model))
    raw = load_file(str(weights_path))
    checked = 0
    for name, param in dino_enc.backbone.named_parameters():
        if name in raw:
            ref = raw[name].to(param.device, param.dtype)
            if ref.shape != param.shape:      # e.g. resampled pos_embed
                continue
            assert torch.allclose(param, ref), \
                f"backbone param {name} was modified after loading (weight_init leak?)"
            checked += 1
    assert checked > 50, f"only {checked} backbone params matched checkpoint keys"
    print(f">>> [OK] {checked} backbone tensors identical to checkpoint "
          f"(weight_init did not clobber them)")

    # --- 3. Trainable / frozen split -----------------------------------------
    frozen = sum(p.numel() for p in dino_enc.backbone.parameters())
    assert all(not p.requires_grad for p in dino_enc.backbone.parameters()), \
        "backbone has trainable params!"
    enc_trainable = sum(p.numel() for p in wm._encoder.parameters() if p.requires_grad)
    assert enc_trainable > 0, "no trainable params in encoder head"
    assert not dino_enc.backbone.training, "backbone not pinned to eval"
    wm.train()
    assert not dino_enc.backbone.training, "backbone left eval mode under wm.train()"
    wm.eval()
    print(f">>> [OK] frozen backbone {frozen/1e6:.2f}M | "
          f"trainable encoder (pool+head+state+fusion) {enc_trainable/1e6:.2f}M | "
          f"world model trainable total {wm.total_params/1e6:.2f}M")

    # --- 4. encode() shapes ---------------------------------------------------
    B, T = 4, cfg.horizon if hasattr(cfg, "horizon") else 3
    H = int(cfg.image_size)
    obs = {
        "state": torch.randn(B, 7, device=device),
        "rgb": torch.randint(0, 256, (B, 9, H, H), device=device, dtype=torch.uint8),
    }
    z = wm.encode(obs, task=None)
    assert z.shape == (B, cfg.latent_dim), z.shape
    groups = z.view(B, -1, cfg.simnorm_dim).sum(-1)
    assert torch.allclose(groups, torch.ones_like(groups), atol=1e-4), \
        "latent not on the SimNorm simplex"
    obs_seq = {
        "state": torch.randn(T, B, 7, device=device),
        "rgb": torch.randint(0, 256, (T, B, 9, H, H), device=device, dtype=torch.uint8),
    }
    z_seq = wm.encode(obs_seq, task=None)
    assert z_seq.shape == (T, B, cfg.latent_dim), z_seq.shape
    # And a dynamics step through the MoE for good measure.
    a = torch.randn(B, cfg.action_dim, device=device).clamp(-1, 1)
    z_next = wm.next(z, a, task=None)
    assert z_next.shape == (B, cfg.latent_dim), z_next.shape
    print(f">>> [OK] encode [B] -> {tuple(z.shape)}, [T,B] -> {tuple(z_seq.shape)}, "
          f"MoE next() -> {tuple(z_next.shape)}; latent on simplex")

    # --- 5. Gradient flow -------------------------------------------------------
    wm.train()
    wm.zero_grad(set_to_none=True)
    z = wm.encode(obs, task=None)
    z.sum().backward()
    assert all(p.grad is None for p in dino_enc.backbone.parameters()), \
        "gradients leaked into the frozen backbone"
    head_named = [(n, p) for n, p in wm._encoder.named_parameters() if p.requires_grad]
    missing = [n for n, p in head_named if p.grad is None]
    assert not missing, f"trainable encoder params with no grad: {missing}"
    wm.zero_grad(set_to_none=True)
    wm.eval()
    print(f">>> [OK] grads reach all {len(head_named)} trainable encoder tensors, "
          f"none reach the backbone")

    # --- 6. Optional throughput timing ------------------------------------------
    if args.timing:
        assert device.type == "cuda", "--timing needs a GPU"
        Bt, Tt = args.batch_size, args.horizon + 1
        big = {
            "state": torch.randn(Tt, Bt, 7, device=device),
            "rgb": torch.randint(0, 256, (Tt, Bt, 9, H, H), device=device, dtype=torch.uint8),
        }
        with torch.no_grad():
            wm.encode(big, task=None)      # warmup
            torch.cuda.synchronize()
            t0 = time.time()
            iters = 5
            for _ in range(iters):
                wm.encode(big, task=None)
            torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        n_vit = Tt * Bt * dino_enc.num_cameras
        print(f">>> [timing] per-update encode volume ({Tt}x{Bt} frames, "
              f"{n_vit} ViT forwards): {dt*1e3:.0f} ms/iter "
              f"(amp={'on' if dino_enc.use_amp else 'off'})")

    print(">>> ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
