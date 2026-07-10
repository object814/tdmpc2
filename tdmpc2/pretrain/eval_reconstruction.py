"""Visualise encoder→decoder reconstruction from a pretraining checkpoint.

Rolls one scripted-policy episode on a (by default non-eval) Meta-World task,
encodes+decodes every frame, and writes a side-by-side video:
    top row    = the N input cameras
    bottom row = their reconstructions

Reads the *training* checkpoint (`<out>.ckpt`), which holds BOTH the encoder
and the decoder — the deliverable `encoder.pt` has only the encoder. Safe to
run while pretraining is ongoing (checkpoints are written atomically); by
default it runs the nets on CPU so it doesn't compete for training GPU memory
(MuJoCo still renders via EGL on the GPU).

Usage (inside the container / tdmpc2 env, from tdmpc2/tdmpc2):
    python pretrain/eval_reconstruction.py \
        --ckpt /Metaworld/third_party/tdmpc2/pretrain_data/encoder_ms19.pt.ckpt \
        --out  /Metaworld/third_party/tdmpc2/pretrain_data/recon_eval.mp4
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("EGL_LOG_LEVEL", "fatal")

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent                       # tdmpc2 source root
sys.path.insert(0, str(_PARENT))

from common import layers
from pretrain.pretrain_encoder import build_cfg, encode
from pretrain.decoder import Decoder
from pretrain.collect_data import (
    _make_env, _load_policy, _to_model_obs, default_pretrain_tasks,
    POLICY_REGISTRY,
)


def _label(img, text):
    """Optionally draw a small caption; no-op if cv2 is unavailable."""
    try:
        import cv2
        return cv2.putText(np.ascontiguousarray(img), text, (6, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    except Exception:  # noqa: BLE001
        return img


def _panel(rgb_in, rgb_rec, num_cameras):
    """(C,H,W) input + recon tensors in [0,1] -> stacked HWC uint8 frame."""
    c, h, w = rgb_in.shape
    cin = rgb_in.view(num_cameras, 3, h, w)
    crec = rgb_rec.view(num_cameras, 3, h, w)
    row_in = torch.cat([cin[k] for k in range(num_cameras)], dim=-1)
    row_re = torch.cat([crec[k] for k in range(num_cameras)], dim=-1)
    top = (row_in.permute(1, 2, 0).numpy() * 255).astype("uint8")
    bot = (row_re.permute(1, 2, 0).numpy() * 255).astype("uint8")
    top = _label(top, "input")
    bot = _label(bot, "recon")
    return np.concatenate([top, bot], axis=0)


def main(args):
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu")
    if not (isinstance(ck, dict) and "decoder" in ck and "encoder" in ck):
        sys.exit("--ckpt must be the training checkpoint (*.ckpt) that holds "
                 "BOTH encoder and decoder; encoder.pt has only the encoder.")
    print(f">>> checkpoint: epoch={ck.get('epoch')} step={ck.get('step')} "
          f"best_val={ck.get('best_val')}")

    num_cameras = len(args.cameras)
    cfg = build_cfg(args.model_size, args.image_size,
                    img_channels=3 * num_cameras, state_dim=7)
    encoder = layers.enc(cfg).to(device).eval()
    decoder = Decoder(cfg).to(device).eval()
    encoder.load_state_dict(ck["encoder"])
    decoder.load_state_dict(ck["decoder"])

    # Pick a task (default: random non-eval pretraining task).
    task = args.task
    if task is None:
        rng = random.Random(args.seed)
        task = rng.choice(default_pretrain_tasks())
    if task not in POLICY_REGISTRY:
        sys.exit(f"no scripted policy for task '{task}'")
    print(f">>> task: {task}  | device={device}  | max_steps={args.max_steps}")

    inner_max = args.max_episode_steps * args.action_repeat
    env = _make_env(task, args.cameras, args.image_size, inner_max)
    policy = _load_policy(task)

    obs, info = env.reset(seed=args.seed)
    frames, rgb_errs = [], []
    for t in range(args.max_steps):
        action = np.asarray(policy.get_action(obs["original_obs"]), dtype=np.float32)
        state, rgb = _to_model_obs(obs)             # rgb uint8 CHW [0,255]
        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        r = torch.from_numpy(rgb).float().unsqueeze(0).to(device)
        with torch.no_grad():
            z = encode(encoder, s, r)
            rgb_hat, _ = decoder(z)
        rin = (r[0] / 255.0).clamp(0, 1).cpu()
        rrec = (rgb_hat[0] + 0.5).clamp(0, 1).cpu()
        rgb_errs.append(float((rin - rrec).pow(2).mean()))
        frames.append(_panel(rin, rrec, num_cameras))

        done = False
        for _ in range(args.action_repeat):
            obs, _rew, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if done:
                break
        if done:
            break
    env.close()

    mean_err = float(np.mean(rgb_errs)) if rgb_errs else float("nan")
    print(f">>> rolled {len(frames)} frames | mean per-pixel recon MSE "
          f"(display space) = {mean_err:.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import imageio
    try:
        imageio.mimsave(out, frames, fps=args.fps)
        print(f">>> wrote {out} ({len(frames)} frames, {args.fps} fps)")
    except Exception as e:  # noqa: BLE001 — ffmpeg may be missing; fall back to gif
        gif = out.with_suffix(".gif")
        imageio.mimsave(gif, frames, fps=args.fps)
        print(f">>> mp4 failed ({e}); wrote {gif} instead")


def build_argparser():
    p = argparse.ArgumentParser(description="Reconstruction video from a pretrain ckpt.")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Training checkpoint *.ckpt (holds encoder+decoder).")
    p.add_argument("--out", type=str,
                   default="../pretrain_data/recon_eval.mp4",
                   help="Output video path (.mp4; falls back to .gif).")
    p.add_argument("--task", type=str, default=None,
                   help="env_name to roll out; default = random non-eval task.")
    p.add_argument("--model-size", type=int, default=19,
                   help="Must match the checkpoint's training model_size.")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--cameras", nargs="+",
                   default=["topview", "front", "gripperPOV"])
    p.add_argument("--max-steps", type=int, default=250,
                   help="Max outer (action-repeated) steps to record.")
    p.add_argument("--max-episode-steps", type=int, default=250)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu (default; avoids training-GPU contention) or cuda:0.")
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
