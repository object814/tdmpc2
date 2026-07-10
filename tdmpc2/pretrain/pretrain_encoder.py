"""Pretrain the TD-MPC2 encoder as a plain autoencoder on offline Meta-World data.

Builds the SAME `layers.enc(cfg)` encoder used by the single-task and sequential
PRISM pipelines (with the model_size architecture overrides applied, so dims
match exactly), pairs it with a mirror `Decoder`, and minimises a reconstruction
loss over the HDF5 dataset from `collect_data.py`:

    z          = fusion(cat([ enc.state(state), enc.rgb(rgb) ]))
    rgb_hat,   = decoder.rgb(z)         # target: rgb/255 - 0.5  (PixelPreprocess)
    state_hat  = decoder.state(z)       # target: raw proprio
    loss       = MSE(rgb_hat, rgb_tgt) + state_coef * MSE(state_hat, state)

Only the ENCODER state_dict is saved (the decoder is discarded). Downstream
runs load it via `--pretrained-encoder <out>` / `cfg.pretrained_encoder`.

Note: the real encoder begins with ShiftAug (a small random pixel shift) which
stays active here, so this is effectively a shift-denoising AE — intended, and
matches the encoder's train-time computation graph.

Run on a GPU node inside the container:
    python pretrain/pretrain_encoder.py \
        --data /Metaworld/.../pretrain_data.h5 \
        --out  /Metaworld/.../encoder_ms19.pt \
        --model-size 19 --epochs 50
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import os
import secrets
import string
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# tdmpc2/ root on path so `from common import ...` resolves.
_HERE = Path(__file__).resolve().parent
_PARENT = _HERE.parent
sys.path.insert(0, str(_PARENT))

from omegaconf import OmegaConf
from common import MODEL_SIZE, layers
from pretrain.dataset import PretrainObsDataset
from pretrain.decoder import Decoder

CONFIG_PATH = _PARENT / "config.yaml"


def build_cfg(model_size, image_size, img_channels, state_dim):
    """Minimal cfg that reproduces the downstream encoder architecture.

    Applies MODEL_SIZE[model_size] exactly like common.parser.parse_cfg, sets
    a task-agnostic multimodal obs_shape, and task_dim=0 (metaworld always has
    task_dim=0; task identity is injected only at the dynamics MoE downstream).
    """
    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.model_size = int(model_size)
    for k, v in MODEL_SIZE[int(model_size)].items():
        cfg[k] = v
    cfg.multitask = False
    cfg.task_dim = 0
    cfg.obs = "rgb"
    cfg.obs_shape = {
        "state": [int(state_dim)],
        "rgb": [int(img_channels), int(image_size), int(image_size)],
    }
    return cfg


def encode(encoder, state, rgb):
    """Multimodal forward, identical to WorldModel.encode / PrismBackbone.encode."""
    s_feat = encoder["state"](state)
    r_feat = encoder["rgb"](rgb)
    return encoder["fusion"](torch.cat([s_feat, r_feat], dim=-1))


def _log_reconstructions(wandb, encoder, decoder, batch, device, step,
                         n, num_cameras):
    """Log an input-vs-reconstruction panel to wandb.

    For each of `n` fixed eval samples, builds a (2H, num_cameras*W, 3) RGB
    image: top row = the `num_cameras` input cameras side-by-side, bottom row =
    their reconstructions. The decoder is trained to reproduce the UNSHIFTED
    normalised image, so the (<=3px ShiftAug) input shown here aligns with the
    reconstruction. Recon images are decoder output mapped [-0.5,0.5] -> [0,1].
    """
    was_training = encoder.training
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        state = batch["state"][:n].to(device, non_blocking=True)
        rgb = batch["rgb"][:n].to(device, non_blocking=True)        # [0,255]
        z = encode(encoder, state, rgb)
        rgb_hat, _ = decoder(z)
    inp = (rgb / 255.0).clamp(0, 1).cpu()
    rec = (rgb_hat + 0.5).clamp(0, 1).cpu()
    c, h, w = inp.shape[1:]
    imgs = []
    for i in range(inp.shape[0]):
        cams_in = inp[i].view(num_cameras, 3, h, w)
        cams_re = rec[i].view(num_cameras, 3, h, w)
        row_in = torch.cat([cams_in[k] for k in range(num_cameras)], dim=-1)
        row_re = torch.cat([cams_re[k] for k in range(num_cameras)], dim=-1)
        panel = torch.cat([row_in, row_re], dim=-2)        # (3, 2h, num*w)
        panel = (panel.permute(1, 2, 0).numpy() * 255).astype("uint8")
        imgs.append(wandb.Image(panel, caption=f"#{i}  top=input  bottom=recon"))
    wandb.log({"recon": imgs}, step=step)
    if was_training:
        encoder.train(); decoder.train()


def _fmt_dur(seconds):
    """Compact h/m/s formatting for progress + ETA lines."""
    s = int(max(seconds, 0))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _new_run_id():
    return "".join(secrets.choice(string.ascii_lowercase + string.digits)
                   for _ in range(8))


# Set by SIGINT/SIGTERM so the training loop can stop *gracefully* (save +
# wandb.finish() => the run is marked "finished", not "crashed"/"killed").
_STOP_REQUESTED = False


def _install_stop_handlers():
    import signal

    def _handler(signum, _frame):
        global _STOP_REQUESTED
        if not _STOP_REQUESTED:
            print(f"\n>>> stop requested (signal {signum}); finishing the "
                  f"current step, then saving + closing wandb cleanly. "
                  f"Press again to force-quit.", flush=True)
            _STOP_REQUESTED = True
        else:
            raise KeyboardInterrupt   # second signal -> hard stop

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):  # e.g. not main thread
            pass


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    ds = PretrainObsDataset(args.data)
    print(f">>> dataset: {len(ds)} transitions | image_size={ds.image_size} "
          f"img_channels={ds.img_channels} | tasks={len(ds.tasks)}")

    n_val = int(len(ds) * args.val_frac)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")) \
        if n_val > 0 else None

    cfg = build_cfg(args.model_size, ds.image_size, ds.img_channels,
                    state_dim=7)
    encoder = layers.enc(cfg).to(device)
    decoder = Decoder(cfg).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    n_params = sum(p.numel() for p in params)
    print(f">>> encoder+decoder params: {n_params/1e6:.2f}M | device={device}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = args.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))

    # ---- Resume from a previous training checkpoint, if present ------------
    ckpt_path = Path(args.ckpt) if args.ckpt else Path(str(args.out) + ".ckpt")
    start_epoch, step, run_id, best_val = 0, 0, None, float("inf")
    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ck["encoder"])
        decoder.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch = int(ck["epoch"])     # next epoch to run
        step = int(ck["step"])
        run_id = ck.get("wandb_run_id")
        best_val = float(ck.get("best_val", float("inf")))
        print(f">>> RESUME from {ckpt_path}: epoch {start_epoch}/{args.epochs}, "
              f"step {step}/{total_steps}, run_id {run_id}")
        if start_epoch >= args.epochs:
            print(">>> Already at/past target epochs; nothing to do.")
            _save_encoder(encoder, cfg, ds, args)
            return
    if run_id is None:
        run_id = _new_run_id()

    # ---- wandb (resume='allow' reuses the persisted run_id) ----------------
    use_wandb = args.enable_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                       name=args.wandb_run_name or run_id,
                       id=run_id, resume="allow", config=dict(vars(args)))
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] disabled ({e})")
            use_wandb = False

    def run_batch(batch, train):
        state = batch["state"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)        # [0,255]
        rgb_tgt = rgb.div(255.0).sub(0.5)
        with torch.set_grad_enabled(train):
            z = encode(encoder, state, rgb)
            rgb_hat, state_hat = decoder(z)
            rgb_loss = F.mse_loss(rgb_hat, rgb_tgt)
            state_loss = F.mse_loss(state_hat, state)
            loss = rgb_loss + args.state_coef * state_loss
        return loss, rgb_loss.item(), state_loss.item()

    # Fixed eval batch for consistent reconstruction panels across the run.
    recon_batch = None
    if use_wandb and args.recon_num > 0:
        src = val_loader if val_loader is not None else train_loader
        recon_batch = next(iter(src))

    def save_ckpt(next_epoch):
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
            "opt": opt.state_dict(), "sched": sched.state_dict(),
            "epoch": int(next_epoch), "step": int(step),
            "wandb_run_id": run_id, "best_val": float(best_val),
        }
        tmp = str(ckpt_path) + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, ckpt_path)   # atomic — survives a mid-write kill

    # Override wandb's own SIGINT handling so Ctrl-C exits cleanly (finished).
    _install_stop_handlers()

    t0 = time.time()
    steps_this_run = 0
    epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        encoder.train(); decoder.train()
        for batch in train_loader:
            if _STOP_REQUESTED:
                break
            loss, rl, sl = run_batch(batch, train=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
            step += 1
            steps_this_run += 1
            if step % args.log_every == 0:
                rate = steps_this_run / max(time.time() - t0, 1e-9)  # steps/s
                eta = (total_steps - step) / max(rate, 1e-9)
                print(f"ep {epoch}/{args.epochs} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} (rgb {rl:.4f} state {sl:.4f}) "
                      f"lr {sched.get_last_lr()[0]:.2e} | {rate:.1f} it/s "
                      f"| elapsed {_fmt_dur(time.time()-t0)} "
                      f"| ETA {_fmt_dur(eta)}", flush=True)
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/rgb": rl,
                               "train/state": sl, "lr": sched.get_last_lr()[0],
                               "eta_sec": eta}, step=step)

            # Optional step-based reconstruction panels (0 = per-epoch only).
            if (use_wandb and recon_batch is not None and args.recon_every > 0
                    and step % args.recon_every == 0):
                import wandb
                _log_reconstructions(wandb, encoder, decoder, recon_batch,
                                     device, step, args.recon_num, ds.num_cameras)

        if _STOP_REQUESTED:
            print(f">>> stopping after step {step} (interrupted mid-epoch "
                  f"{epoch}).", flush=True)
            break

        if val_loader is not None:
            encoder.eval(); decoder.eval()
            vl = vr = vs = nb = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss, rl, sl = run_batch(batch, train=False)
                    vl += loss.item(); vr += rl; vs += sl; nb += 1
            nb = max(nb, 1)
            val_loss = vl / nb
            print(f">>> [val] ep {epoch} loss {val_loss:.4f} "
                  f"(rgb {vr/nb:.4f} state {vs/nb:.4f})", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"val/loss": val_loss, "val/rgb": vr/nb,
                           "val/state": vs/nb}, step=step)
                # Per-epoch reconstruction panel (always, when wandb is on).
                if recon_batch is not None:
                    _log_reconstructions(wandb, encoder, decoder, recon_batch,
                                         device, step, args.recon_num,
                                         ds.num_cameras)
            if val_loss < best_val:
                best_val = val_loss
                _save_encoder(encoder, cfg, ds, args, suffix=".best")

        # Persist full training state every epoch so a kill resumes cleanly,
        # and refresh the deliverable encoder.pt.
        save_ckpt(next_epoch=epoch + 1)
        _save_encoder(encoder, cfg, ds, args)

    # Final save. On a graceful stop we keep `epoch` (the interrupted epoch) as
    # the resume point so re-running continues from there; on normal completion
    # next_epoch = args.epochs marks it done.
    final_next_epoch = epoch if _STOP_REQUESTED else args.epochs
    save_ckpt(next_epoch=final_next_epoch)
    _save_encoder(encoder, cfg, ds, args)

    status = "interrupted (clean)" if _STOP_REQUESTED else "complete"
    print(f">>> Training {status} in {_fmt_dur(time.time()-t0)}. "
          f"Encoder saved to {args.out} (best-val copy: {args.out}.best).")

    # Close the wandb run cleanly so the website marks it FINISHED (not
    # crashed/killed) on both normal completion and a graceful Ctrl-C.
    if use_wandb:
        import wandb
        wandb.finish(exit_code=0)


def _save_encoder(encoder, cfg, ds, args, suffix=""):
    out = Path(str(args.out) + suffix)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_size": int(args.model_size),
        "image_size": int(ds.image_size),
        "img_channels": int(ds.img_channels),
        "num_cameras": int(ds.num_cameras),
        "cameras": ds.cameras,
        "latent_dim": int(cfg.latent_dim),
        "enc_dim": int(cfg.enc_dim),
        "num_channels": int(cfg.num_channels),
        "num_enc_layers": int(cfg.num_enc_layers),
        "pretrain_tasks": ds.tasks,
        "data": str(args.data),
    }
    torch.save({"encoder": encoder.state_dict(), "meta": meta}, out)


def build_argparser():
    p = argparse.ArgumentParser(description="Pretrain TD-MPC2 encoder (plain AE).")
    p.add_argument("--data", type=str, required=True, help="HDF5 from collect_data.py.")
    p.add_argument("--out", type=str, required=True, help="Output encoder .pt path.")
    p.add_argument("--model-size", type=int, default=19, choices=list(MODEL_SIZE.keys()),
                   help="Must match the downstream runs' model_size (default 19).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=20.0)
    p.add_argument("--state-coef", type=float, default=1.0,
                   help="Weight on the proprio reconstruction term.")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    # Reconstruction-image logging to wandb (qualitative eval)
    p.add_argument("--recon-every", type=int, default=0,
                   help="Log a reconstruction panel every N train steps. "
                        "0 (default) = only once per epoch at validation.")
    p.add_argument("--recon-num", type=int, default=10,
                   help="Number of example input/recon pairs per panel "
                        "(0 disables reconstruction logging entirely).")
    # Resume / checkpointing
    p.add_argument("--ckpt", type=str, default=None,
                   help="Training checkpoint path (default: <out>.ckpt). Holds "
                        "encoder+decoder+opt+sched+epoch+wandb_run_id; written "
                        "atomically every epoch.")
    p.add_argument("--resume", dest="resume", action="store_true",
                   help="Resume from <ckpt> if it exists (default).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore any existing checkpoint and start fresh.")
    p.set_defaults(resume=True)
    # wandb (resume='allow' reuses the persisted run id)
    p.add_argument("--enable-wandb", dest="enable_wandb", action="store_true")
    p.add_argument("--no-wandb", dest="enable_wandb", action="store_false")
    p.set_defaults(enable_wandb=True)
    p.add_argument("--wandb-project", type=str, default="prismatic_pretrain_encoder")
    p.add_argument("--wandb-entity", type=str, default="haoyu-a2i")
    p.add_argument("--wandb-run-name", type=str, default=None)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
