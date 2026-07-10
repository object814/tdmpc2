# TD-MPC2 / PRISM-WM — Run Guide

This repo currently supports five training methods. Each method has a paired
local launcher (`training/run_prismatic_*.sh`) and a sweep YAML
(`sweep/sweep_prismatic_*.yaml`) submitted via `sweep/launch_sequential.sh` or
`sweep/launch_single.sh`. Everything else has been moved under `archive/`.

## Layout

```
third_party/tdmpc2/
├── training/                        ← local launchers
│   ├── run_prismatic_single.sh
│   ├── run_prismatic_seq_static.sh
│   ├── run_prismatic_seq_static_er.sh
│   ├── run_prismatic_seq_progressive.sh
│   ├── run_prismatic_seq_progressive_er.sh
│   ├── run_pretrain_collect.sh      (collect offline pretrain dataset)
│   └── run_pretrain_encoder.sh      (train the encoder autoencoder)
├── sweep/                           ← sbatch dispatchers + sweep YAMLs
│   ├── launch_single.sh             (→ sweep_single_prism.yaml)
│   ├── launch_sequential.sh         (→ the 4 seq sweeps)
│   ├── launch_sweep.py, run_array.sbatch, job_to_env.py   (machinery)
│   ├── run_pretrain.sbatch          (one-shot collect+pretrain job)
│   ├── sweep_single_prism.yaml
│   ├── sweep_prismatic_seq_static.yaml
│   ├── sweep_prismatic_seq_static_er.yaml
│   ├── sweep_prismatic_seq_progressive.yaml
│   └── sweep_prismatic_seq_progressive_er.yaml
├── tdmpc2/                          ← Python source root
│   ├── train.py                     (single-task Hydra entry)
│   ├── config.yaml, tdmpc2.py
│   ├── common/, envs/, trainer/     (shared utilities)
│   ├── progmoe_training/            (architecture: backbone, masked_moe,
│   │                                 continual_tdmpc2)
│   ├── sequential/                  ← all 4 sequential entries live here
│   │   ├── sequential_train.py                 (driver + static entry)
│   │   ├── sequential_train_static_er.py
│   │   ├── sequential_train_progressive.py
│   │   └── sequential_train_progressive_er.py
│   └── pretrain/                    ← offline encoder pretraining
│       ├── collect_data.py          (scripted-policy rollouts -> HDF5)
│       ├── dataset.py, decoder.py
│       └── pretrain_encoder.py      (plain-AE encoder pretraining)
└── archive/                         ← everything dropped (recoverable)
```

## What each method is

| # | Method | Entry script | Summary |
|---|---|---|---|
| 1 | `prismatic_single` | `tdmpc2/train.py` | Single-task TD-MPC2 with the PRISM-WM MoE world-model block (K=3, Gram-Schmidt on by default). Hydra-driven. |
| 2 | `prismatic_seq_static` | `sequential/sequential_train.py` | Sequential CL with shared `PrismBackbone` + fresh `TaskModules` per task. MoE pool is **static full-K, all active, all trainable** at every task. No freezing, no ER. |
| 3 | `prismatic_seq_static_er` | `sequential/sequential_train_static_er.py` | Static-K (#2) + **cross-task ER**: Vitter reservoir of prev-task episodes injected into the new task's buffer before training starts. |
| 4 | `prismatic_seq_progressive` | `sequential/sequential_train_progressive.py` | **Progressive MoE**: `total_K = num_tasks · K` experts pre-allocated; at task t experts `[t·K, (t+1)·K)` are open, `[0..t·K)` frozen, future masked. Per-task heads fresh. No ER (uses the merged-batch sampling path in its degenerate, current-only form). |
| 5 | `prismatic_seq_progressive_er` | `sequential/sequential_train_progressive_er.py` | Progressive MoE (#4) + **DreamerV3-style merged-buffer ER**: each step samples a within-batch task-mix from {current buffer + anchored prev-task ER buffers}, single full-loss update with per-sample routing through each row's expert window. |

## How to launch each method

| # | Method | wandb project | Local launch (`training/`) | Sbatch launch (`sweep/`) |
|---|---|---|---|---|
| 1 | Single-task PRISM | `prismatic_single` | `bash training/run_prismatic_single.sh <taskset> [seed]` | `bash sweep/launch_single.sh` |
| 2 | Sequential static-K | `prismatic_seq_static` | `bash training/run_prismatic_seq_static.sh <taskset> [seed]` | `bash sweep/launch_sequential.sh static` |
| 3 | Sequential static-K + ER | `prismatic_seq_static_er` | `bash training/run_prismatic_seq_static_er.sh <taskset> [seed]` | `bash sweep/launch_sequential.sh static_er` |
| 4 | Sequential progressive (no ER) | `prismatic_seq_progressive` | `bash training/run_prismatic_seq_progressive.sh <taskset> [seed]` | `bash sweep/launch_sequential.sh progressive` |
| 5 | Sequential progressive + ER | `prismatic_seq_progressive_er` | `bash training/run_prismatic_seq_progressive_er.sh <taskset> [seed]` | `bash sweep/launch_sequential.sh progressive_er` |

Run `bash sweep/launch_sequential.sh` (no method arg) to submit all four sequential
sweeps back-to-back. Useful sweep flags forwarded to `launch_sweep.py`:

- `--dry-run` — expand the joblist + print the would-be sbatch command, don't submit.
- `--filter taskset=drawerpnp` — restrict the matrix.
- `--seeds 0,1,2` — override the seeds list from the YAML.
- `--rerun-failed` — only resubmit jobs without a `done.flag`.
- `--max-array N` — cap concurrent array elements.

## Available tasksets

**Sequential methods (#2–#5):**
`binpnp`, `drawerpnp`, `pnpblock`, `reach`, `grasp`, `pnpboxclose`.

**Single-task PRISM (#1):**
`binpnp`, `boxclose`, `draweropen`, `graspcube`, `pnpblock`, `pnpcube`,
`reach_xy`, `reach_xyz`, `compo_drawerpnp`, `compo_pnpblock`, `compo_pnpboxclose`.

Task lists per taskset live under `tasksets:` in each sweep YAML.

## Default recipe (applied everywhere)

| Knob | Default | Override (local) | Suffix when overridden |
|---|---|---|---|
| Gram-Schmidt orthogonal experts | **on** | `USE_ORTHOGONAL=false bash …` | `_noortho` |
| `K_per_task` (progressive) / `K` (single) | **3** | `K_PER_TASK=N bash …` (seq) / `NUM_EXPERTS=N bash …` (single) | `_k{N}` |
| Per-task τ snapshot at cross-task eval (progressive only) | **on** | `PER_TASK_TAU=false bash …` | `_notaupt` |
| ER buffer ratio (ER methods only) | **0.05** | `ER_BUFFER_RATIO=X bash …` | — |
| Reward / termination distillation anchor (#3 only) | **0** | `REWARD_ANCHOR_COEF=X bash …`, `TERM_ANCHOR_COEF=X bash …` | `_rwa{x}`, `_terma{x}` |

A default run gets a clean name like `drawerpnp_seed1` (wandb group `drawerpnp`).
Variant suffixes only appear when you deviate from a default.

## Wandb naming convention

- **Project**: identifies the method (one of the 5 names above).
- **Group**: taskset (so all seeds of one taskset group together).
- **Run name**: `{taskset}_seed{seed}` + any variant suffix.
- **Tags**: method, taskset, `seed{seed}`, date, sweep id, plus any variant tags.

## Encoder pretraining (optional)

You can pretrain the encoder offline (plain autoencoder over scripted-policy
rollouts of **non-eval** Meta-World tasks), then plug it into any of the 5
methods — fine-tuned or frozen — to study the effect of a pretrained
representation on learning speed and forgetting.

Code lives in [tdmpc2/pretrain/](tdmpc2/pretrain/):
- `collect_data.py` — rolls scripted experts on all non-eval tasks (47 tasks;
  every eval task + visual twin excluded) and dumps `(state, rgb)` observations
  to one HDF5 file, in the exact format the encoder consumes.
- `decoder.py` — mirror decoder used only during pretraining (discarded after).
- `pretrain_encoder.py` — trains `layers.enc(cfg)` + decoder with MSE
  reconstruction loss; saves **only** the encoder weights.

The pretrained encoder is architecture-identical to the one built in every
run (metaworld always has `cfg.task_dim=0`; task identity enters only at the
dynamics MoE, never the encoder), so it loads with no shape surgery — as long
as `model_size` matches (default **19**).

### Two-step workflow

```bash
# 1) Collect the offline dataset (GPU node, inside container; ~300k transitions)
bash training/run_pretrain_collect.sh                       # default out + 300k
#    or smoke test:  SMOKE=1 bash training/run_pretrain_collect.sh

# 2) Pretrain the encoder (GPU node, inside container)
bash training/run_pretrain_encoder.sh                       # model_size=19, 50 epochs
#    -> writes third_party/tdmpc2/pretrain_data/encoder_ms19.pt

# Or both as one sbatch job (collects if the HDF5 is missing, then trains):
sbatch --partition long --account engs-a2i --qos engs-a2i --gres gpu:l40s:1 \
       --cpus-per-task 16 --mem 80G --time 48:00:00 \
       third_party/tdmpc2/sweep/run_pretrain.sbatch
```

wandb is **on by default** for pretraining (project `prismatic_pretrain_encoder`,
`resume="allow"`). Besides scalar losses, it logs a **reconstruction panel** —
`RECON_NUM` examples (default 10), each showing the 3 input cameras (top row)
vs their reconstructions (bottom row). Default cadence is once per epoch; set
`RECON_EVERY=N` for a panel every N train steps instead. Disable wandb with
`ENABLE_WANDB=false`.

Both scripts print throughput + ETA as they run. Pretraining **auto-resumes**:
a full training checkpoint (`<out>.ckpt`: encoder+decoder+optimizer+scheduler+
epoch+wandb-run-id) is written atomically every epoch, so re-running the same
command (or re-submitting the sbatch) continues from the last completed epoch.
Force a fresh start with `RESUME=false`. Turn on wandb (with matching
resume) via `ENABLE_WANDB=true`; the run id is persisted in the checkpoint, so
a resumed job logs to the same wandb run. Each epoch also refreshes the
deliverable `encoder_ms19.pt`, and a best-val copy is kept at `<out>.best`.

### Three encoder regimes for any method

Set two env vars on any `run_prismatic_*.sh` launcher:

| Regime | Env vars | Run-name suffix |
|---|---|---|
| From scratch (default, current behaviour) | *(none)* | *(none)* |
| Pretrained + fine-tune | `PRETRAINED_ENCODER=/abs/path/encoder_ms19.pt` | `_preEnc` |
| Pretrained + frozen | `PRETRAINED_ENCODER=… FREEZE_ENCODER=true` | `_preEncFr` |

```bash
# examples
PRETRAINED_ENCODER=$PWD/third_party/tdmpc2/pretrain_data/encoder_ms19.pt \
    bash training/run_prismatic_seq_static.sh drawerpnp 1            # fine-tune

PRETRAINED_ENCODER=$PWD/third_party/tdmpc2/pretrain_data/encoder_ms19.pt \
FREEZE_ENCODER=true \
    bash training/run_prismatic_seq_progressive_er.sh drawerpnp 1    # frozen
```

The same `PRETRAINED_ENCODER` / `FREEZE_ENCODER` vars work on the single-task
launcher (they become Hydra overrides `pretrained_encoder=` / `freeze_encoder=`).
When frozen, the encoder's parameters are set `requires_grad=False` and are
excluded from the optimizer for the whole run.

> **Important:** the encoder is loaded with strict key matching. If you change
> `model_size`, `image_size`, `cameras`, `num_channels`, `enc_dim`,
> `latent_dim`, or `num_enc_layers` between pretraining and downstream use, the
> load fails loudly. Keep `model_size=19` end-to-end (the sweeps' default).

## Pre-existing wart (worth knowing)

`sweep_prismatic_seq_static.yaml`'s `--num-experts` isn't passed by the sbatch
path (`run_array.sbatch`) or by `launch_sweep.py`'s extra-args. The sbatch
path therefore uses the entry script's argparse default (8), not `K_per_task ×
num_tasks` (e.g. 9 for `drawerpnp` with `K=3`). The local bash launcher
[run_prismatic_seq_static.sh](training/run_prismatic_seq_static.sh) computes
this correctly. If/when this matters, add `--num-experts N` to the YAML's
`extra_args`.

## Reorganization notes

- `tdmpc2/sequential/` is new. The driver `sequential_train.py` and its three
  wrappers (`*_static_er`, `*_progressive`, `*_progressive_er`) all live here,
  replacing the old mix of `tdmpc2/sequential_train.py` (root) and `tdmpc2/er_training/`
  (now removed).
- `tdmpc2/progmoe_training/` is the **architecture** (backbone, masked MoE,
  continual TDMPC2). All sequential methods import from here regardless of
  whether they're static or progressive.
- Everything else dropped from the active set sits in
  [archive/](archive/) (sweep yamls, training launchers, entry scripts, debug
  files, evaluation scripts). If `evaluate.py` / `evaluate_prismatic.py` matter
  for offline-checkpoint evaluation, pull them back out.
