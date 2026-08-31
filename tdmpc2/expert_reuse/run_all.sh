#!/bin/bash
# Run the complete expert-reuse analysis on a GPU node.
#
# Intended to be run from INSIDE the apptainer container, on a node with a
# GPU. Every path is derived from this script's own location, so it does not
# care whether the repo is mounted at /Metaworld or anywhere else.
#
#   # inside the container, on an a6000 node:
#   bash /Metaworld/third_party/tdmpc2/tdmpc2/expert_reuse/run_all.sh
#
# One invocation covers both tiers, each on the runs it can support:
#   router tier  -- every run that reached its final task (finished or not)
#   causal tier  -- the finished runs only; the rollout needs the final task's
#                   task_modules.pt, which is only written when a task ends.
# Runs that reached the final task but have not finished are analysed by the
# router tier and reported as skipped by the causal tier.
#
# Options (env vars):
#   EPISODES=20            episodes collected once per run for the open-loop tier
#   CLOSEDLOOP_EPISODES=10 episodes per condition for closed-loop MPC (0 = skip,
#                          this is the expensive part)
#   TASKSETS=""            e.g. "grasp reach" to restrict
#   VARIANT=preEncFr       encoder variant; "all" for every run
#   MIN_FINAL_STEPS=200000 floor on steps into the final task
#   NO_WANDB=0             1 to skip wandb logging
#   ROUTER_ONLY=0          1 to run the CPU tier only
#
# Example, a quicker pass:
#   CLOSEDLOOP_EPISODES=5 EPISODES=10 bash .../run_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../tdmpc2/tdmpc2/expert_reuse
PKG_DIR="$(cd "$HERE/.." && pwd)"                      # .../tdmpc2/tdmpc2
TDMPC2_ROOT="$(cd "$PKG_DIR/.." && pwd)"               # .../third_party/tdmpc2
REPO_ROOT="$(cd "$TDMPC2_ROOT/../.." && pwd)"          # repo root (/Metaworld in-container)

EPISODES="${EPISODES:-20}"
CLOSEDLOOP_EPISODES="${CLOSEDLOOP_EPISODES:-10}"
TASKSETS="${TASKSETS:-}"
VARIANT="${VARIANT:-preEncFr}"
MIN_FINAL_STEPS="${MIN_FINAL_STEPS:-200000}"
NO_WANDB="${NO_WANDB:-0}"
ROUTER_ONLY="${ROUTER_ONLY:-0}"

echo "=============================================================="
echo ">>> expert-reuse analysis"
echo ">>> repo root : $REPO_ROOT"
echo ">>> running in: $PKG_DIR"
echo ">>> node      : $(hostname)"
echo "=============================================================="

# ---- conda env -------------------------------------------------------
# The training pipeline runs inside the container's tdmpc2 conda env (see
# sweep/run_array.sbatch). Activate it unless it is already active, since
# the container's default python has torch but not tensordict.
if ! python -c 'import tensordict' >/dev/null 2>&1; then
    CONDA_SH="$REPO_ROOT/conda/miniforge3/etc/profile.d/conda.sh"
    if [ -f "$CONDA_SH" ]; then
        echo ">>> activating $REPO_ROOT/conda/envs/tdmpc2"
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate "$REPO_ROOT/conda/envs/tdmpc2"
    else
        echo "!!! tensordict not importable and no conda.sh at $CONDA_SH" >&2
        echo "!!! activate the tdmpc2 env yourself, then re-run." >&2
        exit 1
    fi
fi
python -c 'import tensordict, torch; print(">>> torch", torch.__version__,
      "| cuda", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU"))'

# ---- runtime env -----------------------------------------------------
# osmesa matches what the training runs used; MuJoCo rendering is on CPU
# either way, the GPU is for the world model and MPC.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
export EGL_LOG_LEVEL=fatal
export LAZY_LEGACY_OP=0
export WANDB_CACHE_DIR="$REPO_ROOT/.wandb_cache"
export WANDB_DATA_DIR="$REPO_ROOT/.wandb_data"
export WANDB_CONFIG_DIR="$REPO_ROOT/.wandb_config"
mkdir -p "$WANDB_CACHE_DIR" "$WANDB_DATA_DIR" "$WANDB_CONFIG_DIR"

cd "$PKG_DIR"

ARGS=(--variant "$VARIANT" --min-final-steps "$MIN_FINAL_STEPS")
[ -n "$TASKSETS" ] && ARGS+=(--tasksets $TASKSETS)
[ "$NO_WANDB" = "1" ] && ARGS+=(--no-wandb)

if [ "$ROUTER_ONLY" = "1" ]; then
    TIER=router
else
    TIER=all
    ARGS+=(--episodes "$EPISODES"
           --closedloop-episodes "$CLOSEDLOOP_EPISODES")
fi

LOG_DIR="$HERE/results"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_all_$(date +%Y%m%d_%H%M%S).log"

echo ">>> tier: $TIER"
echo ">>> args: ${ARGS[*]}"
echo ">>> log : $LOG"
echo

# tee so the console shows progress and the full transcript is kept next to
# the figures.
python -u expert_reuse/analyze.py --tier "$TIER" "${ARGS[@]}" 2>&1 | tee "$LOG"

echo
echo "=============================================================="
echo ">>> done. results in $HERE/results/"
echo ">>>   <taskset>__<seed_dir>/*.png|pdf   figures"
echo ">>>   <taskset>__<seed_dir>/metrics.json"
echo ">>>   summary.json, reuse_summary.png"
echo ">>>   $(basename "$LOG")"
echo "=============================================================="
