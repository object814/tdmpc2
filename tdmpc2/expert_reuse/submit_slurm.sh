#!/bin/bash
# Submit the expert-reuse analysis to SLURM.
#
# The router tier is pure checkpoint arithmetic and runs anywhere in seconds;
# only the rollout (causal) tier needs a GPU, because it re-runs MuJoCo under
# MPC. This script exists for that tier.
#
# Usage:
#   bash expert_reuse/submit_slurm.sh                       # all finished runs
#   bash expert_reuse/submit_slurm.sh --tasksets grasp reach
#   bash expert_reuse/submit_slurm.sh --dry-run
#   PARTITION=short GRES=gpu:a6000:1 bash expert_reuse/submit_slurm.sh
#
# Any other flags are forwarded verbatim to analyze.py.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TDMPC2_PKG="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$TDMPC2_PKG/.." && pwd)"

# Resource profile defaults to the one the training sweep is currently
# running under (sweep_prismatic_seq_progressive.yaml `slurm:` block), with a
# shorter wall clock. Override any of them from the environment.
PARTITION="${PARTITION:-long}"
ACCOUNT="${ACCOUNT:-engs-a2i}"
QOS="${QOS:-engs-a2i}"
GRES="${GRES:-gpu:l40s:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-48G}"
TIME="${TIME:-08:00:00}"

HOST_PATH="${HOST_PATH:-/data/engs-robot-learning/catz0908/Metaworld}"
CONTAINER_PATH="${CONTAINER_PATH:-/Metaworld}"
IMAGE="${IMAGE:-/data/engs-a2i/catz0908/Metaworld/.devcontainer/metaworld.sif}"

SLURM_DIR="$REPO_ROOT/sweep/slurm_run_dir_expert_reuse"
mkdir -p "$SLURM_DIR"

DRY_RUN=0
ARGS=()
for a in "$@"; do
    if [ "$a" = "--dry-run" ]; then DRY_RUN=1; else ARGS+=("$a"); fi
done

SCRIPT="$SLURM_DIR/expert_reuse.sbatch"
cat > "$SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=expert_reuse
#SBATCH --partition=$PARTITION
#SBATCH --account=$ACCOUNT
#SBATCH --qos=$QOS
#SBATCH --gres=$GRES
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$SLURM_DIR/%x_%j.out
#SBATCH --error=$SLURM_DIR/%x_%j.err

set -euo pipefail
echo ">>> node: \$(hostname)   job: \$SLURM_JOB_ID"
nvidia-smi || true

export WANDB_CACHE_DIR=$CONTAINER_PATH/.wandb_cache
export WANDB_DATA_DIR=$CONTAINER_PATH/.wandb_data
export WANDB_CONFIG_DIR=$CONTAINER_PATH/.wandb_config

apptainer exec --nv \\
    --bind $HOST_PATH:$CONTAINER_PATH \\
    "$IMAGE" \\
    bash -lc "
      set -euo pipefail
      mkdir -p \\\$WANDB_CACHE_DIR \\\$WANDB_DATA_DIR \\\$WANDB_CONFIG_DIR
      source $CONTAINER_PATH/conda/miniforge3/etc/profile.d/conda.sh
      conda activate $CONTAINER_PATH/conda/envs/tdmpc2
      cd $CONTAINER_PATH/third_party/tdmpc2/tdmpc2
      python -u expert_reuse/analyze.py --tier all ${ARGS[*]:-}
    "
EOF
chmod +x "$SCRIPT"

echo ">>> wrote $SCRIPT"
if [ "$DRY_RUN" = "1" ]; then
    echo '--- (dry run; not submitted) ---'
    cat "$SCRIPT"
    exit 0
fi
sbatch "$SCRIPT"
