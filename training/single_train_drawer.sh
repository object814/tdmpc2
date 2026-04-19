#!/bin/bash

# Single-task TD-MPC2 training on the DreamerV3-aligned Metaworld env.
# Mirrors third_party/dreamerv3/training/single_train_drawer.sh but uses
# TD-MPC2's Hydra CLI.

# -------- Configuration (tune here) --------
WANDB_ENTITY="haoyu-a2i"
WANDB_PROJECT="Metaworld_Tdmpc2_Single"
RUN_NAME="mw_tdmpc2_single_drawer_$(date +%m%d)"
LOGDIR="../logdir/single_tdmpc2/${RUN_NAME}"

# Metaworld task (mw-<env-name> -v3)
TASK="mw-drawer-open-v3"

# Training budget (env steps)
STEPS=1000

# Seed
SEED=1

# Metaworld env knobs (align with DreamerV3)
IMAGE_SIZE=128
MAX_EPISODE_STEPS=250
ACTION_REPEAT=2
CAMERAS="[topview,front,gripperPOV]"

# TD-MPC2 knobs
BATCH_SIZE=256
BUFFER_SIZE=50000   # matches dreamer dataset-size = steps/4
HORIZON=3
MPC=true
MODEL_SIZE=5
EVAL_FREQ=10000
EVAL_EPISODES=5

# -------- Launch --------
echo "=================================================="
echo ">>> Single Task TD-MPC2 Training"
echo ">>> Task: ${TASK}"
echo ">>> Steps: ${STEPS}"
echo ">>> Logdir: ${LOGDIR}"
echo ">>> Image: ${IMAGE_SIZE}x${IMAGE_SIZE}, cameras=${CAMERAS}"
echo ">>> ep_len=${MAX_EPISODE_STEPS}, action_repeat=${ACTION_REPEAT}"
echo "=================================================="

mkdir -p "${LOGDIR}"

cd "$(dirname "$0")/../tdmpc2"

python train.py \
    task=${TASK} \
    seed=${SEED} \
    steps=${STEPS} \
    model_size=${MODEL_SIZE} \
    batch_size=${BATCH_SIZE} \
    buffer_size=${BUFFER_SIZE} \
    horizon=${HORIZON} \
    mpc=${MPC} \
    eval_freq=${EVAL_FREQ} \
    eval_episodes=${EVAL_EPISODES} \
    image_size=${IMAGE_SIZE} \
    max_episode_steps=${MAX_EPISODE_STEPS} \
    action_repeat=${ACTION_REPEAT} \
    cameras=${CAMERAS} \
    exp_name=${RUN_NAME} \
    wandb_entity=${WANDB_ENTITY} \
    wandb_project=${WANDB_PROJECT} \
    enable_wandb=true \
    hydra.run.dir=${LOGDIR}
