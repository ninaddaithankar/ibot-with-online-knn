#!/bin/bash

#SBATCH --account=bdta-dtai-gh
#SBATCH --partition=ghx4
### NODE/CPU/MEM/GPU ###
#SBATCH --mem-bind=verbose,local
#SBATCH --mem-per-gpu=118G
#SBATCH --cpus-per-gpu=72

### ADDITIONAL RUN INFO ###
#SBATCH --array=0
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

### LOG INFO ###
#SBATCH --job-name=VIT_SMALL|ibot_ssv2
#SBATCH --output=logs/slurm/VIT_SMALL|ibot_ssv2/%A-%a.log
export RUN_NAME="VIT_SMALL|ibot_ssv2"

# ============================================================
#  iBOT pre-training on Something-Something v2
#  Submit: sbatch train_ssv2.sh
# ============================================================

export MASTER_PORT=$((20000 + (${SLURM_ARRAY_JOB_ID:-0} % 9999) + ${SLURM_ARRAY_TASK_ID:-0}))

# ---- Paths ----
SSV2_DIR="/work/hdd/bcsi/ndaithankar/datasets/ssv2"
SSV2_LABELS_DIR="labels/"
OUTPUT_DIR="./work_dirs/${RUN_NAME}"

# ---- W&B ----
WANDB_PROJECT="ibot-ssv2"
WANDB_RUN_NAME=""
WANDB_ENTITY=""

# ---- Hardware ----
NUM_GPUS=2
N_WORKERS=8

# ---- Model ----
ARCH="vit_small"
PATCH_SIZE=16

# ---- KNN evaluation ----
KNN_FREQ=1
NB_KNN="10 20"

# ---- Training ----
EPOCHS=100
BATCH_SIZE_PER_GPU=8
LR=0.0005
PRED_RATIO=0.3
PRED_SHAPE="block"
NUM_FRAMES=16
TIME_BETWEEN_FRAMES=0.25

# ============================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "logs/slurm/${RUN_NAME}"

module purge
ulimit -n 65535

CURDIR=$(cd "$(dirname "$0")"; pwd)

torchrun \
    --master_port ${MASTER_PORT} \
    --nproc_per_node=${NUM_GPUS} \
    "$CURDIR/main_ibot.py" \
    --dataset ssv2 \
    --ssv2_dir "$SSV2_DIR" \
    --ssv2_labels_dir "$SSV2_LABELS_DIR" \
    --ssv2_split train \
    --arch "$ARCH" \
    --patch_size "$PATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size_per_gpu "$BATCH_SIZE_PER_GPU" \
    --lr "$LR" \
    --pred_ratio "$PRED_RATIO" \
    --pred_shape "$PRED_SHAPE" \
    --ssv2_num_frames "$NUM_FRAMES" \
    --ssv2_time_between_frames "$TIME_BETWEEN_FRAMES" \
    --num_workers "$N_WORKERS" \
    --use_fp16 true \
    --saveckp_freq 10 \
    --knn_freq "$KNN_FREQ" \
    --nb_knn $NB_KNN \
    ${WANDB_PROJECT:+--wandb_project "$WANDB_PROJECT"} \
    ${WANDB_RUN_NAME:+--wandb_run_name "$WANDB_RUN_NAME"} \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}
