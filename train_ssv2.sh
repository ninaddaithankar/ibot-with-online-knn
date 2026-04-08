#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  iBOT pre-training on Something-Something v2
#  Usage: bash train_ssv2.sh
# ============================================================

# ---- Paths ----
SSV2_DIR="/shared/nas2/ninadd2/datasets/ssv2"          # root dir containing 20bn-something-something-v2/ and labels/
SSV2_LABELS_DIR="labels/"          # relative to SSV2_DIR, or set an absolute path
OUTPUT_DIR="./work_dirs/ibot_ssv2"
# ---- W&B ----
WANDB_PROJECT="ibot-ssv2"
WANDB_RUN_NAME=""        # leave empty to let W&B auto-generate
WANDB_ENTITY=""          # leave empty to use your default entity

# ---- Hardware ----
N_GPUS=2
N_WORKERS=8   # data loader workers per GPU

# ---- Model ----
ARCH="vit_base"
PATCH_SIZE=16

# ---- KNN evaluation ----
KNN_FREQ=1           # run KNN every N epochs; set to 0 to disable
NB_KNN="10 20"       # k values to evaluate

# ---- Training ----
EPOCHS=100
BATCH_SIZE_PER_GPU=8   # effective batch = N_GPUS * BATCH_SIZE_PER_GPU
LR=0.0005
PRED_RATIO=0.3
PRED_SHAPE="block"      # block or rand
NUM_FRAMES=16
TIME_BETWEEN_FRAMES=0.25

# ============================================================

CURDIR=$(cd "$(dirname "$0")"; pwd)
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo " iBOT SSv2 pre-training"
echo "  GPUs            : $N_GPUS"
echo "  Arch            : $ARCH (patch $PATCH_SIZE)"
echo "  Epochs          : $EPOCHS"
echo "  Batch/GPU       : $BATCH_SIZE_PER_GPU  (total: $((N_GPUS * BATCH_SIZE_PER_GPU)))"
echo "  Frames/video    : $NUM_FRAMES  (dt: ${TIME_BETWEEN_FRAMES}s)"
echo "  SSv2 dir        : $SSV2_DIR"
echo "  Output dir      : $OUTPUT_DIR"
echo "  W&B project     : ${WANDB_PROJECT:-disabled}"
echo "  KNN eval freq   : ${KNN_FREQ} epochs  (k = $NB_KNN)"
echo "========================================"

CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node="$N_GPUS" \
    --master_port=29500 \
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
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
    "$@"   # pass any extra args from the command line
