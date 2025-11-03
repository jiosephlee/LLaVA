#!/bin/bash

set -euo pipefail

echo "➤ CPU TDC DEBUG START"


# --- Configuration ---
# Path to a pre-trained (aligned) Chem-LLaVA checkpoint to start from
# Edit this to your actual path, or pass as first arg to this script.
MODEL_PATH="Qwen/Qwen2.5-1.5B"
PROJECTOR_PATH="checkpoints/debug_cpu_intern_pretrain"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${2:-Tox}"

OUTPUT_DIR="checkpoints/debug_cpu_tdc_${TDC_TASK_GROUP}"

# Use projector weights if available
PRETRAIN_ARG=""
if [ -f "${PROJECTOR_PATH}/mm_projector.bin" ]; then
  PRETRAIN_ARG="--pretrain_mm_mlp_adapter ${PROJECTOR_PATH}/mm_projector.bin"
fi

python llava/train/train.py \
  --model_name_or_path "${MODEL_PATH}" \
  ${PRETRAIN_ARG} \
  --version intern \
  --task_group_name "${TDC_TASK_GROUP}" \
  --mm_projector_type mlp2x_gelu \
  --freeze_backbone False \
  --tune_mm_mlp_adapter True \
  --output_dir "${OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "sdpa" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "linear" \
  --model_max_length 1024 \
  --gradient_checkpointing False \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to none \
  --debug_mode True \
  --max_steps 20

echo "➤ CPU TDC DEBUG DONE"


