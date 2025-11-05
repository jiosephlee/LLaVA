#!/bin/bash

set -euo pipefail

echo "➤ CPU TDC DEBUG START"


# --- Configuration ---
# Base language model (Intern)
MODEL_NAME="Qwen/Qwen3-1.7B"
# Projector weights from the pretrain step
PROJECTOR_BIN="checkpoints/qwen2b_pretrain_100/mm_projector.bin"
# Molecule encoder model (MolFormer)
MOLECULE_TOWER="ibm/MoLFormer-XL-both-10pct"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${2:-Tox}"

OUTPUT_DIR="checkpoints/llava_qwen2b_tdc_${TDC_TASK_GROUP}"

# Use projector weights if available
PRETRAIN_ARG=""
if [ -f "${PROJECTOR_BIN}" ]; then
  PRETRAIN_ARG="--pretrain_mm_mlp_adapter ${PROJECTOR_BIN}"
fi

python llava/train/train.py \
  --model_name_or_path "${MODEL_NAME}" \
  ${PRETRAIN_ARG} \
  --vision_tower "${MOLECULE_TOWER}" \
  --tf32 True \
  --version intern \
  --task_group_name "${TDC_TASK_GROUP}" \
  --mm_projector_type mlp2x_gelu \
  --freeze_backbone False \
  --tune_mm_mlp_adapter False \
  --output_dir "${OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "flash_attention_2" \
  --bf16 True \
  --num_train_epochs 40 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --learning_rate 8e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to none \
  --debug_mode False \
  --ensure_image_token_if_missing True \

echo "➤ TDC TRAINING DONE"

# --- Evaluation ---

echo "➤ STARTING TDC EVALUATION"
OUTPUT_DIR="checkpoints/llava_qwen2b_tdc_${TDC_TASK_GROUP}"

python llava/eval/eval_tdc.py \
  --model-path "${OUTPUT_DIR}" \
  --task-group-name "${TDC_TASK_GROUP}" \
  --output-dir "${OUTPUT_DIR}" \
  --conv-mode "intern" \
  --split "test"

echo "➤ TDC EVALUATION DONE"


echo "➤ CPU TDC DEBUG (TRAIN + EVAL) DONE"


