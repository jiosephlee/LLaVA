#!/bin/bash

set -euo pipefail

echo "➤ CPU DEBUG START"

# Force CPU
# export CUDA_VISIBLE_DEVICES=""

# --- Configuration ---
# Base language model (Intern-S1)
MODEL_NAME="Qwen/Qwen3-0.6B"

# Molecule encoder model (MolFormer)
MOLECULE_TOWER="ibm/MoLFormer-XL-both-10pct"

# Prepared alignment dataset
DATA_PATH="playground/data/llava_medex_alignment_10k.json"

# Output directory
OUTPUT_DIR="checkpoints/debug_cpu_intern_pretrain"

python llava/train/train.py \
    --model_name_or_path "$MODEL_NAME" \
    --version intern \
    --data_path "$DATA_PATH" \
    --vision_tower "$MOLECULE_TOWER" \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature "last_hidden_state" \
    --ensure_image_token_if_missing True \
    --group_by_modality_length False \
    --output_dir "$OUTPUT_DIR" \
    --optim "paged_adamw_8bit" \
    --attn_implementation "sdpa" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "linear" \
    --model_max_length 1024 \
    --gradient_checkpointing False \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --logging_steps 1 \
    --report_to wandb \
    --debug_mode False \
    --max_steps 100 \

echo "➤ CPU DEBUG DONE"


