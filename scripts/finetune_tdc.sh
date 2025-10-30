#!/bin/bash

# This script is for fine-tuning a Chem-LLaVA model on TDC tasks.

# --- Robust Path Setup ---
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
LLAVA_ROOT=$(dirname "$SCRIPT_DIR")

# --- Configuration ---
# Set the path to your fully trained Chem-LLaVA model checkpoint
# This should be the model that has already been pre-trained/aligned.
MODEL_PATH="$LLAVA_ROOT/checkpoints/llava-internlm/Intern-S1-mini-molformer-pretrain"

# Set the TDC task group you want to fine-tune on
# e.g., "Tox", "ADMET_group", "Skin_Reaction"
TDC_TASK_GROUP="Tox"

# Set the output directory for this TDC fine-tuning run
OUTPUT_DIR="$LLAVA_ROOT/checkpoints/llava-internlm/Intern-S1-mini-tdc-$TDC_TASK_GROUP-finetune"

# --- Training Command ---
# deepspeed "$LLAVA_ROOT/llava/train/train.py" \
#    --deepspeed "$LLAVA_ROOT/scripts/zero3.json" \
python "$LLAVA_ROOT/llava/train/train.py" \
    --model_name_or_path $MODEL_PATH \
    --version v1 \
    --task_group_name $TDC_TASK_GROUP \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --bits 8 \
    --optim "paged_adamw_8bit" \
    --attn_implementation "sdpa" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
