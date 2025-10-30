#!/bin/bash

# This script is for the alignment pre-training of a Chem-LLaVA model.
# It trains the multimodal projector to connect MolFormer to the LLM.
# IMPORTANT: This script should be run from the root of the LLaVA project directory.

# --- Configuration ---
# Set the base language model (Intern-S1)
MODEL_NAME="internlm/Intern-S1-mini"

# Set the molecule encoder model (MolFormer)
MOLECULE_TOWER="ibm/MoLFormer-XL-both-10pct"

# Set the path to your prepared alignment dataset
DATA_PATH="./playground/data/llava_medex_alignment.json"

# Set the output directory for the pre-trained projector and model weights
OUTPUT_DIR="./checkpoints/llava-$MODEL_NAME-molformer-pretrain"

# --- Training Command ---
# deepspeed llava/train/train.py \
#    --deepspeed ./scripts/zero3.json \
python llava/train/train.py \
    --model_name_or_path $MODEL_NAME \
    --version v1 \
    --data_path $DATA_PATH \
    --vision_tower $MOLECULE_TOWER \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --optim "paged_adamw_8bit" \
    --attn_implementation "sdpa" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
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
