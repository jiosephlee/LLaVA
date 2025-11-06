#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=llava_100k_two_stage
#SBATCH --output=logs/llava_100k_two_stage.out
#SBATCH --error=logs/llava_100k_two_stage.err

# --- Robust Path Setup ---
# Use the SLURM_SUBMIT_DIR variable to get the directory where the sbatch command was run.
# IMPORTANT: This script must be submitted with `sbatch` from the root of the LLaVA project.

echo "➤ START"
echo "➤ SETTING UP HOST CUDA"
module unload cuda
module load cuda/12.4


# Define the path to your SIF file
YOUR_SIF_FILE="/gpfs/fs001/cbica/home/leejose/joseph/pytorch-2.4.0-cuda12.4-cudnn9-devel.sif"

echo "➤ RUNNING SCRIPT INSIDE APPTAINER: ${YOUR_SIF_FILE}"

# --- Configuration ---
# Base language model (Intern-S1)
MODEL_NAME="jiosephlee/Intern-S1-mini-lm"

# Molecule encoder model (MolFormer)
MOLECULE_TOWER="ibm/MoLFormer-XL-both-10pct"

# Projector weights from the pretrain step
PROJECTOR_BIN="checkpoints/pretrain_100k/mm_projector.bin"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${2:-All}"

# Stage 1: Freeze backbone, tune MLP adapter
STAGE1_OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_100k_${TDC_TASK_GROUP}_stage1_1ep_frozen"

# Stage 2: Full fine-tuning
STAGE2_OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_100k_${TDC_TASK_GROUP}_stage2_2ep_bs64_5e4"

# Use projector weights if available
PRETRAIN_ARG=""
if [ -f "${PROJECTOR_BIN}" ]; then
  PRETRAIN_ARG="--pretrain_mm_mlp_adapter ${PROJECTOR_BIN}"
fi

# --- Stage 1: Freeze backbone, tune MLP adapter for 1 epoch ---
echo "➤ STAGE 1: TRAINING WITH FROZEN BACKBONE (1 EPOCH)"

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${MODEL_NAME}" \
  ${PRETRAIN_ARG} \
  --vision_tower "${MOLECULE_TOWER}" \
  --version intern \
  --task_group_name "${TDC_TASK_GROUP}" \
  --mm_projector_type mlp2x_gelu \
  --freeze_backbone True \
  --tune_mm_mlp_adapter True \
  --output_dir "${STAGE1_OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "flash_attention_2" \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --logging_steps 1 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "no" \
  --save_strategy "epoch" \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --debug_mode False \
  --ensure_image_token_if_missing True

echo "➤ STAGE 1 DONE"

# --- Stage 2: Full fine-tuning (same as original script) ---
echo "➤ STAGE 2: FULL FINE-TUNING (5 EPOCHS)"

# Load checkpoint from stage 1 (check for checkpoint subdirectories first, then output dir)
STAGE1_CHECKPOINT=$(ls -td ${STAGE1_OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -n 1)

if [ -n "${STAGE1_CHECKPOINT}" ]; then
  echo "➤ Loading checkpoint from Stage 1: ${STAGE1_CHECKPOINT}"
  STAGE2_MODEL_PATH="${STAGE1_CHECKPOINT}"
elif [ -f "${STAGE1_OUTPUT_DIR}/pytorch_model.bin" ] || [ -f "${STAGE1_OUTPUT_DIR}/adapter_config.json" ]; then
  echo "➤ Using Stage 1 output directory as checkpoint"
  STAGE2_MODEL_PATH="${STAGE1_OUTPUT_DIR}"
else
  echo "⚠️  WARNING: Stage 1 checkpoint not found, using original model"
  STAGE2_MODEL_PATH="${MODEL_NAME}"
fi

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${STAGE2_MODEL_PATH}" \
  --vision_tower "${MOLECULE_TOWER}" \
  --version intern \
  --task_group_name "${TDC_TASK_GROUP}" \
  --mm_projector_type mlp2x_gelu \
  --freeze_backbone False \
  --tune_mm_mlp_adapter False \
  --output_dir "${STAGE2_OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "flash_attention_2" \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --logging_steps 1 \
  --gradient_accumulation_steps 16 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --learning_rate 8e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --debug_mode False \
  --ensure_image_token_if_missing True

echo "➤ STAGE 2 TRAINING DONE"

# --- Evaluation ---
echo "➤ STARTING TDC EVALUATION"

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/eval/eval_tdc.py \
  --model-path "${STAGE2_OUTPUT_DIR}" \
  --task-group-name "${TDC_TASK_GROUP}" \
  --output-dir "${STAGE2_OUTPUT_DIR}" \
  --conv-mode "intern" \
  --split "test"

echo "➤ TDC EVALUATION DONE"

echo "➤ TWO-STAGE FINE-TUNING COMPLETE"

