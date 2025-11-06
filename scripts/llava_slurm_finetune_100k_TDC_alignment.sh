#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=tdc_projector_then_full
#SBATCH --output=logs/llava_750k_tdc_projector_then_full.out
#SBATCH --error=logs/llava_750k_tdc_projector_then_full.err

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

PROJECTOR_BIN="checkpoints/pretrain_750k_bs64_1e3/mm_projector.bin"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${1:-All}"

# Set the output directory for Stage 1 (TDC projector training)
STAGE1_OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_750k_${TDC_TASK_GROUP}_100k_alignment"

# --- Hyperparameters for Fine-tuning Stage ---
# These can be overridden via command line arguments
FINETUNE_EPOCHS="${2:-3}"
FINETUNE_BATCH_SIZE="${3:-4}"
FINETUNE_GRAD_ACCUM="${4:-16}"
FINETUNE_LR="${5:-5e-4}"

# Calculate effective batch size for filename
FINETUNE_EFF_BS=$((FINETUNE_BATCH_SIZE * FINETUNE_GRAD_ACCUM))

# Transform LR for filename (e.g., 8e-5 -> 8e5, 2e-4 -> 2e4)
# Remove the negative sign and decimal point for cleaner filename
FINETUNE_LR_FILENAME=$(echo "${FINETUNE_LR}" | sed 's/e-/e/g' | sed 's/\.//g')

# Set the output directory for Stage 2 (TDC fine-tuning)
STAGE2_OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_750k_projector_then_100k_alignment_${TDC_TASK_GROUP}_${FINETUNE_EPOCHS}ep_bs${FINETUNE_EFF_BS}_${FINETUNE_LR_FILENAME}"

# Display hyperparameters
echo "➤ HYPERPARAMETERS:"
echo "   TDC Task Group: ${TDC_TASK_GROUP}"
echo "   Stage 2 Epochs: ${FINETUNE_EPOCHS}"
echo "   Stage 2 Batch Size: ${FINETUNE_BATCH_SIZE}"
echo "   Stage 2 Gradient Accumulation Steps: ${FINETUNE_GRAD_ACCUM}"
echo "   Stage 2 Effective Batch Size: ${FINETUNE_EFF_BS}"
echo "   Stage 2 Learning Rate: ${FINETUNE_LR}"
echo "   Stage 1 Output Directory: ${STAGE1_OUTPUT_DIR}"
echo "   Stage 2 Output Directory: ${STAGE2_OUTPUT_DIR}"
echo ""

# --- Stage 1: TDC Projector-Only Training ---
echo "➤ STAGE 1: TDC PROJECTOR-ONLY TRAINING (BACKBONE FROZEN)"

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${MODEL_NAME}" \
  --pretrain_mm_mlp_adapter "${PROJECTOR_BIN}" \
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
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 1 \
  --logging_steps 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "epoch" \
  --save_total_limit 1 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --debug_mode False \
  --ensure_image_token_if_missing True \
  --max_steps 1562 \

echo "➤ STAGE 1 DONE"

# --- Stage 2: TDC Full Fine-tuning ---
echo "➤ STAGE 2: TDC FULL FINE-TUNING (BACKBONE + PROJECTOR)"

# Load checkpoint from Stage 1 (check for checkpoint subdirectories first, then output dir)
STAGE1_CHECKPOINT=$(ls -td ${STAGE1_OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -n 1)

if [ -n "${STAGE1_CHECKPOINT}" ]; then
  echo "➤ Loading checkpoint from Stage 1: ${STAGE1_CHECKPOINT}"
  STAGE2_MODEL_PATH="${MODEL_NAME}"
  STAGE2_PROJECTOR_PATH="${STAGE1_CHECKPOINT}/mm_projector.bin"
elif [ -f "${STAGE1_OUTPUT_DIR}/mm_projector.bin" ]; then
  echo "➤ Using Stage 1 output directory projector: ${STAGE1_OUTPUT_DIR}/mm_projector.bin"
  STAGE2_MODEL_PATH="${MODEL_NAME}"
  STAGE2_PROJECTOR_PATH="${STAGE1_OUTPUT_DIR}/mm_projector.bin"
else
  echo "⚠️  WARNING: Stage 1 checkpoint not found, using original projector"
  STAGE2_MODEL_PATH="${MODEL_NAME}"
  STAGE2_PROJECTOR_PATH="${PROJECTOR_BIN}"
fi

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${STAGE2_MODEL_PATH}" \
  --pretrain_mm_mlp_adapter "${STAGE2_PROJECTOR_PATH}" \
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
  --num_train_epochs ${FINETUNE_EPOCHS} \
  --per_device_train_batch_size ${FINETUNE_BATCH_SIZE} \
  --per_device_eval_batch_size 1 \
  --logging_steps 1 \
  --gradient_accumulation_steps ${FINETUNE_GRAD_ACCUM} \
  --eval_strategy "no" \
  --save_strategy "no" \
  --learning_rate ${FINETUNE_LR} \
  --weight_decay 0.0 \
  --warmup_ratio 0.05 \
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

echo "➤ ALL STAGES COMPLETE"

