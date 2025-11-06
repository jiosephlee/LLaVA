#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=medex_alignment
#SBATCH --output=logs/llava_pretrained_750k_medex_mid_100k_alignment.out
#SBATCH --error=logs/llava_pretrained_750k_medex_mid_100k_alignment.err

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

# Set the path to your prepared alignment dataset
DATA_PATH="playground/data/llava_medex_fact_alignment_100k.json"

PROJECTOR_BIN="checkpoints/pretrain_750k_bs64_1e3/mm_projector.bin"

# Set the output directory for the alignment stage (with unfrozen LM)
ALIGNMENT_OUTPUT_DIR="checkpoints/pretrain_750k_mid_stage_alignment_100k_medex"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${1:-All}"

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

# Set the output directory for TDC fine-tuning
FINETUNE_OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_100k_mid_stage_alignment_100k_medex_${TDC_TASK_GROUP}_${FINETUNE_EPOCHS}ep_bs${FINETUNE_EFF_BS}_${FINETUNE_LR_FILENAME}"

# Display hyperparameters
echo "➤ HYPERPARAMETERS:"
echo "   TDC Task Group: ${TDC_TASK_GROUP}"
echo "   Fine-tuning Epochs: ${FINETUNE_EPOCHS}"
echo "   Batch Size: ${FINETUNE_BATCH_SIZE}"
echo "   Gradient Accumulation Steps: ${FINETUNE_GRAD_ACCUM}"
echo "   Effective Batch Size: ${FINETUNE_EFF_BS}"
echo "   Learning Rate: ${FINETUNE_LR}"
echo "   Output Directory: ${FINETUNE_OUTPUT_DIR}"
echo ""

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${MODEL_NAME}" \
  --pretrain_mm_mlp_adapter "${PROJECTOR_BIN}" \
  --version intern \
  --data_path "${DATA_PATH}" \
  --vision_tower "${MOLECULE_TOWER}" \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --mm_vision_select_feature "last_hidden_state" \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir "${ALIGNMENT_OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "flash_attention_2" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_strategy "no" \
  --save_strategy "no" \
  --save_total_limit 1 \
  --learning_rate 1e-3 \
  --weight_decay 0.0 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --debug_mode False

echo "➤ ALIGNMENT STAGE DONE"

# --- Stage 2: TDC Fine-tuning ---
echo "➤ STARTING TDC FINE-TUNING STAGE"

# Load checkpoint from alignment stage (check for checkpoint subdirectories first, then output dir)
ALIGNMENT_CHECKPOINT=$(ls -td ${ALIGNMENT_OUTPUT_DIR}/checkpoint-* 2>/dev/null | head -n 1)

if [ -n "${ALIGNMENT_CHECKPOINT}" ]; then
  echo "➤ Loading checkpoint from Alignment Stage: ${ALIGNMENT_CHECKPOINT}"
  FINETUNE_MODEL_PATH="${ALIGNMENT_CHECKPOINT}"
  FINETUNE_PROJECTOR_PATH="${ALIGNMENT_CHECKPOINT}/mm_projector.bin"
else
  echo "⚠️  WARNING: Alignment checkpoint not found, using original model"
  FINETUNE_MODEL_PATH="${MODEL_NAME}"
  FINETUNE_PROJECTOR_PATH="checkpoints/pretrain_750k/mm_projector.bin"
fi

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/train/train.py \
  --model_name_or_path "${FINETUNE_MODEL_PATH}" \
  --pretrain_mm_mlp_adapter "${FINETUNE_PROJECTOR_PATH}" \
  --vision_tower "${MOLECULE_TOWER}" \
  --version intern \
  --task_group_name "${TDC_TASK_GROUP}" \
  --mm_projector_type mlp2x_gelu \
  --freeze_backbone False \
  --tune_mm_mlp_adapter False \
  --output_dir "${FINETUNE_OUTPUT_DIR}" \
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

echo "➤ TDC TRAINING DONE"

# --- Evaluation ---
echo "➤ STARTING TDC EVALUATION"

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/eval/eval_tdc.py \
  --model-path "${FINETUNE_OUTPUT_DIR}" \
  --task-group-name "${TDC_TASK_GROUP}" \
  --output-dir "${FINETUNE_OUTPUT_DIR}" \
  --conv-mode "intern" \
  --split "test"

echo "➤ TDC EVALUATION DONE"

echo "➤ ALL STAGES COMPLETE"

