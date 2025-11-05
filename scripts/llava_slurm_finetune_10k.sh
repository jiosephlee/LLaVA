#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB
#SBATCH --job-name=llava_10k_all
#SBATCH --output=logs/llava_10k_all.out
#SBATCH --error=logs/llava_10k_all.err

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
PROJECTOR_BIN="checkpoints/pretrain_10k/mm_projector_10k.bin"
# Molecule encoder model (MolFormer)
MOLECULE_TOWER="ibm/MoLFormer-XL-both-10pct"

# TDC task group (e.g., Tox, ADMET_group, Skin_Reaction)
TDC_TASK_GROUP="${2:-All}"

OUTPUT_DIR="checkpoints/llava_interns1mini_tdc_all_10k_${TDC_TASK_GROUP}_5ep_bs64_5e4"

# Use projector weights if available
PRETRAIN_ARG=""
if [ -f "${PROJECTOR_BIN}" ]; then
  PRETRAIN_ARG="--pretrain_mm_mlp_adapter ${PROJECTOR_BIN}"
fi

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
  --freeze_backbone False \
  --tune_mm_mlp_adapter False \
  --output_dir "${OUTPUT_DIR}" \
  --optim "paged_adamw_8bit" \
  --attn_implementation "flash_attention_2" \
  --bf16 True \
  --num_train_epochs 5 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --eval_strategy "no" \
  --logging_steps 1 \
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
  --ensure_image_token_if_missing True \

echo "➤ TDC TRAINING DONE"

# --- Evaluation ---
echo "➤ STARTING TDC EVALUATION"

apptainer exec --cleanenv --nv \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    ${YOUR_SIF_FILE} \
  python llava/eval/eval_tdc.py \
  --model-path "${OUTPUT_DIR}" \
  --task-group-name "${TDC_TASK_GROUP}" \
  --output-dir "${OUTPUT_DIR}" \
  --conv-mode "intern" \
  --split "test"

echo "➤ TDC EVALUATION DONE"


echo "➤ CPU TDC DEBUG (TRAIN + EVAL) DONE"


