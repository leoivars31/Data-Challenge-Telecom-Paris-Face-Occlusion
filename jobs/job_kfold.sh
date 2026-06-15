#!/bin/bash
# ===========================================================================
# K-Fold training (5 folds) — main script for the final solution.
#
# Usage (final solution):
#   BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k" sbatch jobs/job_kfold.sh
#   BACKBONE="eva02_base_patch14_224.mim_in22k"     sbatch jobs/job_kfold.sh
#
# Optional variables:
#   BACKBONE       timm backbone (default: convnext_tiny.fb_in22k_ft_in1k)
#   MALE_FACTOR    male sample weight multiplier (default: 1.0 — optimal in k-fold)
#   RANDOM_ERASING 1 for --no_occlusion_safe (default: 0 — light regime)
#   BATCH_SIZE     batch size (default: 64)
#   GRAD_ACCUM     gradient accumulation (default: 2, effective batch = 128)
# ===========================================================================
#SBATCH --job-name=face-kfold
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --array=0-4

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

FOLD=$SLURM_ARRAY_TASK_ID
MALE_FACTOR=${MALE_FACTOR:-1.0}
BACKBONE=${BACKBONE:-"convnext_tiny.fb_in22k_ft_in1k"}
BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-2}
RANDOM_ERASING=${RANDOM_ERASING:-0}  # 1 = --no_occlusion_safe
RUN_DIR="data/submissions/checkpoints/kfold_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "K-Fold training — Fold $FOLD/5"
echo "Job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "male_factor=$MALE_FACTOR | backbone=$BACKBONE | random_erasing=$RANDOM_ERASING"
echo "Checkpoint dir: $RUN_DIR"
echo "=========================================="
nvidia-smi

python -m src.train \
    --data_root data/raw \
    --checkpoint_dir "$RUN_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum_steps "$GRAD_ACCUM" \
    --num_epochs 30 \
    --freeze_epochs 2 \
    --lr_head 3e-4 \
    --lr_backbone 3e-5 \
    --warmup_epochs 2 \
    --patience 30 \
    --loss wmse \
    --male_factor "$MALE_FACTOR" \
    $([ "$RANDOM_ERASING" = "1" ] && echo "--no_occlusion_safe") \
    --fold "$FOLD" \
    --n_splits 5 \
    --backbone "$BACKBONE"

echo "=========================================="
echo "Fold $FOLD done at $(date)"
echo "=========================================="
