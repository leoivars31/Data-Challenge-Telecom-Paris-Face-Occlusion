#!/bin/bash
# ===========================================================================
# Multi-backbone training (no K-fold) — trains 3 backbones in parallel.
# ===========================================================================
#SBATCH --job-name=face-ensemble-bb
#SBATCH --output=logs/%x_%j_%a.out
#SBATCH --error=logs/%x_%j_%a.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=36:00:00
#SBATCH --array=0-2

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

# Map array index to backbone
BACKBONES=(
    "convnext_tiny.fb_in22k_ft_in1k"
    "convnextv2_base.fcmae_ft_in22k_in1k"
    "eva02_base_patch14_224.mim_in22k"
)
BACKBONE=${BACKBONES[$SLURM_ARRAY_TASK_ID]}
MALE_FACTOR=${MALE_FACTOR:-2.0}

SWEEP_DIR="data/submissions/checkpoints/ensemble_bb_$(date +%Y%m%d_%H%M%S)"
# Use a safe dir name from backbone
BB_SHORT=$(echo "$BACKBONE" | tr '.' '_')
RUN_DIR="$SWEEP_DIR/$BB_SHORT"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Ensemble backbone training — $BACKBONE"
echo "Job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "male_factor=$MALE_FACTOR"
echo "Checkpoint dir: $RUN_DIR"
echo "=========================================="
nvidia-smi

python -m src.train \
    --data_root data/raw \
    --checkpoint_dir "$RUN_DIR" \
    --backbone "$BACKBONE" \
    --batch_size 64 \
    --num_epochs 30 \
    --freeze_epochs 2 \
    --lr_head 3e-4 \
    --lr_backbone 3e-5 \
    --warmup_epochs 2 \
    --patience 30 \
    --loss wmse \
    --male_factor "$MALE_FACTOR" \
    --no_occlusion_safe

echo "=========================================="
echo "$BACKBONE done at $(date)"
echo "=========================================="
