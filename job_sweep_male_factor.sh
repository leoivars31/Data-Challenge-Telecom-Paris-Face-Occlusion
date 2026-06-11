#!/bin/bash
#SBATCH --job-name=face-sweep-mf-kfold
#SBATCH --output=%x_%j_%a.out
#SBATCH --error=%x_%j_%a.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-24          # 5 male_factors x 5 folds = 25 runs

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

# --- Map array index -> (male_factor, fold) ---
MALE_FACTORS=(1.0 1.5 2.0 2.5 3.0)
N_FOLDS=5
MF_IDX=$(( SLURM_ARRAY_TASK_ID / N_FOLDS ))   # 0..4
FOLD=$(( SLURM_ARRAY_TASK_ID % N_FOLDS ))     # 0..4
MF=${MALE_FACTORS[$MF_IDX]}

BACKBONE="convnext_tiny.fb_in22k_ft_in1k"

# Un seul dossier pour tout l'array (SLURM_ARRAY_JOB_ID identique pour les 25 tâches)
SWEEP_DIR="data/submissions/checkpoints/sweep_mf_kfold_${SLURM_ARRAY_JOB_ID}"
RUN_DIR="$SWEEP_DIR/mf_${MF}_fold${FOLD}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "MF k-fold sweep — mf=$MF | fold=$FOLD/$N_FOLDS | backbone=$BACKBONE"
echo "Job $SLURM_JOB_ID (array ${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Checkpoint dir: $RUN_DIR"
echo "=========================================="
nvidia-smi

python -m src.train \
    --data_root data/raw \
    --checkpoint_dir "$RUN_DIR" \
    --backbone "$BACKBONE" \
    --batch_size 64 \
    --grad_accum_steps 2 \
    --num_epochs 30 \
    --freeze_epochs 2 \
    --lr_head 3e-4 \
    --lr_backbone 3e-5 \
    --warmup_epochs 2 \
    --patience 30 \
    --loss wmse \
    --male_factor "$MF" \
    --fold "$FOLD" \
    --n_splits 5 \
    --seed 42

echo "=========================================="
echo "mf=$MF fold=$FOLD done at $(date)"
echo "=========================================="