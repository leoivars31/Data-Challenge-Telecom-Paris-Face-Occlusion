#!/bin/bash
#SBATCH --job-name=face-sweep-mf
#SBATCH --output=%x_%j_%a.out
#SBATCH --error=%x_%j_%a.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

# Map array index to male_factor value
MALE_FACTORS=(1.0 1.5 2.0 2.5 3.0)
MF=${MALE_FACTORS[$SLURM_ARRAY_TASK_ID]}

SWEEP_DIR="data/submissions/checkpoints/sweep_mf_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$SWEEP_DIR/mf_${MF}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Male factor sweep — mf=$MF"
echo "Job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname) at $(date)"
echo "Checkpoint dir: $RUN_DIR"
echo "=========================================="
nvidia-smi

python -m src.train \
    --data_root data/raw \
    --checkpoint_dir "$RUN_DIR" \
    --batch_size 128 \
    --num_epochs 30 \
    --freeze_epochs 2 \
    --lr_head 3e-4 \
    --lr_backbone 3e-5 \
    --warmup_epochs 2 \
    --patience 30 \
    --loss wmse \
    --male_factor "$MF" \
    --no_occlusion_safe \
    --fold 0 \
    --n_splits 5

echo "=========================================="
echo "mf=$MF done at $(date)"
echo "=========================================="
