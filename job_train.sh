#!/bin/bash
#SBATCH --job-name=face-occlusion
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

echo "=========================================="
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "=========================================="
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
echo "=========================================="

RUN_DIR="data/submissions/checkpoints/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
echo "Checkpoint dir: $RUN_DIR"

VAL_RATIO=${VAL_RATIO:-0.15}

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
    --fairness_lambda 0.0 \
    --val_ratio "$VAL_RATIO"

# Symlink latest run for easy access
ln -sfn "$(basename "$RUN_DIR")" data/submissions/checkpoints/latest
echo "Symlinked latest -> $(basename "$RUN_DIR")"

echo "=========================================="
echo "Training done. Generating predictions..."
echo "=========================================="

python -m src.predict \
    --data_root data/raw \
    --checkpoint "$RUN_DIR/best_model.pt" \
    --output "$RUN_DIR/test_predictions.csv" \
    --batch_size 64

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
