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

python -m src.train \
    --data_root data/raw \
    --batch_size 64 \
    --num_epochs 20 \
    --freeze_epochs 2 \
    --lr_head 1e-4 \
    --lr_backbone 1e-5 \
    --patience 5

echo "=========================================="
echo "Training done. Generating predictions..."
echo "=========================================="

python -m src.predict \
    --data_root data/raw \
    --batch_size 64

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
