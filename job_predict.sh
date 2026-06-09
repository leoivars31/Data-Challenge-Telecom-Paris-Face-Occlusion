#!/bin/bash
#SBATCH --job-name=face-predict
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1:00:00

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate face-occlusion
cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

python -m src.predict \
    --data_root data/raw \
    --checkpoints \
        data/submissions/checkpoints/kfold_20260608_183206/best_model_fold0.pt \
        data/submissions/checkpoints/kfold_20260608_213750/best_model_fold1.pt \
        data/submissions/checkpoints/kfold_20260609_013241/best_model_fold2.pt \
        data/submissions/checkpoints/kfold_20260609_021617/best_model_fold3.pt \
        data/submissions/checkpoints/kfold_20260609_021617/best_model_fold4.pt \
        data/submissions/checkpoints/ensemble_bb_20260609_034224/convnext_tiny_fb_in22k_ft_in1k/best_model.pt \
        data/submissions/checkpoints/ensemble_bb_20260609_041240/convnextv2_base_fcmae_ft_in22k_in1k/best_model.pt \
        data/submissions/checkpoints/ensemble_bb_20260609_041728/eva02_base_patch14_224_mim_in22k/best_model.pt \
    --tta_scales 1.0 0.9 1.1 \
    --output data/submissions/test_predictions.csv \
    --batch_size 64

echo "Predictions done!"
