#!/bin/bash
#SBATCH --job-name=face-predict
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1
eval "$(conda shell.bash hook)"
conda activate face-occlusion
cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

CONVNEXTV2=(
    data/submissions/checkpoints/kfold_20260610_025055/best_model_fold0.pt
    data/submissions/checkpoints/kfold_20260610_031335/best_model_fold1.pt
    data/submissions/checkpoints/kfold_20260610_044757/best_model_fold2.pt
    data/submissions/checkpoints/kfold_20260610_062159/best_model_fold3.pt
    data/submissions/checkpoints/kfold_20260610_063613/best_model_fold4.pt
)

EVA02=(
    data/submissions/checkpoints/kfold_20260610_064628/best_model_fold0.pt
    data/submissions/checkpoints/kfold_20260610_073712/best_model_fold1.pt
    data/submissions/checkpoints/kfold_20260610_091219/best_model_fold2.pt
    data/submissions/checkpoints/kfold_20260610_112136/best_model_fold3.pt
    data/submissions/checkpoints/kfold_20260610_112454/best_model_fold4.pt
)

TINY=(
    data/submissions/checkpoints/kfold_20260610_124017/best_model_fold0.pt
    data/submissions/checkpoints/kfold_20260610_133205/best_model_fold1.pt
    data/submissions/checkpoints/kfold_20260610_150730/best_model_fold2.pt
    data/submissions/checkpoints/kfold_20260610_152706/best_model_fold3.pt
    data/submissions/checkpoints/kfold_20260610_153347/best_model_fold4.pt
)

COMMON_ARGS="--data_root data/raw --tta_scales 1.0 0.9 1.1 --batch_size 64"

mkdir -p data/submissions

echo "=== 1/7 convnextv2_base seul ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_convnextv2.csv \
    --checkpoints "${CONVNEXTV2[@]}"

echo "=== 2/7 eva02_base seul ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_eva02.csv \
    --checkpoints "${EVA02[@]}"

echo "=== 3/7 convnext_tiny seul ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_tiny.csv \
    --checkpoints "${TINY[@]}"

echo "=== 4/7 convnextv2 + eva02 (10 modèles) ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_convnextv2_eva02.csv \
    --checkpoints "${CONVNEXTV2[@]}" "${EVA02[@]}"

echo "=== 5/7 convnextv2 + tiny (10 modèles) ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_convnextv2_tiny.csv \
    --checkpoints "${CONVNEXTV2[@]}" "${TINY[@]}"

echo "=== 6/7 eva02 + tiny (10 modèles) ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_eva02_tiny.csv \
    --checkpoints "${EVA02[@]}" "${TINY[@]}"

echo "=== 7/7 les 3 backbones (15 modèles) ==="
python -m src.predict $COMMON_ARGS \
    --output data/submissions/test_pred_all15.csv \
    --checkpoints "${CONVNEXTV2[@]}" "${EVA02[@]}" "${TINY[@]}"

echo ""
echo "Done! Fichiers générés dans data/submissions/:"
