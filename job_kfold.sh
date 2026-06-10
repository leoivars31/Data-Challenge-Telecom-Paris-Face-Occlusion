#!/bin/bash

set -e

DATA_ROOT="data/raw"

echo "=== 1 convnextv2 base ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_convnextv2.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_025055/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_031335/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_044757/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_062159/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_063613/best_model_fold4.pt

echo "=== 2 eva02 base ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_eva02.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_064628/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_073712/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_091219/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_112136/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_112454/best_model_fold4.pt

echo "=== 3 tiny ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_tiny.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_124017/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_133205/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_150730/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_152706/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_153347/best_model_fold4.pt

echo "=== 4 convnextv2 + eva02 ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_convnextv2_eva02.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_025055/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_031335/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_044757/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_062159/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_063613/best_model_fold4.pt \
  data/submissions/checkpoints/kfold_20260610_064628/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_073712/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_091219/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_112136/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_112454/best_model_fold4.pt

echo "=== 5 convnextv2 + tiny ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_convnextv2_tiny.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_025055/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_031335/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_044757/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_062159/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_063613/best_model_fold4.pt \
  data/submissions/checkpoints/kfold_20260610_124017/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_133205/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_150730/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_152706/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_153347/best_model_fold4.pt

echo "=== 6 eva02 + tiny ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_eva02_tiny.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_064628/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_073712/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_091219/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_112136/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_112454/best_model_fold4.pt \
  data/submissions/checkpoints/kfold_20260610_124017/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_133205/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_150730/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_152706/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_153347/best_model_fold4.pt

echo "=== 7 ALL ==="
python -m src.predict \
  --data_root $DATA_ROOT \
  --tta_scales 1.0 0.9 1.1 \
  --output data/submissions/test_pred_all15.csv \
  --checkpoints \
  data/submissions/checkpoints/kfold_20260610_025055/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_031335/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_044757/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_062159/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_063613/best_model_fold4.pt \
  data/submissions/checkpoints/kfold_20260610_064628/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_073712/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_091219/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_112136/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_112454/best_model_fold4.pt \
  data/submissions/checkpoints/kfold_20260610_124017/best_model_fold0.pt \
  data/submissions/checkpoints/kfold_20260610_133205/best_model_fold1.pt \
  data/submissions/checkpoints/kfold_20260610_150730/best_model_fold2.pt \
  data/submissions/checkpoints/kfold_20260610_152706/best_model_fold3.pt \
  data/submissions/checkpoints/kfold_20260610_153347/best_model_fold4.pt