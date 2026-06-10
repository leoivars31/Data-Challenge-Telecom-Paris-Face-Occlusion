#!/bin/bash
#SBATCH --job-name=sweep-lambda
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

module purge
module load python/3.11 cuda/12.4 miniconda3/25.5.1

eval "$(conda shell.bash hook)"
conda activate face-occlusion

cd ~/Data-Challenge-Telecom-Paris-Face-Occlusion

echo "=========================================="
echo "Lambda sweep - Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=========================================="
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ── Lambda values to test ──
LAMBDAS="0.0 0.25 0.5 1.0 2.0 5.0"

SWEEP_DIR="data/submissions/checkpoints/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SWEEP_DIR"

for LAMBDA in $LAMBDAS; do
    echo ""
    echo "=========================================="
    echo "Training with fairness_lambda=$LAMBDA"
    echo "=========================================="

    RUN_DIR="$SWEEP_DIR/lambda_${LAMBDA}"
    mkdir -p "$RUN_DIR"

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
        --fairness_lambda "$LAMBDA"

    echo "Done lambda=$LAMBDA"
done

# ── Summary: compare all runs ──
echo ""
echo "=========================================="
echo "SWEEP SUMMARY"
echo "=========================================="
python -c "
import json, os, glob

sweep_dir = '$SWEEP_DIR'
results = []
for run in sorted(glob.glob(os.path.join(sweep_dir, 'lambda_*'))):
    hp = os.path.join(run, 'history.json')
    if not os.path.isfile(hp):
        continue
    with open(hp) as f:
        h = json.load(f)
    best_idx = h['val_score'].index(min(h['val_score']))
    lam = os.path.basename(run).replace('lambda_', '')
    results.append({
        'lambda': lam,
        'best_epoch': h['epoch'][best_idx],
        'val_score': h['val_score'][best_idx],
        'err_f': h['err_f'][best_idx],
        'err_m': h['err_m'][best_idx],
        'gap': abs(h['err_f'][best_idx] - h['err_m'][best_idx]),
    })

print(f\"{'lambda':>8} {'score':>10} {'err_f':>10} {'err_m':>10} {'gap':>10} {'epoch':>6}\")
print('-' * 58)
for r in sorted(results, key=lambda x: x['val_score']):
    print(f\"{r['lambda']:>8} {r['val_score']:>10.6f} {r['err_f']:>10.6f} {r['err_m']:>10.6f} {r['gap']:>10.6f} {r['best_epoch']:>6}\")

best = min(results, key=lambda x: x['val_score'])
print(f\"\nBest: lambda={best['lambda']} with score={best['val_score']:.6f}\")
"

# Generate predictions for the best model
BEST_LAMBDA=$(python -c "
import json, os, glob
sweep_dir = '$SWEEP_DIR'
best_score, best_dir = float('inf'), ''
for run in glob.glob(os.path.join(sweep_dir, 'lambda_*')):
    hp = os.path.join(run, 'history.json')
    if not os.path.isfile(hp): continue
    with open(hp) as f: h = json.load(f)
    s = min(h['val_score'])
    if s < best_score: best_score, best_dir = s, run
print(best_dir)
")

echo "Generating predictions from best model: $BEST_LAMBDA"
python -m src.predict \
    --data_root data/raw \
    --checkpoint "$BEST_LAMBDA/best_model.pt" \
    --output "$BEST_LAMBDA/test_predictions.csv" \
    --batch_size 64

# Symlink best run
ln -sfn "$(basename "$BEST_LAMBDA")" "$SWEEP_DIR/best"
ln -sfn "$(basename "$SWEEP_DIR")/best" data/submissions/checkpoints/latest

echo "=========================================="
echo "Sweep finished at $(date)"
echo "=========================================="
