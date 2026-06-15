# Face Occlusion Rate Regression

Data Challenge **Idemia & Télécom Paris** — June 2026  
**Team: Interpol · Final score (leaderboard): 0.00100**

---

## Problem & metric

We regress a continuous occlusion score ∈ [0, 1] from 224×224 face images.

The metric weights heavy occlusions more:

$$Err = \frac{\sum_i w_i (p_i - GT_i)^2}{\sum_i w_i}, \quad w_i = \frac{1}{30} + GT_i$$

The final score penalizes the performance gap between genders:

$$Score = \frac{Err_F + Err_M}{2} + |Err_F - Err_M|$$

> **Key property**: in the observed regime $Err_M > Err_F$, the score rewrites as $1.5\,Err_M - 0.5\,Err_F$ — reducing error on males is worth 3× reducing it on females.

---

## Final solution

**Ensemble of 2 backbones × 5 folds = 10 models**, each with TTA (horizontal flip).

| Backbone | timm name | CV Score (mean ± std) | LB Score |
|---|---|---|---|
| ConvNeXtV2 Base | `convnextv2_base.fcmae_ft_in22k_in1k` | 0.001464 ± 0.000227 | 0.00102 |
| EVA-02 Base | `eva02_base_patch14_224.mim_in22k` | 0.001532 ± 0.000211 | 0.00107 |
| **ConvNeXtV2 + EVA-02** | — | — | **0.00100** |

### Architecture

Pre-trained `timm` backbone (ImageNet-22k) as feature extractor (`num_classes=0`), followed by a regression head:

```
Dropout(0.3) → Linear(d, 256) → GELU → Dropout(0.15) → Linear(256, 1) → Sigmoid
```

### Training (two phases)

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| LR head / backbone | 3e-4 / 3e-5 |
| Weight decay | 1e-2 |
| Schedule | Cosine + warmup (2 ep) |
| Freeze / max epochs / patience | 2 / 30 / 30 |
| Effective batch | 128 (batch 64 × grad\_accum 2) |
| Loss | Weighted-MSE (= metric) |
| K-fold | 5 stratified splits |
| TTA | Horizontal flip |

**Phase 1** — frozen backbone, head only (2 epochs, AdamW lr=3e-4).  
**Phase 2** — full unfreeze, differentiated LR head/backbone, cosine decay with warmup, AMP enabled, early stopping on validation score.

### Augmentations

**Light** regime (occlusion-safe): `HFlip`, `ColorJitter`, `RandomAffine`, `GaussianBlur`. No `RandomErasing` (effect not robustly decisive in k-fold, see §Experiments).

### Ensemble & prediction

The final prediction averages the 10 models (2 backbones × 5 folds), each evaluated with horizontal flip TTA. The final prediction also uses light multi-scale TTA (scales 1.0, 0.9, 1.1 × flip = 6 views per model).

---

## Reproducing the final solution

### 1. Installation

```bash
git clone https://github.com/leoivars31/Data-Challenge-Telecom-Paris-Face-Occlusion.git
cd Data-Challenge-Telecom-Paris-Face-Occlusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create the Slurm logs directory:

```bash
mkdir -p logs
```

### 2. Data

Place the data in `data/raw/`:

```
data/raw/
├── train.csv               # filename, FaceOcclusion, gender
├── test_students.csv       # filename (to predict)
└── Crop_224_5fp_100K/      # ~100k .webp images (224×224)
```

### 3. Training — 2 backbones × 5 folds

Each command submits a Slurm array of 5 jobs (one per fold):

```bash
# ConvNeXtV2 Base — 5 folds
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k" sbatch jobs/job_kfold.sh

# EVA-02 Base — 5 folds
BACKBONE="eva02_base_patch14_224.mim_in22k" sbatch jobs/job_kfold.sh
```

Checkpoints are saved to `data/submissions/checkpoints/kfold_<timestamp>/best_model_fold{0..4}.pt`.

### 4. Prediction — ensemble + TTA

Update the checkpoint paths in `jobs/predict_ensemble.sh`, then:

```bash
sbatch jobs/predict_ensemble.sh
```

Or directly from the command line:

```bash
python -m src.predict \
    --data_root data/raw \
    --checkpoints \
        data/submissions/checkpoints/<convnextv2_run_fold0>/best_model_fold0.pt \
        data/submissions/checkpoints/<convnextv2_run_fold1>/best_model_fold1.pt \
        data/submissions/checkpoints/<convnextv2_run_fold2>/best_model_fold2.pt \
        data/submissions/checkpoints/<convnextv2_run_fold3>/best_model_fold3.pt \
        data/submissions/checkpoints/<convnextv2_run_fold4>/best_model_fold4.pt \
        data/submissions/checkpoints/<eva02_run_fold0>/best_model_fold0.pt \
        data/submissions/checkpoints/<eva02_run_fold1>/best_model_fold1.pt \
        data/submissions/checkpoints/<eva02_run_fold2>/best_model_fold2.pt \
        data/submissions/checkpoints/<eva02_run_fold3>/best_model_fold3.pt \
        data/submissions/checkpoints/<eva02_run_fold4>/best_model_fold4.pt \
    --tta_scales 1.0 0.9 1.1 \
    --output data/submissions/test_predictions.csv \
    --batch_size 64
```

---

## Experiments

### Why K-fold

The metric is dominated by rare highly-occluded samples: the score of a single split varies by ≈6×10⁻⁴ across folds for the same configuration. This variance exceeds most measurable effects; k-fold (mean ± std) is therefore essential for robust decision-making.

### Sweep fairness_lambda

Tests `loss = wMSE + λ|wMSE_F − wMSE_M|` for `λ ∈ {0, 0.25, 0.5, 1.0, 2.0, 5.0}`.  
Result: increasing λ **degrades the score** (ErrF rises to meet ErrM instead of ErrM dropping) with no marginal gap gain. **λ = 0 is optimal.**

```bash
sbatch jobs/job_sweep_lambda.sh
```

### Sweep male_factor × K-fold

Multiplies the weights of male samples in the loss by `male_factor ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` (25 runs = 5 values × 5 folds).  
Result: ErrM remains **flat** across the entire range (amplitude 7×10⁻⁶ ≪ inter-fold std). Male error is an intrinsic difficulty, not a weighting artifact. **male_factor = 1 is optimal.**

```bash
sbatch jobs/job_sweep_male_factor.sh
```

### RandomErasing

Comparison of light regime (without RE) vs. aggressive (with RE) in k-fold. The effect is of the same order as inter-fold variance and appears backbone-dependent. Since not robustly decisive, the final solution uses the **light regime**.

### Backbone comparison (leaderboard scores)

| Configuration | #models | LB Score |
|---|---|---|
| ConvNeXtV2 Base | 5 | 0.00102 |
| EVA-02 Base | 5 | 0.00107 |
| ConvNeXt-Tiny | 5 | 0.00138 |
| **ConvNeXtV2 + EVA-02** | **10** | **0.00100** |
| 3 backbones | 15 | 0.00109 |
| ConvNeXtV2 + Tiny | 10 | 0.00116 |
| EVA-02 + Tiny | 10 | 0.00116 |

ConvNeXt-Tiny is significantly weaker and **degrades** any ensemble containing it. The final ensemble retains only ConvNeXtV2 + EVA-02.

To train the 3 backbones in parallel on a single split (exploration):

```bash
sbatch jobs/job_ensemble_backbones.sh
```

### Baseline single-split

Training a single ConvNeXtV2 model on a stratified 85/15 split:

```bash
sbatch jobs/job_train.sh
```

---

## Project structure

```
├── src/
│   ├── dataset.py          # PyTorch dataset, transforms, K-fold splits
│   ├── model.py            # Architecture (timm backbone + regression head)
│   ├── train.py            # Two-phase training (CLI argparse)
│   ├── predict.py          # Ensemble inference + TTA + submission CSV
│   ├── utils.py            # Metric, losses, seed
│   ├── diagnostics.py      # Error decomposition by gender × occlusion bin
│   └── calibration.py      # Isotonic calibration per gender (experimental)
├── jobs/
│   ├── job_kfold.sh              # ⭐ K-fold training (final solution)
│   ├── predict_ensemble.sh       # ⭐ Multi-backbone ensemble prediction + TTA
│   ├── job_train.sh              # Baseline single-split
│   ├── job_ensemble_backbones.sh # Multi-backbone without K-fold (exploration)
│   ├── job_sweep_lambda.sh       # Sweep fairness_lambda
│   ├── job_sweep_male_factor.sh  # Sweep male_factor × K-fold (25 runs)
│   └── job_predict.sh            # Single-run prediction (example)
├── notebooks/
│   ├── eda.ipynb                 # Exploratory data analysis
│   ├── sweep_analysis.ipynb      # Lambda sweep results analysis
│   ├── sweep_male_factor.ipynb   # Male factor sweep results analysis
│   ├── compare_models.ipynb      # Backbone comparison (CV vs LB)
│   └── DataChallengeExample.ipynb
├── data/
│   ├── raw/                      # Raw data (not versioned)
│   └── submissions/              # Submission CSVs
├── docs/
│   └── task_brief.pdf            # Challenge brief (provided by organizers)
├── test.py                       # CV scores by fold (numerical reference)
├── requirements.txt
└── README.md
```

---

## CLI reference

### `src.train`

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `data/raw` | Data directory |
| `--backbone` | `convnext_tiny.fb_in22k_ft_in1k` | timm backbone |
| `--batch_size` | 128 | Batch size |
| `--grad_accum_steps` | 1 | Gradient accumulation steps |
| `--num_epochs` | 30 | Max epochs |
| `--freeze_epochs` | 2 | Epochs with frozen backbone (phase 1) |
| `--lr_head` | 3e-4 | Head learning rate |
| `--lr_backbone` | 3e-5 | Backbone learning rate |
| `--warmup_epochs` | 2 | Linear warmup (phase 2) |
| `--patience` | 30 | Early stopping (epochs without improvement) |
| `--loss` | `wmse` | Loss: `wmse`, `surrogate`, `groupdro` |
| `--male_factor` | 1.0 | Male sample weight multiplier in the loss |
| `--fairness_lambda` | 0.0 | Gender gap regularization (deprecated, see §Experiments) |
| `--fold` | None | Fold index (0..n\_splits-1); None = single split |
| `--n_splits` | 5 | Number of folds |
| `--occlusion_safe` | True | Light augmentations without RandomErasing (default) |
| `--no_occlusion_safe` | — | Aggressive augmentations with RandomErasing |
| `--checkpoint_dir` | `data/submissions/checkpoints` | Save directory |
| `--val_ratio` | 0.15 | Val ratio (single split mode only) |

### `src.predict`

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `data/raw` | Data directory |
| `--checkpoints` | — | List of checkpoint paths (ensemble) |
| `--tta_scales` | `[1.0]` | TTA scales (e.g., `1.0 0.9 1.1`) |
| `--no_tta` | — | Disable flip TTA |
| `--output` | `data/submissions/test_predictions.csv` | Output CSV file |
| `--batch_size` | 64 | Batch size |

### `src.diagnostics`

```bash
python -m src.diagnostics --checkpoint path/to/best_model.pt --data_root data/raw --fold 0
```

Outputs: occlusion distribution by gender, error decomposition by gender × occlusion bin.

### `src.calibration`

```bash
python -m src.calibration --checkpoint path/to/model.pt --data_root data/raw --fold 0
```

Isotonic calibration per gender on validation predictions (experimental, requires `predict_gender=True` in the checkpoint).

---

## Environment

- Python 3.11
- PyTorch + CUDA 12.4
- `timm` (PyTorch Image Models)
- GPU: NVIDIA RTX 3090 (Télécom Paris cluster)

---

## Authors

A. Donnat, O. Fekih, L. Ivars — Télécom Paris, 2026
