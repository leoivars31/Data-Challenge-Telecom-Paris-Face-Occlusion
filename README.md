# Régression du taux d'occlusion faciale

Data Challenge **Idemia & Télécom Paris** — Juin 2026  
**Groupe : Interpol · Score final (leaderboard) : 0.00100**

---

## Problème & métrique

On régresse un score d'occlusion continu ∈ [0, 1] à partir d'images de visages 224×224.

La métrique pondère davantage les fortes occlusions :

$$Err = \frac{\sum_i w_i (p_i - GT_i)^2}{\sum_i w_i}, \quad w_i = \frac{1}{30} + GT_i$$

Le score final pénalise l'écart de performance entre genres :

$$Score = \frac{Err_F + Err_M}{2} + |Err_F - Err_M|$$

> **Propriété clé** : dans le régime observé $Err_M > Err_F$, le score se réécrit $1.5\,Err_M - 0.5\,Err_F$ — réduire l'erreur sur les hommes vaut 3× réduire celle sur les femmes.

---

## Solution finale

**Ensemble de 2 backbones × 5 folds = 10 modèles**, chacun avec TTA (flip horizontal).

| Backbone | Nom timm | CV Score (mean ± std) | LB Score |
|---|---|---|---|
| ConvNeXtV2 Base | `convnextv2_base.fcmae_ft_in22k_in1k` | 0.001464 ± 0.000227 | 0.00102 |
| EVA-02 Base | `eva02_base_patch14_224.mim_in22k` | 0.001532 ± 0.000211 | 0.00107 |
| **ConvNeXtV2 + EVA-02** | — | — | **0.00100** |

### Architecture

Backbone `timm` pré-entraîné (ImageNet-22k) en extracteur de features (`num_classes=0`), suivi d'une tête de régression :

```
Dropout(0.3) → Linear(d, 256) → GELU → Dropout(0.15) → Linear(256, 1) → Sigmoid
```

### Entraînement (deux phases)

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | AdamW |
| LR tête / backbone | 3e-4 / 3e-5 |
| Weight decay | 1e-2 |
| Schedule | Cosine + warmup (2 ep) |
| Freeze / epochs max / patience | 2 / 30 / 30 |
| Batch effectif | 128 (batch 64 × grad\_accum 2) |
| Loss | Weighted-MSE (= métrique) |
| K-fold | 5 splits stratifiés |
| TTA | Flip horizontal |

**Phase 1** — backbone gelé, tête seule (2 epochs, AdamW lr=3e-4).  
**Phase 2** — dégel complet, LR différenciés tête/backbone, cosine decay avec warmup, AMP activé, early stopping sur le score de validation.

### Augmentations

Régime **léger** (occlusion-safe) : `HFlip`, `ColorJitter`, `RandomAffine`, `GaussianBlur`. Pas de `RandomErasing` (effet non tranché de manière robuste en k-fold, cf. §Expériences).

### Ensemble & prédiction

La prédiction finale moyenne les 10 modèles (2 backbones × 5 folds), chacun évalué avec TTA flip horizontal. La prédiction finale utilise également un léger multi-scale TTA (scales 1.0, 0.9, 1.1 × flip = 6 vues par modèle).

---

## Reproduire la solution finale

### 1. Installation

```bash
git clone https://github.com/leoivars31/Data-Challenge-Telecom-Paris-Face-Occlusion.git
cd Data-Challenge-Telecom-Paris-Face-Occlusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Créer le dossier de logs pour Slurm :

```bash
mkdir -p logs
```

### 2. Données

Placer les données dans `data/raw/` :

```
data/raw/
├── train.csv               # filename, FaceOcclusion, gender
├── test_students.csv       # filename (à prédire)
└── Crop_224_5fp_100K/      # ~100k images .webp (224×224)
```

### 3. Entraînement — 2 backbones × 5 folds

Chaque commande soumet un array de 5 jobs Slurm (un par fold) :

```bash
# ConvNeXtV2 Base — 5 folds
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k" sbatch jobs/job_kfold.sh

# EVA-02 Base — 5 folds
BACKBONE="eva02_base_patch14_224.mim_in22k" sbatch jobs/job_kfold.sh
```

Les checkpoints sont sauvegardés dans `data/submissions/checkpoints/kfold_<timestamp>/best_model_fold{0..4}.pt`.

### 4. Prédiction — ensemble + TTA

Adapter les chemins de checkpoints dans `jobs/predict_ensemble.sh`, puis :

```bash
sbatch jobs/predict_ensemble.sh
```

Ou directement en ligne de commande :

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

## Expériences réalisées

### Pourquoi le K-fold

La métrique est dominée par de rares échantillons fortement occultés : le score d'un split unique varie de ≈6×10⁻⁴ entre folds pour une même configuration. Cette variance dépasse la plupart des effets à mesurer ; le k-fold (moyenne ± écart-type) est donc indispensable pour prendre des décisions robustes.

### Sweep fairness_lambda

Teste `loss = wMSE + λ|wMSE_F − wMSE_M|` pour `λ ∈ {0, 0.25, 0.5, 1.0, 2.0, 5.0}`.  
Résultat : augmenter λ **dégrade le score** (ErrF monte pour rejoindre ErrM au lieu que ErrM descende) sans gain de gap marginal. **λ = 0 est optimal.**

```bash
sbatch jobs/job_sweep_lambda.sh
```

### Sweep male_factor × K-fold

Multiplie les poids des échantillons masculins dans la loss par `male_factor ∈ {1.0, 1.5, 2.0, 2.5, 3.0}` (25 runs = 5 valeurs × 5 folds).  
Résultat : ErrM reste **plat** sur toute la plage (amplitude 7×10⁻⁶ ≪ écart-type inter-fold). L'erreur masculine est une difficulté intrinsèque, pas un artefact de pondération. **male_factor = 1 est optimal.**

```bash
sbatch jobs/job_sweep_male_factor.sh
```

### RandomErasing

Comparaison régime léger (sans RE) vs. agressif (avec RE) en k-fold. L'effet est du même ordre que la variance inter-fold et semble dépendant du backbone. Puisque non tranché de manière robuste, la solution finale utilise le **régime léger**.

### Comparaison des backbones (scores leaderboard)

| Configuration | #modèles | Score LB |
|---|---|---|
| ConvNeXtV2 Base | 5 | 0.00102 |
| EVA-02 Base | 5 | 0.00107 |
| ConvNeXt-Tiny | 5 | 0.00138 |
| **ConvNeXtV2 + EVA-02** | **10** | **0.00100** |
| 3 backbones | 15 | 0.00109 |
| ConvNeXtV2 + Tiny | 10 | 0.00116 |
| EVA-02 + Tiny | 10 | 0.00116 |

ConvNeXt-Tiny est nettement plus faible et **dégrade** tout ensemble le contenant. L'ensemble final retient uniquement ConvNeXtV2 + EVA-02.

Pour entraîner les 3 backbones en parallèle sur split unique (exploration) :

```bash
sbatch jobs/job_ensemble_backbones.sh
```

### Baseline single-split

Entraînement d'un seul modèle ConvNeXtV2 sur un split 85/15 stratifié :

```bash
sbatch jobs/job_train.sh
```

---

## Arborescence du projet

```
├── src/
│   ├── dataset.py          # Dataset PyTorch, transforms, K-fold splits
│   ├── model.py            # Architecture (backbone timm + tête de régression)
│   ├── train.py            # Entraînement two-phase (CLI argparse)
│   ├── predict.py          # Inférence ensemble + TTA + CSV soumission
│   ├── utils.py            # Métrique, losses, seed
│   ├── diagnostics.py      # Décomposition d'erreur par genre × bin d'occlusion
│   └── calibration.py      # Calibration isotonique par genre (expérimental)
├── jobs/
│   ├── job_kfold.sh              # ⭐ Entraînement K-fold (solution finale)
│   ├── predict_ensemble.sh       # ⭐ Prédiction ensemble multi-backbone + TTA
│   ├── job_train.sh              # Baseline single-split
│   ├── job_ensemble_backbones.sh # Multi-backbone sans K-fold (exploration)
│   ├── job_sweep_lambda.sh       # Sweep fairness_lambda
│   ├── job_sweep_male_factor.sh  # Sweep male_factor × K-fold (25 runs)
│   └── job_predict.sh            # Prédiction single-run (exemple)
├── notebooks/
│   ├── eda.ipynb                 # Analyse exploratoire
│   ├── sweep_analysis.ipynb      # Analyse résultats sweep lambda
│   ├── sweep_male_factor.ipynb   # Analyse résultats sweep male_factor
│   ├── compare_models.ipynb      # Comparaison des backbones (CV vs LB)
│   └── DataChallengeExample.ipynb
├── data/
│   ├── raw/                      # Données brutes (non versionné)
│   └── submissions/              # CSVs de soumission
├── docs/
│   └── task_brief.pdf            # Brief du challenge (fourni par les organisateurs)
├── test.py                       # Scores CV des folds (référence numérique)
├── requirements.txt
└── README.md
```

---

## Référence CLI

### `src.train`

| Argument | Défaut | Description |
|---|---|---|
| `--data_root` | `data/raw` | Répertoire des données |
| `--backbone` | `convnext_tiny.fb_in22k_ft_in1k` | Backbone timm |
| `--batch_size` | 128 | Taille du batch |
| `--grad_accum_steps` | 1 | Accumulation de gradient |
| `--num_epochs` | 30 | Nombre d'époques max |
| `--freeze_epochs` | 2 | Époques avec backbone gelé (phase 1) |
| `--lr_head` | 3e-4 | Learning rate de la tête |
| `--lr_backbone` | 3e-5 | Learning rate du backbone |
| `--warmup_epochs` | 2 | Warmup linéaire (phase 2) |
| `--patience` | 30 | Early stopping (epochs sans amélioration) |
| `--loss` | `wmse` | Loss : `wmse`, `surrogate`, `groupdro` |
| `--male_factor` | 1.0 | Sur-pondération des hommes dans la loss |
| `--fairness_lambda` | 0.0 | Régularisation gap de genre (déprécié, cf. §Expériences) |
| `--fold` | None | Index du fold (0..n\_splits-1) ; None = split unique |
| `--n_splits` | 5 | Nombre de folds |
| `--occlusion_safe` | True | Augmentations légères sans RandomErasing (défaut) |
| `--no_occlusion_safe` | — | Augmentations agressives avec RandomErasing |
| `--checkpoint_dir` | `data/submissions/checkpoints` | Répertoire de sauvegarde |
| `--val_ratio` | 0.15 | Ratio val (uniquement en mode split unique) |

### `src.predict`

| Argument | Défaut | Description |
|---|---|---|
| `--data_root` | `data/raw` | Répertoire des données |
| `--checkpoints` | — | Liste de chemins de checkpoints (ensemble) |
| `--tta_scales` | `[1.0]` | Échelles TTA (ex : `1.0 0.9 1.1`) |
| `--no_tta` | — | Désactiver le flip TTA |
| `--output` | `data/submissions/test_predictions.csv` | Fichier CSV de sortie |
| `--batch_size` | 64 | Taille du batch |

### `src.diagnostics`

```bash
python -m src.diagnostics --checkpoint path/to/best_model.pt --data_root data/raw --fold 0
```

Produit : distribution d'occlusion par genre, décomposition de l'erreur par genre × bin d'occlusion.

### `src.calibration`

```bash
python -m src.calibration --checkpoint path/to/model.pt --data_root data/raw --fold 0
```

Calibration isotonique par genre sur les prédictions de validation (expérimental, nécessite `predict_gender=True` dans le checkpoint).

---

## Environnement

- Python 3.11
- PyTorch + CUDA 12.4
- `timm` (PyTorch Image Models)
- GPU : NVIDIA RTX 3090 (cluster Télécom Paris)

---

## Auteurs

A. Donnat, O. Fekih, L. Ivars — Télécom Paris, 2026
