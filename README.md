# Data Challenge Télécom Paris — Face Occlusion

Challenge fourni par **Idemia**.

## Énoncé

L'objectif de ce data challenge est de **prédire le pourcentage d'occlusion d'un visage** à partir d'images de visages recadrées (224×224). Le modèle doit produire un unique score d'occlusion par image. L'évaluation repose sur une erreur pondérée qui pénalise davantage les fortes occlusions, ainsi qu'un écart de performance entre femmes et hommes.

Les résultats doivent être reproductibles.

### Données

- Plus de **100 000 images** de visages humains avec leur label d'occlusion.
- Le **genre** (homme/femme) est fourni pour les données d'entraînement uniquement.


| Fichier | Description |
|---|---|
| `data/raw/train.csv` | Labels d'entraînement (`filename`, `FaceOcclusion`, `gender`) |
| `data/raw/test_students.csv` | Fichiers de test sans labels (à prédire) |

### Métrique d'évaluation

L'erreur pondérée met davantage de poids sur les visages fortement occultés :

$$Err = \frac{\sum_{i} w_i (p_i - GT_i)^2}{\sum_{i} w_i}, \quad w_i = \frac{1}{30} + GT_i$$

Le score final pénalise également l'écart de performance entre femmes et hommes :

$$Score = \frac{Err_F + Err_M}{2} + |Err_F - Err_M|$$

### Contraintes

- Bien performer sur les visages **fortement occultés** (pondération plus élevée).
- Avoir des performances **similaires entre femmes et hommes** (pénalité sur l'écart).

### Soumission

Le fichier de soumission suit le format `test_predictions.csv` avec les colonnes `filename`, `FaceOcclusion` et une colonne `gender` factice requise par la plateforme.

## Arborescence du projet

```
├── data/
│   ├── raw/                # Données brutes — ignoré par git
│   │   ├── train.csv
│   │   ├── test_students.csv
│   │   └── Crop_224_5fp_100K/  # Images (100k .webp)
│   └── submissions/        # Fichiers de prédictions à soumettre
│       └── checkpoints/    # Checkpoints du modèle
├── docs/
│   └── task_brief.pdf      # Énoncé officiel du challenge
├── src/
│   ├── dataset.py          # Dataset PyTorch, transforms, dataloaders
│   ├── model.py            # Architecture du modèle (ConvNeXt-Tiny via timm)
│   ├── train.py            # Script d'entraînement (CLI avec argparse)
│   ├── predict.py          # Inférence + génération du CSV de soumission
│   └── utils.py            # Métrique, loss pondérée, seed
├── notebooks/
│   ├── DataChallengeExample.ipynb  # Notebook de départ fourni par les organisateurs
│   ├── eda.ipynb           # Analyse exploratoire des données
│   └── training.ipynb      # Entraînement, courbes, analyse des prédictions
├── job_train.sh            # Script Slurm pour le cluster GPU
├── .gitignore
├── requirements.txt
└── README.md
```

## Utilisation

### Entraînement (sur le cluster GPU)

```bash
# En interactif
python -m src.train --data_root data/raw --batch_size 32 --num_epochs 20

# Ou via Slurm (recommandé)
sbatch job_train.sh
squeue --me  # suivre l'état du job
```

### Génération des prédictions

```bash
python -m src.predict --data_root data/raw
```

### Arguments disponibles

| Argument | Défaut | Description |
|---|---|---|
| `--data_root` | `data/raw` | Chemin vers les données (train.csv, Crop_224_5fp_100K/) |
| `--batch_size` | 32 / 64 | Taille du batch |
| `--num_epochs` | 20 | Nombre d'époques max |
| `--backbone` | `convnext_tiny.fb_in22k_ft_in1k` | Backbone timm |
| `--patience` | 5 | Early stopping |
| `--no_tta` | - | Désactiver le TTA (predict.py) |

Les checkpoints sont sauvegardés dans `data/submissions/checkpoints/` et les prédictions dans `data/submissions/test_predictions.csv`.

---

## Améliorations (v2)

### Nouveaux flags `src/train.py`

| Flag | Défaut | Description |
|---|---|---|
| `--male_factor` | 1.0 | Surpondération des hommes dans la loss (>1 = plus de poids) |
| `--loss` | `wmse` | Loss : `wmse`, `surrogate` (métrique diff.), `groupdro` (worst-group) |
| `--occlusion_safe` / `--no_occlusion_safe` | safe | Augmentations douces (retire RandomErasing) |
| `--fold` | None | Index de fold K-fold (0..n_splits-1) |
| `--n_splits` | 5 | Nombre de folds |
| `--predict_gender` | off | Active la tête auxiliaire de prédiction du genre |
| `--gender_loss_weight` | 0.2 | Poids de la BCE genre dans la loss totale |

### Nouveaux flags `src/predict.py`

| Flag | Défaut | Description |
|---|---|---|
| `--checkpoints` | - | Plusieurs checkpoints pour ensemble (nargs+) |
| `--tta_scales` | `1.0` | Échelles TTA (ex: `1.0 0.9 1.1`) |
| `--calibrators` | - | Pickle de calibrateurs pour calibration conditionnelle au genre |

### Workflow recommandé

1. **Sweep `male_factor`** sur fold 0 → trouver le meilleur facteur
   ```bash
   sbatch job_sweep_male_factor.sh
   ```

2. **K-fold 5 plis** avec le meilleur `male_factor` + `--occlusion_safe` + `--predict_gender`
   ```bash
   MALE_FACTOR=2.0 sbatch job_kfold.sh
   ```

3. **Fit calibrateurs** par genre sur la val de chaque fold
   ```bash
   python -m src.calibration --checkpoint path/to/best_model_fold0.pt --fold 0
   ```

4. **Predict en ensemble** des 5 folds + TTA multi-échelle + calibration
   ```bash
   python -m src.predict \
       --checkpoints fold0/best_model_fold0.pt fold1/best_model_fold1.pt ... \
       --tta_scales 1.0 0.9 1.1 \
       --calibrators path/to/calibrators.pkl
   ```

5. **Soumettre** et garder un budget de soumissions pour calibrer val↔leaderboard.

### Diagnostics

```bash
python -m src.diagnostics --checkpoint path/to/best_model.pt --data_root data/raw
```

Produit :
- Distribution d'occlusion par genre (PNG)
- Décomposition de l'erreur par genre × bin d'occlusion
- Corrélation `male_factor` vs gap (si plusieurs historiques disponibles)
