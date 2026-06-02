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
