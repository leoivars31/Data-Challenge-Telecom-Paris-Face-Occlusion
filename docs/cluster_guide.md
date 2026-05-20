# Utilisation du cluster GPU Télécom Paris

Documentation officielle : https://computing.telecom-paris.fr/

## Prérequis

- Être sur le réseau Télécom Paris (en salle) ou connecté via **VPN**
- Vérifier que ton compte a accès au cluster : https://computing.telecom-paris.fr/docs/getting-started/prerequisites/

En tant qu'étudiant, accès limité aux GPU **P100 et 3090** (max 4 GPU, max 36h par job).

---

## 1. Connexion SSH au cluster

### Première connexion — configurer les clés SSH

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
ssh-copy-id <tp-username>@gpu-gw.enst.fr
```

### Configurer un alias SSH

Éditer `~/.ssh/config` :

```
# Direct connection (on-campus or VPN)
Host cluster
    HostName gpu-gw.enst.fr
    User <tp-username>
    IdentityFile ~/.ssh/id_ed25519
```

Ensuite se connecter simplement avec :

```bash
ssh cluster
```

### Se connecter et démarrer une session interactive GPU

```bash
ssh <tp-username>@gpu-gw.enst.fr
sinteractive
```

---

## 2. Cloner le repo sur le cluster

```bash
git clone https://github.com/leoivars31/Data-Challenge-Telecom-Paris-Face-Occlusion.git
cd Data-Challenge-Telecom-Paris-Face-Occlusion
```

---

## 3. Transférer les données (images)

Depuis ta machine locale :

```bash
scp -r data/raw/Crop_224_5fp_100K <tp-username>@gpu-gw.enst.fr:~/Data-Challenge-Telecom-Paris-Face-Occlusion/data/raw/
scp data/raw/train.csv data/raw/test_students.csv <tp-username>@gpu-gw.enst.fr:~/Data-Challenge-Telecom-Paris-Face-Occlusion/data/raw/
```

---

## 4. Créer l'environnement conda sur le cluster

```bash
module load python/3.11 cuda/12.4
conda create -n face-occlusion python=3.11
conda activate face-occlusion
pip install -r requirements.txt
```

---

## 5. Lancer le notebook sur le cluster

**Sur le cluster** (dans la session `sinteractive`) :

```bash
module load python/3.11 cuda/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate face-occlusion
cd Data-Challenge-Telecom-Paris-Face-Occlusion
jupyter notebook --no-browser --port=8888
```

**Sur ta machine locale** (dans un autre terminal), ouvrir le tunnel SSH :

```bash
ssh -N -L 8888:localhost:8888 <tp-username>@gpu-gw.enst.fr
```

Puis ouvrir dans le navigateur :

```
http://localhost:8888
```

Naviguer vers `notebooks/DataChallengeExample.ipynb`.

---

## 6. Soumettre un job batch (Slurm)

Créer un fichier `job_script.sh` :

```bash
#!/bin/bash
#SBATCH --job-name=face-occlusion
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00

module purge
module load python/3.11 cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate face-occlusion

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi

python scripts/train.py

echo "Job finished at $(date)"
```

Soumettre :

```bash
sbatch job_script.sh
```

Suivre l'état :

```bash
squeue --me
```

---

## 6. Récupérer les prédictions

Depuis ta machine locale :

```bash
scp <tp-username>@gpu-gw.enst.fr:~/Data-Challenge-Telecom-Paris-Face-Occlusion/data/submissions/test_predictions.csv data/submissions/
```
