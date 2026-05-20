# Utilisation du cluster GPU Télécom Paris

Documentation officielle : https://computing.telecom-paris.fr/

## Prérequis

- Être sur le réseau Télécom Paris (en salle) ou connecté via **VPN**
- Vérifier que ton compte a accès au cluster : https://computing.telecom-paris.fr/docs/getting-started/prerequisites/

En tant qu'étudiant, accès limité aux GPU **P100 et 3090** (max 4 GPU, max 36h par job).

---

## 1. Connexion SSH

```bash
ssh <login>@ssh.enst.fr
```

---

## 2. Cloner le repo sur le cluster

```bash
git clone https://github.com/<username>/Data-Challenge-Telecom-Paris-Face-Occlusion.git
cd Data-Challenge-Telecom-Paris-Face-Occlusion
```

---

## 3. Transférer les données (images)

Depuis ta machine locale :

```bash
scp -r data/raw/Crop_224_5fp_100K <login>@ssh.enst.fr:~/Data-Challenge-Telecom-Paris-Face-Occlusion/data/raw/
scp data/raw/train.csv data/raw/test_students.csv <login>@ssh.enst.fr:~/Data-Challenge-Telecom-Paris-Face-Occlusion/data/raw/
```

---

## 4. Créer l'environnement Python sur le cluster

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 5. Soumettre un job GPU interactif (Slurm)

```bash
srun --partition=default --gres=gpu:1 --time=36:00:00 --pty bash
```

Une fois dans le shell interactif :

```bash
source .venv/bin/activate
cd Data-Challenge-Telecom-Paris-Face-Occlusion
jupyter notebook --no-browser --port=8888
```

---

## 6. Accéder au notebook depuis ta machine locale

Depuis un autre terminal local, créer un tunnel SSH :

```bash
ssh -N -L 8888:localhost:8888 <login>@ssh.enst.fr
```

Puis ouvrir dans le navigateur :

```
http://localhost:8888
```

---

## 7. Soumettre un job batch (pour entraînement long)

Créer un script `train.sh` :

```bash
#!/bin/bash
#SBATCH --partition=default
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --job-name=face-occlusion
#SBATCH --output=logs/slurm_%j.out

source .venv/bin/activate
python scripts/train.py
```

Soumettre le job :

```bash
sbatch train.sh
```

Suivre l'état du job :

```bash
squeue --me
```
