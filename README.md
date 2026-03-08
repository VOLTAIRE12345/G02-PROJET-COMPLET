# Projet G02 — Fine-tuning BERT / IMDb — P02 Régularisation
## Compatible Google Colab ✅ | GPU T4 recommandé

**Groupe**        : G02  
**Dataset**       : D01 — IMDb Reviews (50k, binaire pos/nég)  
**Modèle**        : M02 — BERT-base-uncased (110M paramètres)  
**Problématique** : P02 — Régularisation et Généralisation  
**Méthode**       : Optuna (TPE Sampler, 20 trials)  
**Métrique**      : Accuracy & F1-score  
**Date limite**   : 13 mars 2026  

---

## Structure du projet

```
PROJET G02 COMPLET/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py             ← Chemins centralisés (Colab + local)
│   ├── data_loader.py        ← Chargement IMDb + sous-échantillonnage
│   ├── model_setup.py        ← BERT-base + optimiseur AdamW
│   ├── train_eval.py         ← Boucle entraînement + early stopping
│   ├── optimization.py       ← Optuna P02 (weight_decay × dropout)
│   ├── loss_landscape.py     ← Loss landscape 1D + Sharpness
│   ├── visualization.py      ← Toutes les figures du rapport
│   └── main_experiment.py    ← Pipeline complet (à exécuter en dernier)
├── notebooks/
│   ├── exploration.ipynb     ← Exploration du dataset IMDb
│   └── analysis.ipynb        ← Analyse interactive des résultats
├── figures/                  ← Figures générées automatiquement
├── results/                  ← JSON des résultats Optuna + grid search
└── report/                   ← Rapport PDF final
```

---

## Utilisation sur Google Colab

### Étape 1 — Cloner le dépôt GitHub
```python
!git clone https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET.git "/content/PROJET G02 COMPLET"
%cd "/content/PROJET G02 COMPLET"

```

### Étape 2 — Installer les dépendances
```python
!pip install -r requirements.txt -q
```

### Étape 3 — Vérifier le GPU
```python
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Étape 4 — Lancer l'expérience complète
```python
!python src/main_experiment.py
```

### Étape 5 — Ouvrir les notebooks
Ouvrir `notebooks/exploration.ipynb` puis `notebooks/analysis.ipynb`.

---

## Utilisation en local (CPU)

```bash
pip install -r requirements.txt
python src/main_experiment.py
```

Durée estimée :
- GPU Colab T4 : **8–15 minutes**
- CPU (Intel i5, 8 Go RAM) : **45–90 minutes**

---

## lien vers le dépôt 
- Dépôt : https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET
