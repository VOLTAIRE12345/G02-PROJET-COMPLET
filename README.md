# Projet G02 — Fine-tuning BERT / IMDb — P02 Régularisation
## Compatible Google Colab ✅ | GPU T4 recommandé

**Groupe**        : G02  
**Dataset**       : D01 — IMDb Reviews (50k, binaire pos/nég)  
**Modèle**        : M02 — BERT-base-uncased (109.5M paramètres)  
**Problématique** : P02 — Régularisation et Généralisation  
**Méthode**       : Optuna (TPE Sampler, 20 trials)  
**Métrique**      : Accuracy & F1-score  
**Date limite**   : 13 mars 2026  

---

## Structure du projet

```
G02_PROJET_COMPLET/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py             ← Chemins centralisés (Colab + local)
│   ├── data_loader.py        ← Chargement IMDb + exploration + tokenisation
│   ├── model_setup.py        ← BERT-base + optimiseur AdamW
│   ├── train_eval.py         ← Boucle entraînement + early stopping
│   ├── optimization.py       ← Optuna P02 (weight_decay × dropout)
│   ├── loss_landscape.py     ← Loss landscape 1D + Sharpness
│   ├── visualization.py      ← Toutes les figures (fig00 à fig09)
│   └── main_experiment.py    ← Pipeline complet
├── figures/                  ← Figures générées automatiquement
└── results/                  ← JSON des résultats Optuna
```

---

## Figures générées

| Figure | Description |
|--------|-------------|
| fig00_length_distribution.png | Distribution des longueurs IMDb |
| fig01_all_training_curves.png | Courbes loss/accuracy des 5 configs |
| fig02_optuna_convergence.png | Convergence Optuna + écart généralisation |
| fig03_regularization_heatmap.png | Heatmap Weight Decay × Dropout |
| fig04_wd_dropout_effect.png | Scatter plots effet individuel WD et Dropout |
| fig05_loss_landscape_1d.png | Loss landscape 1D comparatif |
| fig06_sharpness_comparison.png | Sharpness des minima (barplot) |
| fig07_sharpness_correlation.png | Corrélation Sharpness vs F1/Gap |
| fig08_confusion_matrix.png | Matrice de confusion — test set |
| fig09_comparative_summary.png | Tableau comparatif des 5 configurations |

---

## Utilisation sur Google Colab

### Étape 1 — Cloner le dépôt
```python
!git clone https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET.git /content/PROJET G02 COMPLET
%cd /content/PROJET G02 COMPLET
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

### Étape 5 — Récupérer les figures
```python
import os
print("Figures générées :")
for f in sorted(os.listdir('/content/PROJET G02 COMPLET/figures/')):
    print(f"  {f}")
```

---

## Durée estimée
- GPU Colab T4 : **10–20 minutes**
- CPU (Intel i5, 8 Go RAM) : **60–120 minutes**

---

## Lien dépôt
- https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET
