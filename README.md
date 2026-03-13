# Projet G02 : Fine-tuning BERT / IMDb , P02 Régularisation
## Compatible Google Colab ✅ | GPU T4 recommandé

**Groupe**        : Projet G02: AZONFACK - FONKOUA - OGNIMBA 

**Dataset**       : D01 — IMDb Reviews (50k, binaire pos/nég)  
**Modèle**        : M02 — BERT-base-uncased (109.5M paramètres)  
**Problématique** : P02 — Régularisation et Généralisation  
**Méthode**       : Optuna (TPE Sampler, 20 trials)  
**Métrique**      : Accuracy & F1-score  


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
├── notebooks/
│   ├── exploration.ipynb     ← Analyse exploratoire du dataset (étape 1)
│   └── analysis.ipynb        ← Analyse des résultats post-expérience (étape 3)
├── report/
│   └── main.pdf              ← Rapport final (P02 Régularisation)
├── figures/                  ← Figures générées automatiquement
└── results/                  ← JSON des résultats Optuna
```

---

## Workflow complet recommandé

L'exécution se fera en **3 étapes dans cet ordre** :

```
┌─────────────────────────────────────────────────────────────┐
│  ÉTAPE 1 — notebooks/exploration.ipynb                      │
│  Analyse exploratoire : distribution, longueurs, tokens     │
│  → Permet de comprendre le dataset avant l'entraînement     │
├─────────────────────────────────────────────────────────────┤
│  ÉTAPE 2 — src/main_experiment.py                           │
│  Pipeline complet : Optuna + 5 configs + figures + résultats│
│  → Génère figures/ et results/                              │
├─────────────────────────────────────────────────────────────┤
│  ÉTAPE 3 — notebooks/analysis.ipynb                         │
│  Analyse des résultats : lecture JSON + visualisations      │
│  → Nécessite que main_experiment.py ait été exécuté         │
└─────────────────────────────────────────────────────────────┘
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
!git clone https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET.git "/content/PROJET G02 COMPLET"
import os
os.chdir("/content/PROJET G02 COMPLET")
```

### Étape 2 — Vérifier le GPU
```python
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Étape 3 — Exécution du notebook d'exploration (installation les dépendances + analyse du dataset)
```python
# Ouvrir et exécuter notebooks/exploration.ipynb dans Colab
# ✅ Les dépendances sont installées automatiquement par ce notebook
# 📊 Analyse exploratoire du dataset IMDb avant l'entraînement
```

### Étape 4 — Lancement de l'expérience complète
```python
!python src/main_experiment.py
```

### Étape 5 — Analyse des résultats (notebook)
```python
# Ouvrir et exécuter notebooks/analysis.ipynb dans Colab
# Nécessite que main_experiment.py ait été exécuté (résultats dans results/)
```

### Étape 6 — Récupération des figures
```python
import os
print("Figures générées :")
for f in sorted(os.listdir('/content/PROJET G02 COMPLET/figures/')):
    print(f"  {f}")
```
### Étape 7 — Téléchargement des figures

Zippons et téléchargeons toutes les figures générées en une seule fois :

```python
# Zipper toutes les figures
import shutil
shutil.make_archive('/content/figures_G02', 'zip', '/content/PROJET G02 COMPLET/figures/')
print("✅ ZIP créé : /content/figures_G02.zip")

# Télécharger le ZIP
from google.colab import files
files.download('/content/figures_G02.zip')
```

---

## Résultats obtenus

| Configuration | Val F1 | Gap | Sharpness |
|---|---|---|---|
| Défaut (WD=0, Drop=0.1) | 0.8269 | 0.1117 | 0.00001 |
| Fort WD (WD=1e-2, Drop=0.0) | 0.8544 | 0.1033 | 0.00001 |
| Fort Drop (WD=0, Drop=0.3) | 0.7097 | 0.0917 | 0.00003 |
| Combiné (WD=1e-3, Drop=0.1) | 0.8325 | 0.0783 | 0.00001 |
| Optuna Best | 0.8451 | 0.1133 | 0.00001 |

**Test Accuracy : 84,67 %  |  Test F1 : 84,87 %  |  Test Loss : 0,360
Meilleur val F1 Optuna : 86,83 %  (WD=7.82×10⁻⁴, Dropout=13,5 %, LR=4.60×10⁻⁵, 4 époques)
**

---

## Durée estimée
- GPU Colab T4 : ** 30 minutes**
- CPU (Intel i5, 8 Go RAM) : **60–120 minutes**

---

## Rapport
Le rapport complet est sera envoyé avec le dépôt par mail.

---

## Lien vers le dépôt
- https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET
