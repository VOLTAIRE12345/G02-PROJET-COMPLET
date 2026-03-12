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

L'exécution se fait en **3 étapes dans cet ordre** :

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

### Étape 2 — Installer les dépendances
```python
!pip install -r requirements.txt -q
```

### Étape 3 — Vérifier le GPU
```python
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### Étape 4 — Exécuter le notebook d'exploration (optionnel mais recommandé)
```python
# Ouvrir et exécuter notebooks/exploration.ipynb dans Colab
# Analyse exploratoire du dataset IMDb avant l'entraînement
```

### Étape 5 — Lancer l'expérience complète
```python
!python src/main_experiment.py
```

### Étape 6 — Analyser les résultats (notebook)
```python
# Ouvrir et exécuter notebooks/analysis.ipynb dans Colab
# Nécessite que main_experiment.py ait été exécuté (résultats dans results/)
```

### Étape 7 — Récupérer les figures
```python
import os
print("Figures générées :")
for f in sorted(os.listdir('/content/PROJET G02 COMPLET/figures/')):
    print(f"  {f}")
```
### Étape 8 — Télécharger les figures

Zipper et télécharger toutes les figures générées en une seule fois :

```python
# Zipper toutes les figures
import shutil
shutil.make_archive('/content/figures_G02', 'zip', '/content/PROJET G02 COMPLET/figures/')
print("✅ ZIP créé : /content/figures_G02.zip")

# Télécharger le ZIP
from google.colab import files
files.download('/content/figures_G02.zip')

---

## Résultats obtenus

| Configuration | Val F1 | Gap | Sharpness |
|---|---|---|---|
| Défaut (WD=0, Drop=0.1) | 0.843 | 0.102 | 0.60×10⁻⁵ |
| Fort WD (WD=1e-2, Drop=0.0) | **0.844** | 0.127 | 1.71×10⁻⁵ |
| Fort Drop (WD=0, Drop=0.3) | 0.791 | -0.013 | 1.29×10⁻⁵ |
| Combiné (WD=1e-3, Drop=0.1) | 0.804 | 0.113 | 1.87×10⁻⁵ |
| Optuna Best | 0.843 | 0.112 | 1.19×10⁻⁵ |

**Meilleure config (test set) : Fort WD → Accuracy = 87.67% | F1 = 87.46%**

---

## Durée estimée
- GPU Colab T4 : **10–20 minutes**
- CPU (Intel i5, 8 Go RAM) : **60–120 minutes**

---

## Rapport
Le rapport complet est disponible dans `report/main.pdf`.

---

## Lien vers le dépôt
- https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET
