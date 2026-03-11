"""
main_experiment.py — Groupe G02
Pipeline complet :
  1. Exploration du dataset IMDb (D01)
  2. Optimisation Bayésienne Optuna — P02 (weight_decay × dropout)
  3. Entraînement des 5 configurations clés
  4. Analyse du loss landscape et sharpness
  5. Génération de toutes les figures (fig00 à fig09)
  6. Évaluation finale sur le test set
  7. Affichage du résumé final

Usage sur Google Colab :
    !git clone https://github.com/VOLTAIRE12345/G02-PROJET-COMPLET.git /content/G02_PROJET
    %cd /content/G02_PROJET
    !pip install -r requirements.txt -q
    !python src/main_experiment.py
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config         import RESULTS_DIR, FIGURES_DIR, SEED
from data_loader    import load_imdb_subset, get_dataloaders, explore_dataset, analyze_tokenization
from model_setup    import get_model_and_tokenizer, build_optimizer, get_device
from train_eval     import train_model, evaluate
from optimization   import run_optuna_study, _get_shared_data
from loss_landscape import analyze_configs
from visualization  import (
    plot_all_training_curves,
    plot_training_curves,
    plot_optuna_convergence,
    plot_regularization_heatmap,
    plot_wd_dropout_effect,
    plot_loss_landscape_1d,
    plot_sharpness_comparison,
    plot_sharpness_correlation,
    plot_confusion_matrix,
    plot_comparative_summary,
)

DEVICE = get_device()

print(f"\n{'='*65}")
print("  PROJET G02 — Fine-tuning BERT / IMDb — P02 Régularisation")
print(f"  Device  : {DEVICE}")
print(f"  Results : {RESULTS_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Exploration du dataset
# ═══════════════════════════════════════════════════════════════════════════════
print("── ÉTAPE 1 : Chargement et exploration IMDb ──")
subsets = load_imdb_subset(
    num_train_per_class=300,
    num_val_per_class=100,
    num_test_per_class=150,
    seed=SEED,
)
explore_dataset(subsets, max_length=200, save=True)   # → fig00
analyze_tokenization(subsets)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Optimisation Bayésienne (Optuna)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 2 : Optimisation Optuna (20 trials) ──")
study = run_optuna_study(n_trials=20)
best_params = study.best_params
print(f"  Best params : {best_params}")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement des 5 configurations clés
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 3 : Entraînement des 5 configurations ──")

CONFIGS_TO_COMPARE = {
    "Défaut (WD=0, Drop=0.1)":     {"weight_decay": 0.0,   "dropout_prob": 0.1,  "lr": 2e-5},
    "Fort WD (WD=1e-2, Drop=0.0)": {"weight_decay": 1e-2,  "dropout_prob": 0.0,  "lr": 2e-5},
    "Fort Drop (WD=0, Drop=0.3)":  {"weight_decay": 0.0,   "dropout_prob": 0.3,  "lr": 2e-5},
    "Combiné (WD=1e-3, Drop=0.1)": {"weight_decay": 1e-3,  "dropout_prob": 0.1,  "lr": 2e-5},
    "Optuna Best": {
        "weight_decay": best_params.get("weight_decay", 1e-4),
        "dropout_prob": best_params.get("dropout_prob", 0.1),
        "lr":           best_params.get("lr",           2e-5),
    },
}

trained_configs  = []
all_histories    = {}

for cfg_name, hp in CONFIGS_TO_COMPARE.items():
    print(f"\n  ── Config : {cfg_name} ──")
    model, tokenizer = get_model_and_tokenizer(
        dropout_prob=hp["dropout_prob"], device=DEVICE
    )
    loaders   = get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200)
    optimizer = build_optimizer(model, lr=hp["lr"], weight_decay=hp["weight_decay"])

    history = train_model(
        model, loaders, optimizer,
        num_epochs=3, warmup_ratio=0.1,
        device=DEVICE, patience=2, verbose=True,
    )

    all_histories[cfg_name] = history
    trained_configs.append({
        "label":   cfg_name,
        "model":   model,
        "history": history,
        "hp":      hp,
    })

# fig01 — Courbes d'entraînement toutes configs
plot_all_training_curves(all_histories)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Loss Landscape & Sharpness
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 4 : Analyse du loss landscape ──")

_, tokenizer = get_model_and_tokenizer(device=DEVICE)
loaders_landscape = get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200)

landscape_results = analyze_configs(trained_configs, loaders_landscape, DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Génération de toutes les figures
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 5 : Génération des figures ──")

# fig02 — Convergence Optuna + écart généralisation
plot_optuna_convergence(study)

# fig03 — Heatmap WD × Dropout
plot_regularization_heatmap(study)

# fig04 — Scatter plots effet WD et Dropout
plot_wd_dropout_effect(study)

# fig05 — Loss landscape 1D
landscape_plot_data = [
    {"label": label, "alphas": res["alphas"], "losses": res["losses"]}
    for label, res in landscape_results.items()
]
plot_loss_landscape_1d(landscape_plot_data)

# fig06 — Sharpness barplot
plot_sharpness_comparison(landscape_results)

# fig07 — Corrélation Sharpness vs F1 / Gap
plot_sharpness_correlation(landscape_results)

# fig09 — Tableau comparatif
plot_comparative_summary(landscape_results)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Évaluation finale sur le test set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 6 : Évaluation finale ──")

best_cfg      = max(trained_configs, key=lambda c: max(c["history"]["val_f1"]))
print(f"  Meilleure config : {best_cfg['label']}")

final_model   = best_cfg["model"]
loaders_final = get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200)
test_metrics  = evaluate(final_model, loaders_final["test"], DEVICE)

print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Test F1-score : {test_metrics['f1']:.4f}")
print(f"  Test Loss     : {test_metrics['loss']:.4f}")

# fig08 — Matrice de confusion
final_model.eval()
all_preds, all_labels_list = [], []
with torch.no_grad():
    for batch in loaders_final["test"]:
        batch   = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = final_model(**batch)
        preds   = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels_list.extend(batch["labels"].cpu().numpy())
plot_confusion_matrix(all_labels_list, all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Sauvegarde du résumé final + affichage
# ═══════════════════════════════════════════════════════════════════════════════
sharpness_dict = {label: res["sharpness"] for label, res in landscape_results.items()}
gap_dict       = {label: res["generalization_gap"] for label, res in landscape_results.items()}

final_summary = {
    "best_config":         best_cfg["label"],
    "best_hp":             best_cfg["hp"],
    "test_metrics":        test_metrics,
    "sharpness":           sharpness_dict,
    "generalization_gaps": gap_dict,
    "optuna_best_params":  best_params,
    "optuna_best_val_f1":  study.best_value,
    "n_trials":            len(study.trials),
    "best_val_f1":         study.best_value,
    "best_params":         best_params,
}
summary_path = os.path.join(RESULTS_DIR, "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_summary, f, indent=2, default=float)

# ─── Résumé lisible ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("  RÉSUMÉ FINAL")
print(f"{'='*65}")
print(f"\nMeilleur trial Optuna :")
for k, v in final_summary.items():
    if k not in ["sharpness", "generalization_gaps"]:
        print(f"  {k:30s} : {v}")

print(f"\nSharpness par configuration :")
for cfg, s in sharpness_dict.items():
    print(f"  {cfg:38s} : {s:.5f}")

print(f"\nGap de généralisation par configuration :")
for cfg, g in gap_dict.items():
    print(f"  {cfg:38s} : {g:.5f}")

print(f"\n{'─'*65}")
print(f"  Meilleure config  : {best_cfg['label']}")
print(f"  Test Accuracy     : {test_metrics['accuracy']:.4f}")
print(f"  Test F1-score     : {test_metrics['f1']:.4f}")
print(f"\n  Meilleurs hyperparamètres Optuna :")
for k, v in best_params.items():
    print(f"    {k:20s} : {v}")

print(f"\n{'='*65}")
print("  EXPÉRIENCE TERMINÉE")
print(f"  Figures   → {FIGURES_DIR}")
print(f"  Résultats → {RESULTS_DIR}")
print(f"  Résumé    → {summary_path}")
print(f"{'='*65}")

# ─── Liste des figures générées ───────────────────────────────────────────────
print("\nFigures générées :")
for f in sorted(os.listdir(FIGURES_DIR)):
    if f.endswith(".png"):
        print(f"  ✓ {f}")
