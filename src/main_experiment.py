"""
main_experiment.py — Groupe G02
Pipeline complet :
  1. Chargement données IMDb (D01)
  2. Optimisation Optuna P02 (weight_decay × dropout)
  3. Entraînement des 5 configurations clés
  4. Analyse du loss landscape et sharpness
  5. Génération de toutes les figures
  6. Évaluation finale sur le test set

Usage :
    python src/main_experiment.py

Sur Google Colab :
    !python src/main_experiment.py
"""
import sys, os, json, copy
import numpy as np
import torch

# ─── Ajout de src/ au path (Colab + local) ───────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config        import RESULTS_DIR, FIGURES_DIR, SEED
from data_loader   import load_imdb_subset, get_dataloaders
from model_setup   import get_model_and_tokenizer, build_optimizer, get_device
from train_eval    import train_model, evaluate
from optimization  import run_optuna_study, _get_shared_data
from loss_landscape import analyze_configs
from visualization  import (plot_training_curves, plot_regularization_heatmap,
                              plot_loss_landscape_1d, plot_sharpness_comparison,
                              plot_sharpness_correlation, plot_optuna_history,
                              plot_confusion_matrix, plot_generalization_gap)

DEVICE = get_device()

print(f"\n{'='*65}")
print("  PROJET G02 — Fine-tuning BERT / IMDb — P02 Régularisation")
print(f"  Device  : {DEVICE}")
print(f"  Results : {RESULTS_DIR}")
print(f"  Figures : {FIGURES_DIR}")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Chargement des données
# ═══════════════════════════════════════════════════════════════════════════════
print("── ÉTAPE 1 : Chargement des données IMDb ──")
subsets = load_imdb_subset(
    num_train_per_class=300,
    num_val_per_class=100,
    num_test_per_class=150,
    seed=SEED,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Optimisation Bayésienne (Optuna)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 2 : Optimisation Optuna (20 trials) ──")
study = run_optuna_study(n_trials=20)
best_params = study.best_params
print(f"  Best params : {best_params}")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Entraînement des configurations clés (P02)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 3 : Entraînement des 5 configurations ──")

# 5 configurations représentatives de l'espace P02
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

trained_configs = []

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

    # Courbe individuelle
    safe_name = cfg_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    plot_training_curves(
        history,
        title=f"Entraînement : {cfg_name}",
        save_name=f"fig1_training_{safe_name}.png",
    )

    trained_configs.append({
        "label":   cfg_name,
        "model":   model,
        "history": history,
        "hp":      hp,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Loss Landscape & Sharpness
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 4 : Analyse du loss landscape ──")

# Recharger un tokenizer commun pour les loaders de la phase landscape
_, tokenizer = get_model_and_tokenizer(device=DEVICE)
loaders_landscape = get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200)

landscape_results = analyze_configs(trained_configs, loaders_landscape, DEVICE)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Génération des figures de synthèse
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 5 : Génération des figures ──")

# 5.1 Convergence Optuna
plot_optuna_history(study)

# 5.2 Loss landscape 1D comparatif
landscape_plot_data = [
    {"label":  label,
     "alphas": np.array(res["alphas"]),
     "losses": np.array(res["losses"])}
    for label, res in landscape_results.items()
]
plot_loss_landscape_1d(landscape_plot_data)

# 5.3 Sharpness bar chart
sharpness_dict = {label: res["sharpness"] for label, res in landscape_results.items()}
plot_sharpness_comparison(sharpness_dict)

# 5.4 Corrélation Sharpness / Généralisation
plot_sharpness_correlation(landscape_results)

# 5.5 Écart généralisation
gap_dict = {
    label: {"train_acc": res["train_acc"], "val_acc": res["val_acc"]}
    for label, res in landscape_results.items()
}
plot_generalization_gap(gap_dict)

# 5.6 Heatmap régularisation (depuis les trials Optuna)
try:
    grid = {}
    for t in study.trials:
        if t.value is not None:
            wd = round(t.params.get("weight_decay", 1e-4), 7)
            dp = round(t.params.get("dropout_prob", 0.1), 2)
            if (wd, dp) not in grid or grid[(wd, dp)] < t.value:
                grid[(wd, dp)] = t.value
    if len(grid) >= 4:
        plot_regularization_heatmap(grid)
    else:
        print("  Heatmap : pas assez de points distincts.")
except Exception as e:
    print(f"  Heatmap non disponible : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Évaluation finale sur le test set
# ═══════════════════════════════════════════════════════════════════════════════
print("\n── ÉTAPE 6 : Évaluation finale ──")

# Sélectionner la meilleure configuration (val F1)
best_cfg = max(trained_configs, key=lambda c: max(c["history"]["val_f1"]))
print(f"  Meilleure config : {best_cfg['label']}")

final_model = best_cfg["model"]
loaders_final = get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200)
test_metrics  = evaluate(final_model, loaders_final["test"], DEVICE)

print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Test F1-score : {test_metrics['f1']:.4f}")
print(f"  Test Loss     : {test_metrics['loss']:.4f}")

# Matrice de confusion
final_model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in loaders_final["test"]:
        batch   = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = final_model(**batch)
        preds   = outputs.logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
plot_confusion_matrix(all_labels, all_preds)


# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Sauvegarde du résumé final
# ═══════════════════════════════════════════════════════════════════════════════
final_summary = {
    "best_config":          best_cfg["label"],
    "best_hp":              best_cfg["hp"],
    "test_metrics":         test_metrics,
    "sharpness":            sharpness_dict,
    "generalization_gaps":  {k: v["train_acc"] - v["val_acc"] for k, v in gap_dict.items()},
    "optuna_best_params":   best_params,
    "optuna_best_val_f1":   study.best_value,
}
summary_path = os.path.join(RESULTS_DIR, "final_summary.json")
with open(summary_path, "w") as f:
    json.dump(final_summary, f, indent=2, default=float)

print(f"\n{'='*65}")
print("  EXPÉRIENCE TERMINÉE")
print(f"  Figures   → {FIGURES_DIR}")
print(f"  Résultats → {RESULTS_DIR}")
print(f"  Résumé    → {summary_path}")
print(f"{'='*65}")
