"""
optimization.py — Groupe G02
Optimisation bayésienne des hyperparamètres P02 (weight_decay + dropout) via Optuna.
Compatible CPU et GPU Colab.
"""
import sys, os, json, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, SEED

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner

from data_loader  import load_imdb_subset, get_dataloaders
from model_setup  import get_model_and_tokenizer, build_optimizer, get_device
from train_eval   import train_model, evaluate

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEVICE = get_device()


# ─── Espace de recherche P02 ──────────────────────────────────────────────────
# Problématique : Comment weight_decay et dropout affectent la généralisation ?
SEARCH_SPACE = {
    "weight_decay": ("log_float", 1e-5, 1e-2),   # Régularisation L2
    "dropout_prob": ("float",     0.0,  0.3),     # Régularisation par dropout
    "lr":           ("log_float", 1e-6, 5e-4),    # Learning rate
    "batch_size":   ("categorical", [8, 16]),      # Taille de batch
    "warmup_ratio": ("float",     0.0,  0.15),    # Warmup scheduler
    "num_epochs":   ("int",       2,    4),        # Nombre d'époques
}


# ─── Données partagées entre les trials ──────────────────────────────────────
_shared_data = {}

def _get_shared_data(num_train=300, num_val=100):
    """Charge les données une seule fois pour tous les trials Optuna."""
    global _shared_data
    if not _shared_data:
        _shared_data["subsets"] = load_imdb_subset(
            num_train_per_class=num_train,
            num_val_per_class=num_val,
            seed=SEED,
        )
    return _shared_data["subsets"]


# ─── Fonction objectif Optuna ─────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    """
    Un trial = une configuration (weight_decay, dropout, lr, batch_size…).
    Retourne le Val F1-score (à maximiser).
    """
    # Suggestion des hyperparamètres P02
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout_prob = trial.suggest_float("dropout_prob", 0.0, 0.3)
    lr           = trial.suggest_float("lr",           1e-6, 5e-4, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)
    num_epochs   = trial.suggest_int("num_epochs", 2, 4)

    # Données
    subsets = _get_shared_data()

    # Modèle avec dropout variable (hyperparamètre P02)
    model, tokenizer = get_model_and_tokenizer(dropout_prob=dropout_prob, device=DEVICE)
    loaders          = get_dataloaders(subsets, tokenizer,
                                       batch_size=batch_size, max_length=200)
    optimizer        = build_optimizer(model, lr=lr, weight_decay=weight_decay)

    # Entraînement
    history = train_model(
        model, loaders, optimizer,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        device=DEVICE,
        patience=2,
        verbose=False,
    )

    val_f1       = max(history["val_f1"])
    val_acc      = max(history["val_acc"])
    train_acc    = max(history["train_acc"])
    gap          = train_acc - val_acc

    # Sauvegarde du trial
    trial_info = {
        "trial":        trial.number,
        "weight_decay": weight_decay,
        "dropout":      dropout_prob,
        "lr":           lr,
        "batch_size":   batch_size,
        "val_f1":       val_f1,
        "val_accuracy": val_acc,
        "train_accuracy": train_acc,
        "gap":          gap,
    }
    path = os.path.join(RESULTS_DIR, f"trial_{trial.number:03d}.json")
    with open(path, "w") as f:
        json.dump(trial_info, f, indent=2, default=float)

    print(f"  [Trial {trial.number:3d}] "
          f"wd={weight_decay:.1e} | drop={dropout_prob:.2f} | "
          f"lr={lr:.1e} | val_f1={val_f1:.4f} | val_acc={val_acc:.4f} | gap={gap:.4f}")

    # Pruning des trials peu performants
    trial.report(val_f1, step=len(history["val_f1"]))
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_f1


# ─── Lancer l'étude Optuna ────────────────────────────────────────────────────
def run_optuna_study(n_trials=20):
    """
    Lance l'optimisation bayésienne et sauvegarde les résultats.

    Args:
        n_trials : Nombre de trials à exécuter.

    Returns:
        study : Objet optuna.Study avec les résultats.
    """
    print(f"\n{'='*60}")
    print(f"  Optuna — P02 Régularisation ({n_trials} trials)")
    print(f"  Device : {DEVICE}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name="G02_BERT_IMDb_P02",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Résumé
    bt = study.best_trial
    print(f"\n{'='*60}")
    print(f"  Meilleur trial : #{bt.number}")
    print(f"  Val F1         : {bt.value:.4f}")
    for k, v in bt.params.items():
        print(f"  {k:20s} : {v}")
    print(f"{'='*60}\n")

    # Sauvegarde du meilleur résultat
    best_summary = {
        "best_trial":       bt.number,
        "best_val_f1":      bt.value,
        "best_params":      bt.params,
        "n_trials":         len(study.trials),
        "n_pruned":         sum(1 for t in study.trials
                                if t.state == optuna.trial.TrialState.PRUNED),
    }
    path = os.path.join(RESULTS_DIR, "optuna_best.json")
    with open(path, "w") as f:
        json.dump(best_summary, f, indent=2, default=float)
    print(f"Résultats Optuna sauvegardés : {path}")

    return study


if __name__ == "__main__":
    study = run_optuna_study(n_trials=5)
    print("Meilleurs paramètres :", study.best_params)
