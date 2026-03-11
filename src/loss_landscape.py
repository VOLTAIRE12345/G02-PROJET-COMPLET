"""
loss_landscape.py — Groupe G02
Analyse du loss landscape 1D (méthode simplifiée CPU/GPU).
Calcul de la sharpness selon Keskar et al. (2017).
Filter-wise normalization selon Li et al. (2018).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR

import numpy as np
import torch


def evaluate_on_subset(model, loader, device, n_samples=64):
    """Calcule la loss moyenne sur n_samples exemples."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            bs      = batch["labels"].size(0)
            total_loss += outputs.loss.item() * bs
            n          += bs
            if n >= n_samples:
                break
    return total_loss / max(n, 1)


def compute_loss_landscape_1d(model, loader, device, n_points=15, epsilon=0.05):
    """
    Perturbation 1D du modèle autour de son point courant.
    Utilise la filter-wise normalization (Li et al., 2018).
    """
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]

    direction = []
    for p in model.parameters():
        d      = torch.randn_like(p)
        norm_p = p.data.norm() + 1e-8
        d      = d / d.norm() * norm_p
        direction.append(d)

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + alpha * d
        loss = evaluate_on_subset(model, loader, device, n_samples=64)
        losses.append(loss)

    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0.clone()

    return alphas.tolist(), losses


def compute_sharpness(model, loader, device, rho=0.05, n_directions=5):
    """
    Sharpness = moyenne sur plusieurs directions de
    |L(theta + eps) - L(theta)| avec ||eps|| <= rho.
    Un minima plat (sharpness faible) → meilleure généralisation.
    """
    base_loss       = evaluate_on_subset(model, loader, device, n_samples=128)
    original_params = [p.clone().detach() for p in model.parameters()]
    max_deltas      = []

    for _ in range(n_directions):
        direction  = [torch.randn_like(p) for p in model.parameters()]
        total_norm = sum(d.norm()**2 for d in direction).sqrt() + 1e-8
        direction  = [rho * d / total_norm for d in direction]

        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + d

        pert_loss = evaluate_on_subset(model, loader, device, n_samples=128)
        max_deltas.append(abs(pert_loss - base_loss))

        for p, p0 in zip(model.parameters(), original_params):
            p.data = p0.clone()

    return float(np.mean(max_deltas))


def analyze_configs(configs: list, loaders: dict, device):
    """
    Calcule loss landscape et sharpness pour chaque configuration entraînée.
    """
    results = {}

    for cfg in configs:
        label = cfg["label"]
        model = cfg["model"]
        print(f"\n  Landscape → {label}")

        alphas, losses = compute_loss_landscape_1d(
            model, loaders["validation"], device, n_points=15, epsilon=0.05
        )
        sharpness = compute_sharpness(
            model, loaders["validation"], device, rho=0.05, n_directions=5
        )

        val_f1    = max(cfg["history"]["val_f1"])
        val_acc   = max(cfg["history"]["val_acc"])
        train_acc = max(cfg["history"]["train_acc"])

        results[label] = {
            "alphas":             alphas,
            "losses":             losses,
            "sharpness":          sharpness,
            "val_f1":             val_f1,
            "val_acc":            val_acc,
            "train_acc":          train_acc,
            "generalization_gap": train_acc - val_acc,
        }
        print(f"    Sharpness = {sharpness:.5f} | Val F1 = {val_f1:.4f} | "
              f"Gap = {train_acc - val_acc:.4f}")

    path = os.path.join(RESULTS_DIR, "sharpness_results.json")
    with open(path, "w") as f:
        save = {k: {kk: vv for kk, vv in v.items() if kk not in ["alphas", "losses"]}
                for k, v in results.items()}
        json.dump(save, f, indent=2, default=float)
    print(f"\n  Sharpness sauvegardée : {path}")

    return results
