"""
visualization.py — Groupe G02
Toutes les figures du rapport : courbes d'entraînement, heatmap régularisation,
loss landscape, sharpness, matrice de confusion, écart généralisation.
Compatible Colab (chemins depuis config.py).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES_DIR

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})


# ─── 1. Courbes d'entraînement ────────────────────────────────────────────────
def plot_training_curves(history: dict, title="Courbes d'entraînement",
                          save_name="fig1_training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train")
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Validation")
    axes[0].set_title("Loss par époque")
    axes[0].set_xlabel("Époque"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "b-o",  label="Train Acc")
    axes[1].plot(epochs, history["val_acc"],   "r-o",  label="Val Acc")
    axes[1].plot(epochs, history["val_f1"],    "g--s", label="Val F1")
    axes[1].set_title("Accuracy & F1 par époque")
    axes[1].set_xlabel("Époque"); axes[1].set_ylabel("Score")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 2. Heatmap weight_decay × dropout ───────────────────────────────────────
def plot_regularization_heatmap(results_grid: dict,
                                 save_name="fig3_regularization_heatmap.png",
                                 save=True):
    """
    results_grid : {(wd, dp): val_f1}  OU  liste de dicts Optuna trials
    """
    # Normaliser l'entrée si c'est une liste de dicts
    if isinstance(results_grid, list):
        grid = {}
        for r in results_grid:
            wd = round(r.get("weight_decay", 0), 7)
            dp = round(r.get("dropout", r.get("dropout_prob", 0)), 2)
            v  = r.get("val_f1", r.get("val_accuracy", 0))
            if (wd, dp) not in grid or grid[(wd, dp)] < v:
                grid[(wd, dp)] = v
        results_grid = grid

    wds    = sorted(set(k[0] for k in results_grid))
    dps    = sorted(set(k[1] for k in results_grid))
    matrix = np.array([[results_grid.get((wd, dp), np.nan)
                        for dp in dps] for wd in wds])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(matrix) * 0.97, vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(dps))); ax.set_yticks(range(len(wds)))
    ax.set_xticklabels([f"{d:.2f}" for d in dps])
    ax.set_yticklabels([f"{w:.0e}" for w in wds])
    ax.set_xlabel("Dropout probability"); ax.set_ylabel("Weight decay")
    ax.set_title("Val F1-score : Weight Decay × Dropout\n(G02 — P02 Régularisation)")
    plt.colorbar(im, ax=ax, label="F1-score")

    for i in range(len(wds)):
        for j in range(len(dps)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    if save:
        path = os.path.join(FIGURES_DIR, save_name)
        fig.savefig(path, bbox_inches="tight"); plt.close(fig)
        print(f"  Figure : {path}")
        return path
    else:
        plt.show()


# ─── 3. Loss Landscape 1D ─────────────────────────────────────────────────────
def plot_loss_landscape_1d(configs: list, save_name="fig5_loss_landscape_1d.png"):
    """configs : liste de dict avec 'label', 'alphas', 'losses'"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(configs)))

    for cfg, color in zip(configs, colors):
        alphas = np.array(cfg["alphas"])
        losses = np.array(cfg["losses"])
        ax.plot(alphas, losses, label=cfg["label"], color=color, linewidth=2)
        idx_min = np.argmin(losses)
        ax.scatter(alphas[idx_min], losses[idx_min], color=color, s=100, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="θ courant")
    ax.set_xlabel("Direction de perturbation α")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Landscape 1D — Comparaison configurations\n(BERT-base / IMDb — G02 P02)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 4. Sharpness bar chart ───────────────────────────────────────────────────
def plot_sharpness_comparison(sharpness_dict: dict,
                               save_name="fig6_sharpness_comparison.png"):
    labels = list(sharpness_dict.keys())
    values = list(sharpness_dict.values())
    colors = ["#2196F3" if v == min(values) else "#FF5722" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Sharpness")
    ax.set_title("Sharpness des minima par configuration\n(bleu = minima le plus plat → meilleure généralisation)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 5. Corrélation Sharpness / Généralisation ───────────────────────────────
def plot_sharpness_correlation(landscape_results: dict,
                                save_name="fig7_sharpness_correlation.png"):
    labels    = list(landscape_results.keys())
    sharpness = [landscape_results[l]["sharpness"]           for l in labels]
    gaps      = [landscape_results[l]["generalization_gap"]  for l in labels]
    f1s       = [landscape_results[l]["val_f1"]              for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Corrélation Sharpness — Généralisation (G02 P02)", fontweight="bold")

    axes[0].scatter(sharpness, f1s, s=100, c="steelblue", edgecolors="black", zorder=3)
    for i, lbl in enumerate(labels):
        axes[0].annotate(lbl, (sharpness[i], f1s[i]),
                         textcoords="offset points", xytext=(5, 4), fontsize=7)
    axes[0].set_xlabel("Sharpness"); axes[0].set_ylabel("Val F1")
    axes[0].set_title(f"Sharpness vs Val F1"); axes[0].grid(True, alpha=0.3)

    axes[1].scatter(sharpness, gaps, s=100, c="salmon", edgecolors="black", zorder=3)
    for i, lbl in enumerate(labels):
        axes[1].annotate(lbl, (sharpness[i], gaps[i]),
                         textcoords="offset points", xytext=(5, 4), fontsize=7)
    axes[1].set_xlabel("Sharpness"); axes[1].set_ylabel("Gap (train − val)")
    axes[1].set_title(f"Sharpness vs Généralisation Gap"); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 6. Historique Optuna ─────────────────────────────────────────────────────
def plot_optuna_history(study, save_name="fig2_optuna_history.png"):
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    values        = [t.value  for t in study.trials if t.value is not None]
    best_so_far   = np.maximum.accumulate(values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Optimisation Bayésienne — Optuna G02 P02", fontsize=14, fontweight="bold")

    axes[0].plot(trial_numbers, values, "o", color="steelblue", alpha=0.6, label="Trial")
    axes[0].plot(trial_numbers, best_so_far, "-", color="red", linewidth=2, label="Best so far")
    axes[0].set_xlabel("Numéro du trial"); axes[0].set_ylabel("Val F1-score")
    axes[0].set_title("Convergence Optuna"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    try:
        import optuna
        importance = optuna.importance.get_param_importances(study)
        names = list(importance.keys()); vals = list(importance.values())
        axes[1].barh(names, vals, color="teal", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Importance relative")
        axes[1].set_title("Importance des hyperparamètres\n(FanovaImportanceEvaluator)")
        axes[1].grid(True, axis="x", alpha=0.3)
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Non disponible\n({e})",
                     ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 7. Matrice de confusion ──────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names=("Négatif", "Positif"),
                           save_name="fig8_confusion_matrix.png"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion — Test set")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── 8. Écart généralisation ──────────────────────────────────────────────────
def plot_generalization_gap(configs_gap: dict, save_name="fig4_generalization_gap.png"):
    """configs_gap : {label: {'train_acc': float, 'val_acc': float}}"""
    labels     = list(configs_gap.keys())
    train_accs = [configs_gap[l]["train_acc"] for l in labels]
    val_accs   = [configs_gap[l]["val_acc"]   for l in labels]
    gaps       = [tr - va for tr, va in zip(train_accs, val_accs)]

    x     = np.arange(len(labels)); width = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, train_accs, width, label="Train Acc", color="#42A5F5")
    ax.bar(x + width/2, val_accs,   width, label="Val Acc",   color="#EF5350")
    ax2 = ax.twinx()
    ax2.plot(x, gaps, "ko--", linewidth=1.5, markersize=7, label="Gap")
    ax2.set_ylabel("Gap (train − val)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Écart train/val par configuration de régularisation\n(G02 — P02)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3); plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Figure : {path}")
    return path


if __name__ == "__main__":
    print("Module visualization chargé — test avec données synthétiques...")
    h = {"train_loss":[0.65,0.42,0.30], "val_loss":[0.60,0.48,0.45],
         "train_acc":[0.63,0.80,0.88],  "val_acc":[0.68,0.78,0.80],
         "val_f1":[0.67,0.77,0.79]}
    plot_training_curves(h, title="Test", save_name="test_curves.png")
    print("OK")
