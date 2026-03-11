"""
visualization.py — Groupe G02
Toutes les figures du rapport :
  fig00 - Distribution des longueurs
  fig01 - Courbes d'entraînement (loss + accuracy) par config
  fig02 - Historique convergence Optuna + écart généralisation
  fig03 - Heatmap Weight Decay × Dropout
  fig04 - Scatter plots effet WD et Dropout
  fig05 - Loss landscape 1D comparatif
  fig06 - Sharpness des minima (barplot)
  fig07 - Corrélation Sharpness vs F1 / Gap
  fig08 - Matrice de confusion
  fig09 - Tableau comparatif des 5 configurations (barplot)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES_DIR

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     150,
})

CONFIG_COLORS = {
    "Défaut (WD=0, Drop=0.1)":     "#E74C3C",
    "Fort WD (WD=1e-2, Drop=0.0)": "#3498DB",
    "Fort Drop (WD=0, Drop=0.3)":  "#2ECC71",
    "Combiné (WD=1e-3, Drop=0.1)": "#F39C12",
    "Optuna Best":                  "#9B59B6",
}


# ─── fig01 : Courbes d'entraînement (toutes configs sur un seul graphe) ───────
def plot_all_training_curves(all_histories: dict,
                              save_name="fig01_all_training_curves.png"):
    """
    all_histories : {cfg_name: history_dict}
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Courbes d'entraînement — 5 configurations\n(G02 — BERT-base / IMDb — P02)",
                 fontsize=14, fontweight="bold")

    metrics = [("train_loss", "val_loss",  "Loss",     "Loss par époque"),
               ("train_acc",  "val_acc",   "Accuracy", "Accuracy par époque"),
               (None,         "val_f1",    "F1-score", "Val F1-score par époque")]

    for ax, (train_key, val_key, ylabel, title) in zip(axes, metrics):
        for cfg_name, history in all_histories.items():
            color  = CONFIG_COLORS.get(cfg_name, "gray")
            epochs = range(1, len(history[val_key]) + 1)
            label_short = cfg_name.split("(")[0].strip()

            if train_key:
                ax.plot(epochs, history[train_key], "--", color=color, alpha=0.4, linewidth=1.2)
            ax.plot(epochs, history[val_key], "-o", color=color,
                    label=label_short, linewidth=2, markersize=5)

        ax.set_xlabel("Époque")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc="upper right",
                   title="— val  ·· train", title_fontsize=7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig01b : Courbe individuelle par config ──────────────────────────────────
def plot_training_curves(history: dict, title="Courbes d'entraînement",
                          save_name="fig01_training_curves.png"):
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
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig02 : Convergence Optuna + écart généralisation ────────────────────────
def plot_optuna_convergence(study, save_name="fig02_optuna_convergence.png"):
    """Convergence Optuna + écart généralisation par trial."""
    import json, os
    from config import RESULTS_DIR

    trials_data = []
    for t in study.trials:
        if t.value is not None:
            gap = None
            path = os.path.join(RESULTS_DIR, f"trial_{t.number:03d}.json")
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                    gap = d.get("gap", None)
            trials_data.append({"n": t.number, "f1": t.value, "gap": gap})

    numbers = [d["n"] for d in trials_data]
    f1s     = [d["f1"] for d in trials_data]
    gaps    = [d["gap"] for d in trials_data if d["gap"] is not None]
    gap_ns  = [d["n"]  for d in trials_data if d["gap"] is not None]
    best_so_far = np.maximum.accumulate(f1s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Optimisation Bayésienne — Optuna G02 P02", fontsize=14, fontweight="bold")

    axes[0].plot(numbers, f1s, "o", color="#3498DB", alpha=0.7, label="Trial", markersize=7)
    axes[0].plot(numbers, best_so_far, "-", color="#E74C3C", linewidth=2.5, label="Meilleur cumulatif")
    best_idx = int(np.argmax(f1s))
    axes[0].scatter(numbers[best_idx], f1s[best_idx], marker="*",
                    color="gold", s=300, zorder=5,
                    label=f"Best: Trial #{numbers[best_idx]} (F1={f1s[best_idx]:.4f})")
    axes[0].set_xlabel("Numéro du trial")
    axes[0].set_ylabel("Val F1-score")
    axes[0].set_title("Convergence")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(gap_ns, gaps, color="#F39C12", edgecolor="white", linewidth=0.5)
    axes[1].axhline(0.08, color="red", linestyle="--", linewidth=1.5,
                    label="Seuil sur-apprentissage")
    axes[1].set_xlabel("Numéro du trial")
    axes[1].set_ylabel("Gap (train - val)")
    axes[1].set_title("Ecart Généralisation par trial")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig03 : Heatmap Weight Decay × Dropout ───────────────────────────────────
def plot_regularization_heatmap(study, save_name="fig03_regularization_heatmap.png"):
    grid = {}
    for t in study.trials:
        if t.value is not None:
            wd = round(t.params.get("weight_decay", 1e-4), 7)
            dp = round(t.params.get("dropout_prob", 0.1), 2)
            if (wd, dp) not in grid or grid[(wd, dp)] < t.value:
                grid[(wd, dp)] = t.value

    if len(grid) < 4:
        print("  Heatmap : pas assez de points distincts.")
        return

    wds    = sorted(set(k[0] for k in grid))
    dps    = sorted(set(k[1] for k in grid))
    matrix = np.array([[grid.get((wd, dp), np.nan) for dp in dps] for wd in wds])

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                   vmin=np.nanmin(matrix) * 0.97, vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(dps)));  ax.set_xticklabels([f"{d:.2f}" for d in dps])
    ax.set_yticks(range(len(wds)));  ax.set_yticklabels([f"{w:.0e}" for w in wds])
    ax.set_xlabel("Dropout probability"); ax.set_ylabel("Weight decay")
    ax.set_title("Val F1-score : Weight Decay × Dropout\n(G02 — P02 Régularisation)")
    plt.colorbar(im, ax=ax, label="F1-score")

    for i in range(len(wds)):
        for j in range(len(dps)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if v > np.nanmax(matrix) * 0.92 else "black")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig04 : Scatter plots effet individuel WD et Dropout ─────────────────────
def plot_wd_dropout_effect(study, save_name="fig04_wd_dropout_effect.png"):
    data = [(t.params["weight_decay"], t.params["dropout_prob"], t.value)
            for t in study.trials if t.value is not None]
    wds  = np.array([d[0] for d in data])
    dps  = np.array([d[1] for d in data])
    f1s  = np.array([d[2] for d in data])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Effet du Weight Decay et du Dropout — G02 P02",
                 fontsize=14, fontweight="bold")

    sc1 = axes[0].scatter(wds, f1s, c=dps, cmap="coolwarm", s=80,
                           edgecolors="black", linewidths=0.5, zorder=3)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Weight Decay (log)")
    axes[0].set_ylabel("Val F1")
    axes[0].set_title("Val F1 vs Weight Decay")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0], label="Dropout")

    sc2 = axes[1].scatter(dps, f1s, c=np.log10(wds), cmap="viridis", s=80,
                           edgecolors="black", linewidths=0.5, zorder=3)
    axes[1].set_xlabel("Dropout")
    axes[1].set_ylabel("Val F1")
    axes[1].set_title("Val F1 vs Dropout")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[1], label="log10(Weight Decay)")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig05 : Loss Landscape 1D ────────────────────────────────────────────────
def plot_loss_landscape_1d(configs: list, save_name="fig05_loss_landscape_1d.png"):
    """configs : liste de dict avec 'label', 'alphas', 'losses'"""
    fig, ax = plt.subplots(figsize=(11, 5))

    for cfg in configs:
        color  = CONFIG_COLORS.get(cfg["label"], "gray")
        alphas = np.array(cfg["alphas"])
        losses = np.array(cfg["losses"])
        label_short = cfg["label"].split("(")[0].strip()
        ax.plot(alphas, losses, label=label_short, color=color, linewidth=2.2)
        idx_min = np.argmin(losses)
        ax.scatter(alphas[idx_min], losses[idx_min], color=color, s=80, zorder=5)

    ax.axvline(0, color="gray", linestyle="--", alpha=0.5, label="θ courant (α=0)")
    ax.set_xlabel("Direction de perturbation α")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Landscape 1D — Comparaison configurations\n"
                 "(BERT-base / IMDb — G02 P02 | Filter-wise normalization)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig06 : Sharpness barplot ────────────────────────────────────────────────
def plot_sharpness_comparison(landscape_results: dict,
                               save_name="fig06_sharpness_comparison.png"):
    labels = list(landscape_results.keys())
    values = [landscape_results[l]["sharpness"] for l in labels]
    min_v  = min(values)
    max_v  = max(values)

    colors = []
    for v in values:
        if v == min_v:
            colors.append("#3498DB")   # bleu = meilleure généralisation
        elif v == max_v:
            colors.append("#E74C3C")   # rouge = sur-apprentissage
        else:
            colors.append("#7F8C8D")   # gris = intermédiaire

    short_labels = [l.split("(")[0].strip() for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Comparaison Sharpness", fontsize=13, fontweight="bold")
    ax.set_title("Sharpness des minima par configuration\n"
                 "G02 : BERT-base / IMDb — P02 Régularisation",
                 fontsize=11)

    bars = ax.bar(short_labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#3498DB", label="Minima le plus plat → Meilleure généralisation"),
        Patch(facecolor="#E74C3C", label="Minima le plus pointu → Sur-apprentissage"),
        Patch(facecolor="#7F8C8D", label="Configurations intermédiaires"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.set_ylabel("Sharpness")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig07 : Corrélation Sharpness vs F1 / Gap ────────────────────────────────
def plot_sharpness_correlation(landscape_results: dict,
                                save_name="fig07_sharpness_correlation.png"):
    labels    = list(landscape_results.keys())
    sharpness = [landscape_results[l]["sharpness"]          for l in labels]
    gaps      = [landscape_results[l]["generalization_gap"] for l in labels]
    f1s       = [landscape_results[l]["val_f1"]             for l in labels]
    colors    = [CONFIG_COLORS.get(l, "gray") for l in labels]
    short     = [l.split("(")[0].strip() for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Corrélation Sharpness — Généralisation (G02 P02)",
                 fontsize=14, fontweight="bold")

    axes[0].scatter(sharpness, f1s, s=120, c=colors, edgecolors="black", zorder=3)
    for i, lbl in enumerate(short):
        axes[0].annotate(lbl, (sharpness[i], f1s[i]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[0].set_xlabel("Sharpness"); axes[0].set_ylabel("Val F1")
    axes[0].set_title("Sharpness vs Val F1\n(plus plat = meilleure perf ?)")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(sharpness, gaps, s=120, c=colors, edgecolors="black", zorder=3)
    for i, lbl in enumerate(short):
        axes[1].annotate(lbl, (sharpness[i], gaps[i]),
                         textcoords="offset points", xytext=(6, 4), fontsize=8)
    axes[1].set_xlabel("Sharpness"); axes[1].set_ylabel("Gap (train − val)")
    axes[1].set_title("Sharpness vs Gap de Généralisation\n(plus plat = moins de sur-apprentissage ?)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig08 : Matrice de confusion ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names=("Négatif", "Positif"),
                           save_name="fig08_confusion_matrix.png"):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
    ax.set_title("Matrice de Confusion — Test set\n"
                 f"(Accuracy={sum(np.diag(cm))/cm.sum():.3f})")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


# ─── fig09 : Tableau comparatif des 5 configurations ──────────────────────────
def plot_comparative_summary(landscape_results: dict,
                              save_name="fig09_comparative_summary.png"):
    """
    Barplot comparant Val F1, Gap et Sharpness normalisée pour les 5 configs.
    """
    labels    = list(landscape_results.keys())
    short     = [l.split("(")[0].strip() for l in labels]
    val_f1s   = [landscape_results[l]["val_f1"]             for l in labels]
    gaps      = [landscape_results[l]["generalization_gap"] for l in labels]
    sharpness = [landscape_results[l]["sharpness"]          for l in labels]

    # Normaliser sharpness pour la lisibilité
    s_norm = np.array(sharpness) / max(sharpness)

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width, val_f1s,  width, label="Val F1",              color="#9B59B6", alpha=0.85)
    ax.bar(x,         gaps,     width, label="Gap (train−val)",      color="#E74C3C", alpha=0.85)
    ax.bar(x + width, s_norm,   width, label="Sharpness (normalisée)", color="#3498DB", alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(short, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Tableau comparatif des 5 configurations — G02 P02\n"
                 "(Val F1 ↑ | Gap ↓ | Sharpness ↓ = meilleur)")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure : {path}")
    return path


if __name__ == "__main__":
    print("Module visualization chargé.")
