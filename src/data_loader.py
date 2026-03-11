"""
data_loader.py — Groupe G02
Dataset : IMDb (D01) | Modèle : BERT-base (M02) | Problématique : P02 | Méthode : Optuna
Chargement, sous-échantillonnage équilibré et analyse exploratoire.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED, FIGURES_DIR

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


# ─── 1. Chargement et sous-échantillonnage ────────────────────────────────────
def load_imdb_subset(num_train_per_class=300, num_val_per_class=100,
                     num_test_per_class=150, seed=SEED, verbose=True):
    """
    Charge IMDb depuis HuggingFace et retourne un sous-ensemble équilibré.
    """
    print("Chargement du dataset IMDb...")
    raw = load_dataset("imdb")
    rng = np.random.default_rng(seed)

    def _sample(split_data, n_per_class):
        pos = [ex for ex in split_data if ex["label"] == 1]
        neg = [ex for ex in split_data if ex["label"] == 0]
        n   = min(n_per_class, len(pos), len(neg))
        idx_pos = rng.choice(len(pos), n, replace=False).tolist()
        idx_neg = rng.choice(len(neg), n, replace=False).tolist()
        subset  = [pos[i] for i in idx_pos] + [neg[i] for i in idx_neg]
        rng.shuffle(subset)
        return subset

    train_full = list(raw["train"])
    rng.shuffle(train_full)
    pivot        = int(0.85 * len(train_full))
    raw_train    = train_full[:pivot]
    raw_val_pool = train_full[pivot:]

    subsets = {
        "train":      _sample(raw_train,        num_train_per_class),
        "validation": _sample(raw_val_pool,      num_val_per_class),
        "test":       _sample(list(raw["test"]), num_test_per_class),
    }

    if verbose:
        for split, data in subsets.items():
            n_pos = sum(1 for d in data if d["label"] == 1)
            n_neg = sum(1 for d in data if d["label"] == 0)
            print(f"  {split:>12s}: {len(data):4d} exemples  (pos={n_pos}, neg={n_neg})")

    return subsets


# ─── 2. Analyse exploratoire ──────────────────────────────────────────────────
def explore_dataset(subsets, max_length=200, save=True):
    """
    Affiche les statistiques de longueur et génère la figure de distribution.
    """
    print("\nTaille des splits :")
    stats = {}
    for split, data in subsets.items():
        lengths = [len(ex["text"].split()) for ex in data]
        stats[split] = lengths
        print(f"  {split:>12s} | {len(data):4d} ex. "
              f"| longueur moy. : {np.mean(lengths):.0f} mots "
              f"| max : {np.max(lengths)} mots")

    # Troncature
    train_lengths = stats["train"]
    pct_intact = sum(1 for l in train_lengths if l <= max_length) / len(train_lengths) * 100
    print(f"\nTroncature à {max_length} tokens : "
          f"conserve environ {pct_intact:.0f}% des exemples d'entraînement intacts.")

    # Figure distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = {"train": "#4C9BE8", "validation": "#FFA54F", "test": "#4CAF50"}
    for split, lengths in stats.items():
        ax.hist(lengths, bins=40, alpha=0.6, label=split,
                color=colors[split], edgecolor="none")
    ax.axvline(max_length, color="red", linestyle="--",
               linewidth=2, label=f"max_length={max_length}")
    ax.set_xlabel("Longueur (mots)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des longueurs — IMDb")
    ax.legend()
    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, "fig00_length_distribution.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure : {path}")
    else:
        plt.show()

    return stats


# ─── 3. Analyse tokenisation ──────────────────────────────────────────────────
def analyze_tokenization(subsets, tokenizer_name="bert-base-uncased", n_samples=5):
    """
    Compare le nombre de mots vs tokens BERT pour illustrer le ratio de tokenisation.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"\nComparaison mots / tokens BERT :")
    print(f"  {'Mots':>6}  {'Tokens':>6}  {'Ratio':>6}  Texte")
    print("-" * 70)

    for ex in subsets["train"][:n_samples]:
        text   = ex["text"]
        words  = len(text.split())
        tokens = len(tok.encode(text, add_special_tokens=False))
        ratio  = tokens / max(words, 1)
        print(f"  {words:>6}  {tokens:>6}  {ratio:>6.2f}  {text[:60]}...")


# ─── 4. Dataset PyTorch ───────────────────────────────────────────────────────
class IMDbDataset(Dataset):
    """Dataset PyTorch pour IMDb — tokenise à la construction."""

    def __init__(self, examples, tokenizer, max_length=200):
        self.labels = [ex["label"] for ex in examples]
        texts = [ex["text"] for ex in examples]
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─── 5. DataLoaders ──────────────────────────────────────────────────────────
def get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200):
    """Crée les DataLoaders pour train, validation et test."""
    loaders = {}
    for split, examples in subsets.items():
        ds = IMDbDataset(examples, tokenizer, max_length)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
    return loaders


if __name__ == "__main__":
    data = load_imdb_subset(50, 20, 30)
    explore_dataset(data)
    analyze_tokenization(data)
