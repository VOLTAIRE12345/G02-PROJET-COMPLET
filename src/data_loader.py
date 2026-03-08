"""
data_loader.py — Groupe G02
Dataset : IMDb (D01) | Modèle : BERT-base (M02) | Problématique : P02 | Méthode : Optuna
Chargement et sous-échantillonnage équilibré pour CPU et GPU Colab.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


# ─── 1. Chargement et sous-échantillonnage ────────────────────────────────────
def load_imdb_subset(num_train_per_class=300, num_val_per_class=100,
                     num_test_per_class=150, seed=SEED):
    """
    Charge IMDb depuis HuggingFace et retourne un sous-ensemble équilibré.

    Args:
        num_train_per_class : Exemples par classe pour l'entraînement.
        num_val_per_class   : Exemples par classe pour la validation.
        num_test_per_class  : Exemples par classe pour le test.
        seed                : Reproductibilité.

    Returns:
        dict avec clés 'train', 'validation', 'test'
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

    # IMDb n'a pas de split validation — on découpe train en 85/15
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

    for split, data in subsets.items():
        n_pos = sum(1 for d in data if d["label"] == 1)
        n_neg = sum(1 for d in data if d["label"] == 0)
        print(f"  {split:>12s}: {len(data):4d} exemples  (pos={n_pos}, neg={n_neg})")

    return subsets


# ─── 2. Dataset PyTorch ───────────────────────────────────────────────────────
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


# ─── 3. DataLoaders ──────────────────────────────────────────────────────────
def get_dataloaders(subsets, tokenizer, batch_size=16, max_length=200):
    """
    Crée les DataLoaders pour train, validation et test.

    Args:
        subsets    : dict retourné par load_imdb_subset
        tokenizer  : tokenizer HuggingFace
        batch_size : taille des batchs
        max_length : troncature des séquences

    Returns:
        dict de DataLoaders
    """
    loaders = {}
    for split, examples in subsets.items():
        ds = IMDbDataset(examples, tokenizer, max_length)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,   # num_workers=0 obligatoire sur Colab
            pin_memory=False,
        )
    return loaders


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok  = AutoTokenizer.from_pretrained("bert-base-uncased")
    data = load_imdb_subset(50, 20, 30)
    loaders = get_dataloaders(data, tok, batch_size=8)
    batch = next(iter(loaders["train"]))
    print("input_ids :", batch["input_ids"].shape)
    print("labels    :", batch["labels"])
