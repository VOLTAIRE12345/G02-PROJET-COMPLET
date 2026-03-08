"""
train_eval.py — Groupe G02
Boucle d'entraînement avec scheduler linéaire, gradient clipping et early stopping.
Compatible CPU et GPU Colab.
"""
import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


# ─── 1. Scheduler linéaire avec warmup ───────────────────────────────────────
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return max(0.0, float(num_training_steps - step) /
                   max(1, num_training_steps - num_warmup_steps))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ─── 2. Une époque d'entraînement ────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device, max_grad_norm=1.0):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        bs = batch["labels"].size(0)
        total_loss    += loss.item() * bs
        preds          = outputs.logits.argmax(dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total_samples += bs

    return total_loss / total_samples, total_correct / total_samples


# ─── 3. Évaluation ───────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """
    Évalue le modèle sur un DataLoader.

    Returns:
        dict : loss, accuracy, f1
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            bs      = batch["labels"].size(0)
            total_loss    += outputs.loss.item() * bs
            preds          = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            total_samples += bs

    return {
        "loss":     total_loss / total_samples,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="binary"),
    }


# ─── 4. Boucle complète avec early stopping ──────────────────────────────────
def train_model(model, loaders, optimizer, num_epochs=3, warmup_ratio=0.1,
                device=None, max_grad_norm=1.0, patience=2, verbose=True):
    """
    Entraîne le modèle et retourne l'historique des métriques.

    Args:
        model         : Modèle BERT.
        loaders       : dict de DataLoaders (train, validation).
        optimizer     : Optimiseur AdamW.
        num_epochs    : Nombre maximum d'époques.
        warmup_ratio  : Fraction des steps dédiés au warmup.
        device        : CPU ou CUDA.
        patience      : Nombre d'époques sans amélioration avant arrêt.
        verbose       : Affichage des métriques à chaque époque.

    Returns:
        history : dict avec train_loss, train_acc, val_loss, val_acc, val_f1
    """
    if device is None:
        device = next(model.parameters()).device

    total_steps  = num_epochs * len(loaders["train"])
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_state        = None
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc  = train_epoch(model, loaders["train"], optimizer,
                                        scheduler, device, max_grad_norm)
        val_metrics      = evaluate(model, loaders["validation"], device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])

        if verbose:
            print(f"  Epoch {epoch}/{num_epochs} | "
                  f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f} | "
                  f"val_loss={val_metrics['loss']:.4f}  "
                  f"val_acc={val_metrics['accuracy']:.4f}  "
                  f"val_f1={val_metrics['f1']:.4f}")

        # Early stopping sur val_loss
        if val_metrics["loss"] < best_val_loss:
            best_val_loss     = val_metrics["loss"]
            epochs_no_improve = 0
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  ► Early stopping à l'époque {epoch}.")
                break

    elapsed = time.time() - t0
    if verbose:
        print(f"  Durée : {elapsed:.1f}s")

    # Restaurer le meilleur état
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    history["train_time"] = elapsed
    return history


if __name__ == "__main__":
    print("Module train_eval chargé.")
