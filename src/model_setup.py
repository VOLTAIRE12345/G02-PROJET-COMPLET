"""
model_setup.py — Groupe G02
Chargement de BERT-base-uncased (M02) avec adaptation automatique CPU / GPU Colab.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_NAME, NUM_LABELS

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_device():
    """Détecte automatiquement GPU (Colab T4) ou CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(4)
    print(f"Device : {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    return device


def get_model_and_tokenizer(model_name=MODEL_NAME, num_labels=NUM_LABELS,
                             dropout_prob=0.1, device=None):
    """
    Charge BERT-base-uncased avec dropout configurable (hyperparamètre P02).

    Args:
        dropout_prob : Taux de dropout appliqué à hidden ET attention layers.
    """
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        hidden_dropout_prob=dropout_prob,
        attention_probs_dropout_prob=dropout_prob,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modèle : {model_name} | dropout={dropout_prob:.4f} | {n_params/1e6:.1f}M paramètres")
    return model, tokenizer


def build_optimizer(model, lr=2e-5, weight_decay=1e-4):
    """
    AdamW avec weight decay découplé (recommandé pour BERT).
    Les biais et LayerNorm ne sont PAS régularisés (pratique standard).
    """
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    from torch.optim import AdamW
    return AdamW(grouped_params, lr=lr)


if __name__ == "__main__":
    device = get_device()
    model, tok = get_model_and_tokenizer(dropout_prob=0.1, device=device)
    opt = build_optimizer(model, lr=2e-5, weight_decay=1e-4)
    print("Modèle et optimiseur prêts.")
    inputs = tok("This movie was absolutely amazing!",
                 return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        out = model(**inputs)
    pred = out.logits.argmax().item()
    print(f"Test prédiction : {'POSITIF' if pred == 1 else 'NEGATIF'}")
