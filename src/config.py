"""
config.py — Groupe G02
Chemins centralisés compatibles Colab ET local.
Tous les autres fichiers importent leurs chemins depuis ce module.
"""
import os

# ─── Détection automatique de l'environnement ─────────────────────────────────
def _get_project_root() -> str:
    """
    Retourne la racine du projet quel que soit l'environnement :
    - Google Colab : /content/G02_COLAB
    - Local        : dossier parent de src/
    """
    colab_path = "/content/G02_COLAB"
    if os.path.isdir(colab_path):
        return colab_path
    # Local : remonter depuis src/ jusqu'à la racine du projet
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_dir)


ROOT        = _get_project_root()
SRC_DIR     = os.path.join(ROOT, "src")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
REPORT_DIR  = os.path.join(ROOT, "report")

# Créer les dossiers s'ils n'existent pas
for _d in [RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Paramètres globaux ───────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
MAX_LENGTH = 200
SEED       = 42

if __name__ == "__main__":
    print("=== Config G02 ===")
    print(f"  ROOT        : {ROOT}")
    print(f"  RESULTS_DIR : {RESULTS_DIR}")
    print(f"  FIGURES_DIR : {FIGURES_DIR}")
    print(f"  MODEL       : {MODEL_NAME}")
