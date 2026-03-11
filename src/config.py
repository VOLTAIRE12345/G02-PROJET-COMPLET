"""
config.py — Groupe G02
Chemins centralisés compatibles Colab ET local.
"""
import os

def _get_project_root() -> str:
    for colab_path in ["/content/G02_PROJET", "/content/G02-PROJET"]:
        if os.path.isdir(colab_path):
            return colab_path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_dir)

ROOT        = _get_project_root()
SRC_DIR     = os.path.join(ROOT, "src")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
REPORT_DIR  = os.path.join(ROOT, "report")

for _d in [RESULTS_DIR, FIGURES_DIR, REPORT_DIR]:
    os.makedirs(_d, exist_ok=True)

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
