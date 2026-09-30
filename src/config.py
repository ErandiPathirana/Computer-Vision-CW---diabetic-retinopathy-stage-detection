"""
Central configuration for the DR stage detection project.

Every path and hyper-parameter lives here so that the notebook, the training
scripts and the web app all agree with each other. Nothing in this file imports
TensorFlow, so it is cheap to import anywhere (tests, web server, Colab).
"""
from pathlib import Path
import os

# --------------------------------------------------------------------------- #
# Environment detection: Colab keeps data on /content, a laptop keeps it in the
# repository folder. REPO_ROOT is derived from this file, so no absolute path
# to one particular machine is ever hard-coded.
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
IN_COLAB = ("COLAB_RELEASE_TAG" in os.environ) or Path("/content/sample_data").exists()

DATA_ROOT = Path("/content/data") if IN_COLAB else REPO_ROOT / "data"
EXTRACT_DIR = DATA_ROOT / "extracted"       # raw unzip target (temporary)
RAW_DIR = DATA_ROOT / "raw"                 # cleaned copy used by the project
IMG_DIR = RAW_DIR / "train_images"          # the labelled fundus images
CSV_PATH = RAW_DIR / "train.csv"            # columns: id_code, diagnosis
PROCESSED_DIR = DATA_ROOT / "processed"     # cache of preprocessed 224x224 PNGs

# Google Drive source (no Kaggle credentials are needed)
DRIVE_MOUNT = "/content/drive"
DRIVE_ZIP_NAME = "Images and train file.zip"

# Outputs (results/ is git-ignored except what we choose to export)
RESULTS_DIR = REPO_ROOT / "results"
SPLIT_DIR = RESULTS_DIR / "splits"
EDA_DIR = RESULTS_DIR / "eda"
PREPROC_DIR = RESULTS_DIR / "preprocessing"
AUG_DIR = RESULTS_DIR / "augmentation"
TRAIN_DIR = RESULTS_DIR / "training"
EVAL_DIR = RESULTS_DIR / "evaluation"
EXPERIMENT_DIR = RESULTS_DIR / "experiments"

MODELS_DIR = REPO_ROOT / "models"                    # what the web app loads
ASSETS_DIR = REPO_ROOT / "webapp" / "assets"         # small figures for the web app
BEST_MODEL_PATH = RESULTS_DIR / "best_model.keras"   # written by training
EXPORT_MODEL_PATH = MODELS_DIR / "best_model.keras"  # copy used by the web app

# --------------------------------------------------------------------------- #
# Classes (ICDR / APTOS labelling)
# --------------------------------------------------------------------------- #
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- #
# Fixed constants (kept unchanged across the project)
# --------------------------------------------------------------------------- #
IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)   # train / validation / test, stratified

# Training hyper-parameters
DROPOUT = 0.3
HEAD_LR = 1e-3          # phase 1: only the new classification head is trained
FINETUNE_LR = 1e-5      # phase 2: top backbone layers are fine-tuned
UNFREEZE_LAYERS = 30
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 20
EARLY_STOP_PATIENCE = 6
LR_PATIENCE = 3
LR_FACTOR = 0.5

# Preprocessing parameters
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
DENOISE_KSIZE = 3
UNSHARP_SIGMA = 2.0
UNSHARP_AMOUNT = 0.5
BLACK_THRESHOLD = 7

# Web app
LOW_CONFIDENCE_THRESHOLD = 0.6

BACKBONE_LAYER_NAME = "backbone"   # name of the pretrained sub-model (used by Grad-CAM)


def ensure_dirs() -> None:
    """Create output folders on demand (never at import time)."""
    for d in (RESULTS_DIR, SPLIT_DIR, EDA_DIR, PREPROC_DIR, AUG_DIR, TRAIN_DIR,
              EVAL_DIR, EXPERIMENT_DIR, MODELS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
