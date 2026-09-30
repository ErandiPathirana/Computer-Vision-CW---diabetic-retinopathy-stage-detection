"""
Configuration file containing paths and hyperparameters.
Compatible with Google Colab (/content/...) and Windows local execution.

Path resolution strategy
------------------------
REPO_ROOT is derived from this file's own location (src/config.py → parent.parent),
so it works regardless of the current working directory.
An environment variable DR_REPO_ROOT can override it when needed.
"""
import os
import pathlib

# -----------------
# Hyperparameters  (DO NOT CHANGE — kept identical for result reproducibility)
# -----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
BACKBONE = "EfficientNetB0"  # Options: "EfficientNetB0", "ResNet50", "MobileNetV2"

# -----------------
# Class Definition
# -----------------
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = len(CLASS_NAMES)

# -----------------
# Grad-CAM layer name (single source of truth)
# -----------------
# EfficientNetB0's final conv layer is "top_conv".
# Change this constant if you swap the backbone.
GRADCAM_LAYER_MAP = {
    "EfficientNetB0": "top_conv",
    "ResNet50": "conv5_block3_out",
    "MobileNetV2": "out_relu",
}
GRADCAM_LAYER = GRADCAM_LAYER_MAP.get(BACKBONE, "top_conv")

# -----------------
# Paths — cross-platform (Colab + Windows)
# -----------------
# This file lives at <repo>/src/config.py → parent = src/ → parent.parent = repo root
_THIS_FILE = pathlib.Path(__file__).resolve()
_DEFAULT_REPO_ROOT = _THIS_FILE.parent.parent  # e.g. /content/<repo> or C:\<repo>

# Allow an env-var override for special layouts (e.g. mounted Drive folders)
REPO_ROOT = pathlib.Path(os.environ.get("DR_REPO_ROOT", _DEFAULT_REPO_ROOT))

# Raw dataset inputs — placed outside the repo so they survive a repo re-clone.
# On Colab this resolves to /content/data/raw (matches legacy path).
# On Windows it resolves to <repo>/../data/raw (next to the repo folder).
DATA_DIR = REPO_ROOT.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset-specific paths (APTOS 2019)
TRAIN_IMAGES_DIR = RAW_DATA_DIR / "train_images"
TEST_IMAGES_DIR  = RAW_DATA_DIR / "test_images"
TRAIN_CSV        = RAW_DATA_DIR / "train.csv"
TEST_CSV         = RAW_DATA_DIR / "test.csv"

# Outputs — always inside the repo's results/ folder
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR  = RESULTS_DIR / "models"
LOGS_DIR    = RESULTS_DIR / "logs"

# Canonical best-model path (written by train.py, read by evaluate.py and app.py)
BEST_MODEL_PATH = RESULTS_DIR / "best_model.keras"

# -----------------
# Convert pathlib.Path objects to str for libraries that don't accept Path objects
# (os.path.join, pandas, cv2, etc. all accept str; TF/Keras accept both)
# -----------------
DATA_DIR           = str(DATA_DIR)
RAW_DATA_DIR       = str(RAW_DATA_DIR)
PROCESSED_DATA_DIR = str(PROCESSED_DATA_DIR)
TRAIN_IMAGES_DIR   = str(TRAIN_IMAGES_DIR)
TEST_IMAGES_DIR    = str(TEST_IMAGES_DIR)
TRAIN_CSV          = str(TRAIN_CSV)
TEST_CSV           = str(TEST_CSV)
RESULTS_DIR        = str(RESULTS_DIR)
MODELS_DIR         = str(MODELS_DIR)
LOGS_DIR           = str(LOGS_DIR)
BEST_MODEL_PATH    = str(BEST_MODEL_PATH)

# NOTE: Output directories are created lazily by the functions that need them,
# NOT at import time — avoids creating folders on systems where they're irrelevant.
