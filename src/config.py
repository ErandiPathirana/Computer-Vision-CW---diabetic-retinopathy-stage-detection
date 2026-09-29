"""
Configuration file containing paths and hyperparameters.
Designed to be compatible with Google Colab execution.
"""
import os

# -----------------
# Hyperparameters
# -----------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
BACKBONE = "EfficientNetB0" # Options: "EfficientNetB0", "ResNet50", "MobileNetV2"

# -----------------
# Class Definition
# -----------------
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
NUM_CLASSES = len(CLASS_NAMES)

# -----------------
# Paths (Colab Environment)
# -----------------
# Absolute paths specifically constructed for Google Colab environment
BASE_DIR = "/content"
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Dataset specific paths (APTOS 2019)
TRAIN_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(RAW_DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_CSV = os.path.join(RAW_DATA_DIR, "test.csv")

# Outputs
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Ensure critical output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
