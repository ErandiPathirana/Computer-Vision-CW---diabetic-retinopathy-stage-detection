"""
Collect everything the web app needs into two small folders:
  models/          best_model.keras, class_names.json, metrics.json
  webapp/assets/   figures and JSON used by the app's Dataset, Training and Evaluation pages
Run after training + evaluation (the notebook does this for you).
"""
from __future__ import annotations

import json
from typing import Optional
import shutil
from pathlib import Path

import pandas as pd

from . import config as C

# (source path, destination name inside webapp/assets)
_ASSET_FILES = [
    (C.EDA_DIR / "class_distribution.png", "class_distribution.png"),
    (C.EDA_DIR / "sample_images.png", "sample_images.png"),
    (C.EDA_DIR / "quality_statistics.png", "quality_statistics.png"),
    (C.EDA_DIR / "eda_summary.json", "eda_summary.json"),
    (C.SPLIT_DIR / "split_distribution.png", "split_distribution.png"),
    (C.SPLIT_DIR / "split_counts.csv", "split_counts.csv"),
    (C.PREPROC_DIR / "preprocessing_steps.png", "preprocessing_steps.png"),
    (C.PREPROC_DIR / "preprocessing_metrics.png", "preprocessing_metrics.png"),
    (C.PREPROC_DIR / "preprocessing_metrics.json", "preprocessing_metrics.json"),
    (C.AUG_DIR / "augmentation_examples.png", "augmentation_examples.png"),
    (C.AUG_DIR / "imbalance_handling.png", "imbalance_handling.png"),
    (C.TRAIN_DIR / "training_curves.png", "training_curves.png"),
    (C.TRAIN_DIR / "history.json", "history.json"),
    (C.EVAL_DIR / "confusion_matrix.png", "confusion_matrix.png"),
    (C.EVAL_DIR / "roc_curves.png", "roc_curves.png"),
    (C.EVAL_DIR / "confidence_histogram.png", "confidence_histogram.png"),
    (C.EVAL_DIR / "misclassified_gallery.png", "misclassified_gallery.png"),
    (C.EVAL_DIR / "misclassified.json", "misclassified.json"),
    (C.EVAL_DIR / "gradcam_examples.png", "gradcam_examples.png"),
    (C.EVAL_DIR / "error_analysis.md", "error_analysis.md"),
    (C.EVAL_DIR / "classification_report.txt", "classification_report.txt"),
]


def _csv_to_json(src: Path, dst: Path) -> None:
    dst.write_text(json.dumps(pd.read_csv(src).to_dict(orient="records"), indent=2))


def export_samples(test_df: Optional[pd.DataFrame] = None, per_class: int = 1) -> None:
    """Copy a few TEST images (one per grade) into webapp/samples/ for the app's "Try a sample" buttons.

    Pass the test dataframe, or nothing (then results/splits/test.csv is used). A plain number is accepted
    as per_class for backwards compatibility. These are dataset images, so webapp/samples/ is git-ignored
    (the Kaggle terms do not allow redistribution); keep them on your own computer for the demo only.
    """
    if isinstance(test_df, int):                      # old call style: export_samples(2)
        per_class, test_df = test_df, None
    if test_df is None:
        test_csv = C.SPLIT_DIR / "test.csv"
        if not test_csv.exists():
            print("No test split found; skipping demo samples.")
            return
        test_df = pd.read_csv(test_csv)
    samples_dir = C.REPO_ROOT / "webapp" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for cls in range(C.NUM_CLASSES):
        for _, row in test_df[test_df["diagnosis"] == cls].head(per_class).iterrows():
            src = Path(row["path"])
            if src.exists():
                shutil.copy(src, samples_dir / f"grade{cls}_{src.stem}{src.suffix.lower()}")
                copied += 1
    print(f"Sample images copied: {copied} -> {samples_dir}")


def export_all() -> None:
    """Copy model + metrics to models/ and figures/tables to webapp/assets/."""
    C.ensure_dirs()
    missing = []

    # --- model files
    for src, dst in ((C.BEST_MODEL_PATH, C.EXPORT_MODEL_PATH),
                     (C.EVAL_DIR / "metrics.json", C.MODELS_DIR / "metrics.json"),
                     (C.EVAL_DIR / "class_names.json", C.MODELS_DIR / "class_names.json")):
        if src.exists():
            shutil.copy(src, dst)
        else:
            missing.append(str(src))

    # --- app assets
    for src, name in _ASSET_FILES:
        if src.exists():
            shutil.copy(src, C.ASSETS_DIR / name)
        else:
            missing.append(str(src))

    # --- experiment tables (CSV -> JSON so the app can render them)
    for stem in ("backbone_comparison", "hyperparameter_search", "imbalance_comparison"):
        src = C.EXPERIMENT_DIR / f"{stem}.csv"
        if src.exists():
            shutil.copy(src, C.ASSETS_DIR / f"{stem}.csv")
            _csv_to_json(src, C.ASSETS_DIR / f"{stem}.json")
        else:
            missing.append(str(src))

    export_samples()

    (C.ASSETS_DIR / "manifest.json").write_text(json.dumps(sorted(p.name for p in C.ASSETS_DIR.iterdir()), indent=2))
    size_mb = sum(p.stat().st_size for p in C.ASSETS_DIR.iterdir()) / 1e6
    model_mb = C.EXPORT_MODEL_PATH.stat().st_size / 1e6 if C.EXPORT_MODEL_PATH.exists() else 0
    print(f"Exported. Assets: {size_mb:.1f} MB | model: {model_mb:.1f} MB")
    if missing:
        print("Not found (skipped):\n  " + "\n  ".join(missing))
