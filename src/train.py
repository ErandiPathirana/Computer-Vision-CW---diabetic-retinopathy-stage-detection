"""
Two-phase training with the callbacks required by the coursework rubric:
validation monitoring, early stopping, learning-rate scheduling and check-pointing,
plus class weights (or oversampling) against class imbalance.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config as C
from .augment import (check_pixel_range, compute_class_weights, make_dataset, oversample_minorities)
from .model import build_model, configure_phase1, configure_phase2, parameter_counts


def set_seeds(seed: int = C.SEED) -> None:
    """Make runs reproducible (Python, NumPy and TensorFlow)."""
    keras.utils.set_random_seed(seed)


def _callbacks(tag: str, phase: int, out_dir: Path, save_ckpt: bool):
    """Early stopping + LR scheduling + checkpoint + CSV log + TensorBoard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.EARLY_STOP_PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=C.LR_FACTOR, patience=C.LR_PATIENCE, min_lr=1e-7,
                                          verbose=1),
        keras.callbacks.CSVLogger(str(out_dir / f"{tag}_phase{phase}_log.csv")),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "tensorboard" / f"{tag}_phase{phase}")),
    ]
    if save_ckpt:
        cbs.append(keras.callbacks.ModelCheckpoint(str(out_dir / f"{tag}_phase{phase}.keras"), monitor="val_loss",
                                                   save_best_only=True))
    return cbs


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, backbone: str = "efficientnetb0",
                dropout: float = C.DROPOUT, head_lr: float = C.HEAD_LR, finetune_lr: float = C.FINETUNE_LR,
                unfreeze: int = C.UNFREEZE_LAYERS, epochs1: int = C.PHASE1_EPOCHS, epochs2: int = C.PHASE2_EPOCHS,
                imbalance: str = "class_weights", tag: str = "main", out_dir: Path = C.TRAIN_DIR,
                save_final: bool = False, verbose: int = 1):
    """Train phase 1 (head) then phase 2 (fine-tuning).

    imbalance: 'class_weights' (default), 'oversample' or 'none'. All statistics come
    from train_df only.  Returns (model, history dict).
    """
    set_seeds()
    fit_df = oversample_minorities(train_df) if imbalance == "oversample" else train_df
    class_weight = compute_class_weights(train_df) if imbalance == "class_weights" else None

    train_ds = make_dataset(fit_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    check_pixel_range(train_ds)

    model = build_model(backbone, dropout)
    hist: Dict[str, Dict[str, list]] = {}
    t0 = time.time()

    # ---- Phase 1: train the new head only
    configure_phase1(model, head_lr)
    print("Phase 1 parameters:", parameter_counts(model))
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs1, class_weight=class_weight,
                   callbacks=_callbacks(tag, 1, out_dir, save_final), verbose=verbose)
    hist["phase1"] = {k: [float(v) for v in vals] for k, vals in h1.history.items()}
    best1 = min(h1.history["val_loss"])
    if save_final:
        model.save(out_dir / f"{tag}_phase1_best.keras")

    # ---- Phase 2: fine-tune the top backbone layers with a tiny learning rate
    if epochs2 > 0:
        configure_phase2(model, unfreeze, finetune_lr)
        print("Phase 2 parameters:", parameter_counts(model))
        h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs2, class_weight=class_weight,
                       callbacks=_callbacks(tag, 2, out_dir, save_final), verbose=verbose)
        hist["phase2"] = {k: [float(v) for v in vals] for k, vals in h2.history.items()}
        best2 = min(h2.history["val_loss"])
        # Keep whichever phase gave the lower validation loss (never look at the test set here).
        if save_final and best2 > best1:
            print("Fine-tuning did not improve val_loss; reverting to the phase-1 model.")
            model = keras.models.load_model(out_dir / f"{tag}_phase1_best.keras")
    hist["train_seconds"] = round(time.time() - t0, 1)  # type: ignore

    if save_final:
        C.BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save(C.BEST_MODEL_PATH)
        (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
        plot_history(hist, out_dir)
        print("Saved best model to", C.BEST_MODEL_PATH)
    return model, hist


def plot_history(hist: Dict, out_dir: Path = C.TRAIN_DIR) -> None:
    """Accuracy and loss curves for phase 1, phase 2 and combined."""
    out_dir.mkdir(parents=True, exist_ok=True)
    phases = [p for p in ("phase1", "phase2") if p in hist]
    fig, axes = plt.subplots(len(phases) + 1, 2, figsize=(12, 4 * (len(phases) + 1)))
    for row, p in enumerate(phases):
        for col, (m, title) in enumerate((("accuracy", "Accuracy"), ("loss", "Loss"))):
            ax = axes[row, col]
            ax.plot(hist[p][m], label="train")
            ax.plot(hist[p]["val_" + m], label="validation")
            ax.set_title(f"{p.replace('phase', 'Phase ')} - {title}")
            ax.set_xlabel("Epoch")
            ax.legend()
    for col, (m, title) in enumerate((("accuracy", "Accuracy"), ("loss", "Loss"))):
        ax = axes[-1, col]
        tr = sum((hist[p][m] for p in phases), [])
        va = sum((hist[p]["val_" + m] for p in phases), [])
        ax.plot(tr, label="train")
        ax.plot(va, label="validation")
        if "phase2" in hist:
            ax.axvline(len(hist["phase1"][m]) - 0.5, color="grey", linestyle="--", label="fine-tuning starts")
        ax.set_title(f"Combined - {title}")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)
