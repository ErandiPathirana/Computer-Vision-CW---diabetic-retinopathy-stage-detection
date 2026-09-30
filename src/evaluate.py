"""
Model evaluation and error analysis on the untouched TEST set.

Produces: accuracy, per-class and macro precision / recall / F1, quadratic weighted
kappa, confusion matrices, per-class ROC/AUC, a confidence histogram, a gallery of
misclassified images, Grad-CAM examples, and a data-driven written interpretation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from . import config as C


# --------------------------------------------------------------------------- #
# Predictions and metrics
# --------------------------------------------------------------------------- #
def predict_df(model, df: pd.DataFrame, batch_size: int = C.BATCH_SIZE) -> np.ndarray:
    """Class probabilities for every row (no augmentation, no shuffling)."""
    from .augment import make_dataset      # imported here so this module loads without TensorFlow when not needed
    ds = make_dataset(df, training=False, batch_size=batch_size)
    return model.predict(ds, verbose=0)


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> Dict:
    """All headline metrics in one dictionary."""
    y_pred = probs.argmax(1)
    labels = list(range(C.NUM_CLASSES))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    y_bin = label_binarize(y_true, classes=labels)
    auc = {}
    for i in labels:
        if y_bin[:, i].sum() > 0:
            auc[C.CLASS_NAMES[i]] = float(roc_auc_score(y_bin[:, i], probs[:, i]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "per_class": {C.CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                          "support": int(s[i])} for i in labels},
        "roc_auc_per_class": auc,
        "macro_roc_auc": float(np.mean(list(auc.values()))) if auc else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": int(len(y_true)),
    }


def quick_val_scores(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Small helper used by the experiments (accuracy, macro F1, QWK)."""
    y_pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]),
            "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(cm: np.ndarray, out_dir: Path) -> None:
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, data, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "d"),
                                 (axes[1], norm, "Confusion matrix (row-normalised = recall)", ".2f")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(C.NUM_CLASSES))
        ax.set_yticks(range(C.NUM_CLASSES))
        ax.set_xticklabels(C.CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(C.CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(C.NUM_CLASSES):
            for j in range(C.NUM_CLASSES):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if data[i, j] > data.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_roc(y_true: np.ndarray, probs: np.ndarray, out_dir: Path) -> None:
    y_bin = label_binarize(y_true, classes=list(range(C.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i in range(C.NUM_CLASSES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{C.CLASS_NAMES[i]} (AUC {roc_auc_score(y_bin[:, i], probs[:, i]):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest ROC curves (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_confidence(y_true: np.ndarray, probs: np.ndarray, out_dir: Path) -> None:
    conf, ok = probs.max(1), probs.argmax(1) == y_true
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(conf[ok], bins=20, alpha=0.7, label="correct", color="#2a9d8f")
    ax.hist(conf[~ok], bins=20, alpha=0.7, label="wrong", color="#e76f51")
    ax.axvline(C.LOW_CONFIDENCE_THRESHOLD, color="k", linestyle="--", label="low-confidence threshold")
    ax.set_xlabel("Top-class probability")
    ax.set_ylabel("Images")
    ax.set_title("Prediction confidence: correct vs wrong")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "confidence_histogram.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def error_analysis(model, test_df: pd.DataFrame, probs: np.ndarray, out_dir: Path, n_gallery: int = 12,
                   n_cam: int = 4) -> Dict:
    """Misclassified gallery, Grad-CAM panels and a written, data-driven interpretation."""
    from .gradcam import make_gradcam, overlay_heatmap
    from .preprocess import load_rgb

    y_true, y_pred = test_df["diagnosis"].values, probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(y_true != y_pred)[0]
    right = np.where(y_true == y_pred)[0]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C.NUM_CLASSES)))

    # most confused (true -> predicted) pairs
    pairs = sorted(((int(cm[i, j]), i, j) for i in range(C.NUM_CLASSES) for j in range(C.NUM_CLASSES) if i != j),
                   reverse=True)[:5]
    confused = [{"true": C.CLASS_NAMES[i], "predicted": C.CLASS_NAMES[j], "count": n} for n, i, j in pairs if n > 0]

    def read(idx):
        return load_rgb(test_df.iloc[idx]["proc_path"])

    # gallery: most confident mistakes first (they are the most informative failures)
    order = wrong[np.argsort(-conf[wrong])][:n_gallery]
    gallery = [{"id_code": test_df.iloc[i]["id_code"], "true": C.CLASS_NAMES[y_true[i]],
                "predicted": C.CLASS_NAMES[y_pred[i]], "confidence": round(float(conf[i]), 3)} for i in order]
    if len(order):
        cols = 4
        rows = int(np.ceil(len(order) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), order):
            ax.imshow(read(i))
            ax.set_title(f"True: {C.CLASS_NAMES[y_true[i]]}\nPred: {C.CLASS_NAMES[y_pred[i]]} ({conf[i]:.0%})", fontsize=9)
        fig.suptitle("Misclassified test images (most confident errors first)")
        fig.tight_layout()
        fig.savefig(out_dir / "misclassified_gallery.png", dpi=120)
        plt.close(fig)
    (out_dir / "misclassified.json").write_text(json.dumps(gallery, indent=2))

    # Grad-CAM for some correct and some wrong predictions
    picks = [("Correct", i) for i in right[:n_cam]] + [("Wrong", i) for i in order[:n_cam]]
    if picks:
        fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.4), squeeze=False)
        for k, (kind, i) in enumerate(picks):
            img = read(i)
            heat, _, _ = make_gradcam(model, img)
            axes[0, k].imshow(img)
            axes[0, k].set_title(f"{kind}\nTrue {C.CLASS_NAMES[y_true[i]]}\nPred {C.CLASS_NAMES[y_pred[i]]}", fontsize=8)
            axes[1, k].imshow(overlay_heatmap(img, heat))
            axes[0, k].axis("off")
            axes[1, k].axis("off")
        fig.suptitle("Grad-CAM: where the model looks")
        fig.tight_layout()
        fig.savefig(out_dir / "gradcam_examples.png", dpi=120)
        plt.close(fig)

    # ---- data-driven interpretation (numbers computed, not invented)
    recalls = {C.CLASS_NAMES[i]: cm[i, i] / max(cm[i].sum(), 1) for i in range(C.NUM_CLASSES)}
    worst = min(recalls, key=recalls.get)
    off_by_one = float(np.mean(np.abs(y_true[wrong] - y_pred[wrong]) == 1)) if len(wrong) else 0.0
    severe_missed = int(np.sum((y_true >= 3) & (y_pred <= 1)))
    text = f"""# Error analysis (test set, n={len(y_true)})

* Overall the model made **{len(wrong)}** errors ({len(wrong) / len(y_true):.1%}).
* Weakest class by recall: **{worst}** ({recalls[worst]:.0%}). Per-class recall: {({k: round(v, 2) for k, v in recalls.items()})}.
* Most confused pairs (true -> predicted): {confused}.
* **{off_by_one:.0%}** of the errors are between neighbouring grades, which is expected because DR severity is
  a continuum and the grade boundaries (and labels) are partly subjective.
* Mean confidence on correct predictions is {conf[right].mean():.2f} versus {conf[wrong].mean() if len(wrong) else float('nan'):.2f}
  on wrong ones, so low confidence is a usable signal for sending an image to a human reader.
* Clinically the most worrying errors are severe/proliferative eyes predicted as none/mild: **{severe_missed}** case(s).

## Likely causes
* Class imbalance: minority grades have few training images, so their decision boundaries are weaker even with class weights.
* Resolution: downscaling to {C.IMG_SIZE}x{C.IMG_SIZE} can erase tiny lesions (microaneurysms, small haemorrhages).
* Label noise and blur/exposure variation in the source images.
"""
    (out_dir / "error_analysis.md").write_text(text)
    return {"most_confused": confused, "off_by_one_share": off_by_one, "severe_missed_as_none_or_mild": severe_missed}


# --------------------------------------------------------------------------- #
# One call for everything
# --------------------------------------------------------------------------- #
def evaluate_model(model, test_df: pd.DataFrame, out_dir: Path = C.EVAL_DIR) -> Dict:
    """Full evaluation on the test set; writes figures + metrics.json + class_names.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    probs = predict_df(model, test_df)
    y_true = test_df["diagnosis"].values
    metrics = compute_metrics(y_true, probs)
    metrics["error_analysis"] = error_analysis(model, test_df, probs, out_dir)
    plot_confusion_matrices(np.array(metrics["confusion_matrix"]), out_dir)
    plot_roc(y_true, probs, out_dir)
    plot_confidence(y_true, probs, out_dir)
    report = classification_report(y_true, probs.argmax(1), labels=list(range(C.NUM_CLASSES)),
                                   target_names=C.CLASS_NAMES, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "class_names.json").write_text(json.dumps(C.CLASS_NAMES))
    print(report)
    print(f"Accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
          f"QWK {metrics['quadratic_weighted_kappa']:.3f}")
    return metrics
