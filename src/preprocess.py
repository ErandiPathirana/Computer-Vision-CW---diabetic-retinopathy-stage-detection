"""
Image preprocessing for retinal fundus photographs.

ONE shared pipeline is used by training, evaluation and the web app, so the model
always sees identical inputs (a common source of silent bugs otherwise).

Pipeline (each step has a reason):
  1. Crop black borders   - fundus photos sit inside a black frame of varying size;
                            cropping removes non-informative pixels and makes the
                            retina fill the frame.
  2. Pad to square+resize - keeps the aspect ratio (no stretching), then resizes to
                            IMG_SIZE x IMG_SIZE, the input size of EfficientNetB0.
  3. Noise removal        - a small Gaussian blur suppresses sensor noise that the
                            next step would otherwise amplify.
  4. CLAHE                - contrast-limited adaptive histogram equalisation on the
                            L channel of LAB colour space: lifts local contrast so
                            microaneurysms/exudates stand out without changing colours.
  5. Edge enhancement     - unsharp masking sharpens vessel edges and lesion borders.
Normalisation: Keras' EfficientNetB0 contains its own rescaling/normalisation layers
and expects pixel values in [0, 255]. We therefore output uint8 [0, 255] and only
cast to float32; dividing by 255 here would normalise twice and hurt accuracy.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

try:                       # tqdm is optional (nice progress bar on Colab)
    from tqdm.auto import tqdm
except ImportError:        # pragma: no cover
    def tqdm(x, **kw):     # type: ignore
        return x


# --------------------------------------------------------------------------- #
# Individual steps
# --------------------------------------------------------------------------- #
def crop_black_borders(img: np.ndarray, threshold: int = C.BLACK_THRESHOLD) -> np.ndarray:
    """Crop the black frame around the retina (RGB uint8 in, RGB uint8 out)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if mask.sum() < 0.05 * mask.size:      # nearly black image: do not crop
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pad_to_square(img: np.ndarray) -> np.ndarray:
    """Pad with black to a square so resizing never distorts the retina."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize(img: np.ndarray, size: int = C.IMG_SIZE) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def denoise(img: np.ndarray, ksize: int = C.DENOISE_KSIZE) -> np.ndarray:
    """Light Gaussian smoothing (small kernel so tiny lesions are preserved)."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def apply_clahe(img: np.ndarray, clip: float = C.CLAHE_CLIP, tile: Tuple[int, int] = C.CLAHE_TILE) -> np.ndarray:
    """CLAHE on the lightness channel only (colour is left untouched)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)


def unsharp_mask(img: np.ndarray, sigma: float = C.UNSHARP_SIGMA, amount: float = C.UNSHARP_AMOUNT) -> np.ndarray:
    """Edge enhancement: original + amount * (original - blurred)."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
def preprocess_image(img: np.ndarray, size: int = C.IMG_SIZE, return_steps: bool = False):
    """Run the full pipeline on an RGB uint8 image.

    Returns the final uint8 image (size x size x 3, values 0-255), or, if
    return_steps=True, a list of (step name, image) pairs for visualisation.
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    original = resize(pad_to_square(img), size)
    cropped = resize(pad_to_square(crop_black_borders(img)), size)
    smooth = denoise(cropped)
    enhanced = apply_clahe(smooth)
    sharp = unsharp_mask(enhanced)
    if return_steps:
        return [("Original", original), ("Cropped", cropped), ("Denoised", smooth),
                ("CLAHE", enhanced), ("Sharpened", sharp)]
    return sharp


def load_rgb(path: str) -> np.ndarray:
    """Read an image file as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_bytes(data: bytes, size: int = C.IMG_SIZE, return_steps: bool = False):
    """Same pipeline for an uploaded file (used by the web app)."""
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("The uploaded file is not a readable image.")
    return preprocess_image(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), size, return_steps)


def to_model_input(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,255] with a batch axis (EfficientNet normalises internally)."""
    return img_uint8.astype(np.float32)[None, ...]


# --------------------------------------------------------------------------- #
# Caching: preprocess every image once, then training just reads small PNGs
# --------------------------------------------------------------------------- #
def _process_one(args) -> Tuple[str, str]:
    image_id, src = args
    dst = C.PROCESSED_DIR / f"{image_id}.png"
    if not dst.exists():
        out = preprocess_image(load_rgb(src))
        cv2.imwrite(str(dst), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return image_id, str(dst)


def cache_preprocessed(df: pd.DataFrame, workers: int = 4) -> pd.DataFrame:
    """Preprocess all images once and add a 'proc_path' column.

    This is safe with respect to data leakage: the pipeline is deterministic and
    uses no statistics learned from the data.
    """
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    jobs = list(zip(df["id_code"], df["path"]))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = dict(tqdm(ex.map(_process_one, jobs), total=len(jobs), desc="Preprocessing"))
    out = df.copy()
    out["proc_path"] = out["id_code"].map(results)
    return out


# --------------------------------------------------------------------------- #
# Figures and quantitative evidence for the report
# --------------------------------------------------------------------------- #
def contrast_and_sharpness(img: np.ndarray) -> Tuple[float, float]:
    """Contrast = std of the grey levels; sharpness = variance of the Laplacian."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return float(gray.std()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_preprocessing_figures(df: pd.DataFrame, out_dir: Path = C.PREPROC_DIR, n_metric_images: int = 60) -> Dict:
    """Before/after figure per class, histograms, and contrast/sharpness measurements."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(C.SEED)

    # 1) one example per class, every step
    steps_names = ["Original", "Cropped", "Denoised", "CLAHE", "Sharpened"]
    fig, axes = plt.subplots(C.NUM_CLASSES, len(steps_names), figsize=(3 * len(steps_names), 3 * C.NUM_CLASSES))
    first_steps = None
    for cls in range(C.NUM_CLASSES):
        row = df[df["diagnosis"] == cls].sample(1, random_state=rng).iloc[0]
        steps = preprocess_image(load_rgb(row["path"]), return_steps=True)
        first_steps = first_steps or steps
        for j, (name, im) in enumerate(steps):
            ax = axes[cls, j]
            ax.imshow(im)
            ax.axis("off")
            if cls == 0:
                ax.set_title(name)
            if j == 0:
                ax.text(-0.05, 0.5, f"Grade {cls}", transform=ax.transAxes, rotation=90, va="center", ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "preprocessing_steps.png", dpi=120)
    plt.close(fig)

    # 2) histograms + quantitative measures on a sample of images
    sample = df.sample(min(n_metric_images, len(df)), random_state=C.SEED)
    before, after = [], []
    hist_b, hist_a = np.zeros(256), np.zeros(256)
    for p in sample["path"]:
        steps = dict(preprocess_image(load_rgb(p), return_steps=True))
        b, a = steps["Original"], steps["Sharpened"]
        before.append(contrast_and_sharpness(b))
        after.append(contrast_and_sharpness(a))
        hist_b += cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)], [0], None, [256], [0, 256]).ravel()
        hist_a += cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)], [0], None, [256], [0, 256]).ravel()
    before, after = np.array(before), np.array(after)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist_b[1:] / hist_b[1:].sum(), label="Before")
    axes[0].plot(hist_a[1:] / hist_a[1:].sum(), label="After")
    axes[0].set_title("Grey-level histogram (background excluded)")
    axes[0].set_xlabel("Grey level")
    axes[0].legend()
    for ax, k, name in zip(axes[1:], (0, 1), ("Contrast (std of grey levels)", "Sharpness (Laplacian variance)")):
        ax.bar(["Before", "After"], [before[:, k].mean(), after[:, k].mean()], color=["#9aa5b1", "#2a9d8f"])
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_dir / "preprocessing_metrics.png", dpi=150)
    plt.close(fig)

    metrics = {"n_images": int(len(sample)),
               "contrast_before": round(float(before[:, 0].mean()), 2), "contrast_after": round(float(after[:, 0].mean()), 2),
               "sharpness_before": round(float(before[:, 1].mean()), 2), "sharpness_after": round(float(after[:, 1].mean()), 2)}
    (out_dir / "preprocessing_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
    return metrics
