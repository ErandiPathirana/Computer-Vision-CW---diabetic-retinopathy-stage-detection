"""
tf.data input pipelines, data augmentation and class-imbalance handling.

Rules that prevent data leakage / silent bugs:
  * augmentation is applied to the TRAINING set only (validation/test stay untouched);
  * class weights and oversampling use the training labels only;
  * images are already preprocessed (src/preprocess.py) and stay in [0, 255];
    EfficientNetB0 normalises internally, so no /255 is applied here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight

from . import config as C


# --------------------------------------------------------------------------- #
# Class imbalance
# --------------------------------------------------------------------------- #
def compute_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """'Balanced' weights n_samples / (n_classes * n_class_samples) from TRAIN labels only."""
    classes = np.arange(C.NUM_CLASSES)
    w = compute_class_weight("balanced", classes=classes, y=train_df["diagnosis"].values)
    return {int(c): float(x) for c, x in zip(classes, w)}


def oversample_minorities(train_df: pd.DataFrame, target_ratio: float = 0.5, seed: int = C.SEED) -> pd.DataFrame:
    """Duplicate minority-class rows until each class has at least target_ratio x majority size.

    Duplicates get different random augmentations every epoch, so they are not
    identical inputs to the network. Only the training dataframe is ever passed in.
    """
    counts = train_df["diagnosis"].value_counts()
    target = int(counts.max() * target_ratio)
    parts = [train_df]
    for cls, n in counts.items():
        if n < target:
            parts.append(train_df[train_df["diagnosis"] == cls].sample(target - n, replace=True, random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def plot_imbalance_handling(train_df: pd.DataFrame, oversampled_df: pd.DataFrame, weights: Dict[int, float],
                            out_dir: Path = C.AUG_DIR) -> None:
    """Figure: class counts before/after oversampling and the class weights."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, d, title in zip(axes[:2], (train_df, oversampled_df), ("Training set (original)", "After minority oversampling")):
        c = d["diagnosis"].value_counts().sort_index()
        ax.bar(C.CLASS_NAMES, c.values, color="#2a9d8f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
    axes[2].bar(C.CLASS_NAMES, [weights[i] for i in range(C.NUM_CLASSES)], color="#e9c46a")
    axes[2].set_title("Class weights (balanced)")
    axes[2].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "imbalance_handling.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def build_geometric_augmenter() -> keras.Sequential:
    """Flips, rotation (+-20 degrees) and zoom (+-10%), applied per-sample on a batch.

    Justification: a fundus image can be photographed in any orientation, so flips
    and rotations are label-preserving; small zoom mimics different camera framing.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=C.SEED),
        keras.layers.RandomRotation(20 / 360, fill_mode="constant", fill_value=0.0, seed=C.SEED),
        keras.layers.RandomZoom(0.1, fill_mode="constant", fill_value=0.0, seed=C.SEED),
    ], name="geometric_augmentation")


def _photometric(image: tf.Tensor, label: tf.Tensor):
    """Brightness / contrast jitter per image (simulates different exposure and cameras)."""
    image = tf.image.random_brightness(image, max_delta=20.0)      # pixel scale is 0-255
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return tf.clip_by_value(image, 0.0, 255.0), label


def _load(path: tf.Tensor, label: tf.Tensor):
    """Read a cached preprocessed PNG as float32 in [0, 255]."""
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.cast(img, tf.float32)                                   # NO division by 255 here
    img.set_shape([C.IMG_SIZE, C.IMG_SIZE, 3])
    return img, label


def make_dataset(df: pd.DataFrame, training: bool, batch_size: int = C.BATCH_SIZE, augment: bool = True
                 ) -> tf.data.Dataset:
    """Build a batched tf.data pipeline from a dataframe with 'proc_path' and 'diagnosis'."""
    paths = df["proc_path"].astype(str).values
    labels = df["diagnosis"].values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(df), seed=C.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training and augment:
        ds = ds.map(_photometric, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment:
        aug = build_geometric_augmenter()
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def check_pixel_range(ds: tf.data.Dataset) -> None:
    """Fail loudly if images were accidentally normalised to [0,1] (double-normalisation bug)."""
    x, _ = next(iter(ds.take(1)))
    hi = float(tf.reduce_max(x))
    assert 1.5 < hi <= 255.0 + 1e-3, f"Unexpected pixel range (max={hi}); EfficientNet expects 0-255."
    print(f"Pixel range check passed (max value {hi:.0f}).")


def save_augmentation_figure(df: pd.DataFrame, out_dir: Path = C.AUG_DIR, n: int = 7) -> None:
    """One training image and several random augmentations of it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    row = df.iloc[0]
    img, _ = _load(tf.constant(row["proc_path"]), tf.constant(int(row["diagnosis"])))
    aug = build_geometric_augmenter()
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    axes[0].imshow(img.numpy().astype("uint8"))
    axes[0].set_title("Original")
    for i in range(1, n + 1):
        a, _ = _photometric(img, 0)
        a = aug(a[None], training=True)[0]
        axes[i].imshow(np.clip(a.numpy(), 0, 255).astype("uint8"))
        axes[i].set_title(f"Augmented {i}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "augmentation_examples.png", dpi=130)
    plt.close(fig)
