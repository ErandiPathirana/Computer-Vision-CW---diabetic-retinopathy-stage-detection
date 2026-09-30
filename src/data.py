"""
Dataset handling: loading the APTOS 2019 data from Google Drive, validating it,
exploring it (EDA) and creating the stratified train / validation / test split.

The dataset originally comes from the Kaggle "APTOS 2019 Blindness Detection"
competition. No Kaggle credentials are needed here: the zip file was downloaded
once by hand and stored in Google Drive.
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config as C

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# 1. Loading the dataset from Google Drive
# --------------------------------------------------------------------------- #
def mount_drive() -> None:
    """Mount Google Drive (Colab only)."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:  # pragma: no cover - only outside Colab
        raise RuntimeError("Google Drive can only be mounted inside Google Colab.") from exc
    drive.mount(C.DRIVE_MOUNT)


def find_drive_zip(zip_name: str = C.DRIVE_ZIP_NAME, root: str = C.DRIVE_MOUNT + "/MyDrive") -> Path:
    """Search Drive recursively for the dataset zip.

    First tries an exact (case-insensitive) file-name match. If nothing is found
    it lists every zip file it saw, so the user can copy the right name.
    """
    seen: List[Path] = []
    wanted = zip_name.lower()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".zip"):
                p = Path(dirpath) / f
                seen.append(p)
                if f.lower() == wanted:
                    return p
    listing = "\n  ".join(str(p) for p in seen[:30]) or "(no zip files found in Drive)"
    raise FileNotFoundError(
        f"Could not find '{zip_name}' in Google Drive.\nZip files that were found:\n  {listing}\n"
        "Fix: set config.DRIVE_ZIP_NAME or pass zip_path=... to load_dataset_from_drive()."
    )


def _find_image_folder(root: Path) -> Path:
    """Return the sub-folder that contains the most image files."""
    best, best_n = None, 0
    for dirpath, _, files in os.walk(root):
        n = sum(f.lower().endswith(IMAGE_EXTS) for f in files)
        if n > best_n:
            best, best_n = Path(dirpath), n
    if best is None:
        raise RuntimeError(f"No images were found inside {root}.")
    return best


def _find_label_csv(search_dirs: List[Path]) -> Path:
    """Find a CSV that has both 'id_code' and 'diagnosis' columns."""
    checked = []
    for base in search_dirs:
        for dirpath, _, files in os.walk(base):
            for f in files:
                if f.lower().endswith(".csv"):
                    p = Path(dirpath) / f
                    try:
                        cols = list(pd.read_csv(p, nrows=2).columns)
                    except Exception:
                        continue
                    checked.append(f"{p.name}: {cols}")
                    if "id_code" in cols and "diagnosis" in cols:
                        return p
    raise RuntimeError(
        "No labelled CSV found (needs columns 'id_code' and 'diagnosis'). "
        "A CSV without 'diagnosis' (e.g. the unlabelled Kaggle test.csv) cannot be used for "
        "training. CSV files seen: " + (", ".join(checked) or "none")
    )


def load_dataset_from_drive(zip_path: Optional[str] = None, force: bool = False) -> pd.DataFrame:
    """Copy the dataset from Google Drive into the layout the project expects.

    Steps: mount Drive -> locate zip -> unzip on the fast local disk -> move the
    image folder to RAW_DIR/train_images -> copy the labelled CSV to RAW_DIR/train.csv
    -> validate. Returns the validated dataframe.
    """
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if C.CSV_PATH.exists() and C.IMG_DIR.exists() and not force:
        print("Dataset already prepared, skipping. (use force=True to redo)")
        return validate_dataset(pd.read_csv(C.CSV_PATH))

    mount_drive()
    zpath = Path(zip_path) if zip_path else find_drive_zip()
    print("Using zip:", zpath)

    if C.EXTRACT_DIR.exists():
        shutil.rmtree(C.EXTRACT_DIR)
    C.EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(C.EXTRACT_DIR)

    img_folder = _find_image_folder(C.EXTRACT_DIR)
    # The CSV is normally inside the zip; also look next to the zip in Drive.
    csv_file = _find_label_csv([C.EXTRACT_DIR, zpath.parent])

    if C.IMG_DIR.exists():
        shutil.rmtree(C.IMG_DIR)
    shutil.move(str(img_folder), str(C.IMG_DIR))
    shutil.copy(csv_file, C.CSV_PATH)
    print(f"Images: {C.IMG_DIR}\nLabels: {C.CSV_PATH} (from {csv_file.name})")
    return validate_dataset(pd.read_csv(C.CSV_PATH))


# --------------------------------------------------------------------------- #
# 2. Validation
# --------------------------------------------------------------------------- #
def _resolve_image_paths(df: pd.DataFrame, img_dir: Path) -> pd.Series:
    """Map each id_code to a file on disk (any supported extension)."""
    on_disk = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    return df["id_code"].astype(str).map(lambda i: str(on_disk[i]) if i in on_disk else "")


def validate_dataset(df: pd.DataFrame, img_dir: Path = C.IMG_DIR, check_readable: bool = True) -> pd.DataFrame:
    """Check labels and images, report problems, and return a clean dataframe.

    The returned dataframe has columns: id_code, diagnosis, path.
    Raises a clear error (never an IndexError) if the data cannot be used.
    """
    if not {"id_code", "diagnosis"}.issubset(df.columns):
        raise ValueError(f"CSV needs 'id_code' and 'diagnosis' columns, has {list(df.columns)}")
    df = df[["id_code", "diagnosis"]].copy()
    df["id_code"] = df["id_code"].astype(str)

    bad_labels = ~df["diagnosis"].isin(range(C.NUM_CLASSES))
    if bad_labels.any():
        raise ValueError(f"{int(bad_labels.sum())} rows have labels outside 0-{C.NUM_CLASSES - 1}.")
    df["diagnosis"] = df["diagnosis"].astype(int)

    dup = int(df["id_code"].duplicated().sum())
    df = df.drop_duplicates("id_code").reset_index(drop=True)

    df["path"] = _resolve_image_paths(df, img_dir)
    missing = int((df["path"] == "").sum())
    df = df[df["path"] != ""].reset_index(drop=True)

    unreadable = 0
    if check_readable:
        keep = []
        for p in df["path"]:
            try:
                with Image.open(p) as im:      # reads the header only (fast)
                    im.verify()
                keep.append(True)
            except Exception:
                keep.append(False)
        unreadable = keep.count(False)
        df = df[keep].reset_index(drop=True)

    print(f"Validation: {len(df)} usable images | duplicates removed: {dup} | "
          f"CSV rows without an image: {missing} | unreadable images: {unreadable}")
    if len(df) == 0:
        raise RuntimeError("No labelled images matched the CSV. Do the zip and CSV come from the same set?")
    return df


# --------------------------------------------------------------------------- #
# 3. Exploratory data analysis
# --------------------------------------------------------------------------- #
def _read_small(path: str) -> Optional[np.ndarray]:
    """Read an image at 1/4 size (much faster) as RGB."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def explore_dataset(df: pd.DataFrame, out_dir: Path = C.EDA_DIR, samples_per_class: int = 3,
                    stat_samples: int = 60) -> Dict:
    """Produce the EDA figures and a JSON/markdown summary used in the report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = df["diagnosis"].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    imbalance = float(counts.max() / counts.min())

    # --- class distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(C.CLASS_NAMES, counts.values, color=["#2a9d8f", "#8ab17d", "#e9c46a", "#f4a261", "#e76f51"])
    for b, c, p in zip(bars, counts.values, pct.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c}\n({p}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Class distribution (imbalance ratio {imbalance:.1f}:1)")
    ax.set_ylabel("Number of images")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    # --- sample images per class
    rng = np.random.RandomState(C.SEED)
    fig, axes = plt.subplots(C.NUM_CLASSES, samples_per_class, figsize=(3 * samples_per_class, 2.8 * C.NUM_CLASSES))
    for cls in range(C.NUM_CLASSES):
        rows = df[df["diagnosis"] == cls]
        pick = rows.sample(min(samples_per_class, len(rows)), random_state=rng)
        for j in range(samples_per_class):
            ax = axes[cls, j]
            ax.axis("off")
            if j < len(pick):
                img = _read_small(pick.iloc[j]["path"])
                if img is not None:
                    ax.imshow(img)
            if j == 0:
                ax.set_title(f"Grade {cls}: {C.CLASS_NAMES[cls]}", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sample_images.png", dpi=120)
    plt.close(fig)

    # --- image size statistics (header read only, so it is fast for all images)
    sizes = []
    for p in df["path"]:
        with Image.open(p) as im:
            sizes.append(im.size)
    sizes_arr = np.array(sizes)
    unique_sizes = len({tuple(s) for s in sizes})

    # --- brightness / sharpness statistics on a sample per class
    stats_rows = []
    for cls in range(C.NUM_CLASSES):
        rows = df[df["diagnosis"] == cls].sample(min(stat_samples, (df["diagnosis"] == cls).sum()), random_state=C.SEED)
        for p in rows["path"]:
            img = _read_small(p)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            stats_rows.append({"class": cls, "brightness": float(gray.mean()),
                               "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())})
    stats = pd.DataFrame(stats_rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ["brightness", "sharpness"], ["Mean brightness", "Sharpness (Laplacian variance)"]):
        data = [stats[stats["class"] == c][col].values for c in range(C.NUM_CLASSES)]
        ax.boxplot(data)
        ax.set_xticklabels([str(c) for c in range(C.NUM_CLASSES)])
        ax.set_title(f"{title} per class")
        ax.set_xlabel("DR grade")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_statistics.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_images": int(len(df)),
        "class_counts": {C.CLASS_NAMES[i]: int(counts.get(i, 0)) for i in range(C.NUM_CLASSES)},
        "class_percent": {C.CLASS_NAMES[i]: float(pct.get(i, 0)) for i in range(C.NUM_CLASSES)},
        "imbalance_ratio": round(imbalance, 2),
        "n_distinct_image_sizes": int(unique_sizes),
        "width_min_max": [int(sizes_arr[:, 0].min()), int(sizes_arr[:, 0].max())],
        "height_min_max": [int(sizes_arr[:, 1].min()), int(sizes_arr[:, 1].max())],
        "mean_brightness_by_class": stats.groupby("class")["brightness"].mean().round(1).to_dict(),
        "mean_sharpness_by_class": stats.groupby("class")["sharpness"].mean().round(1).to_dict(),
    }
    (out_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "eda_summary.md").write_text(_eda_markdown(summary))
    print(json.dumps(summary, indent=2))
    return summary


def _eda_markdown(s: Dict) -> str:
    """Turn the summary numbers into report-ready text (limitations and ethics included)."""
    return f"""# Dataset summary (APTOS 2019 Blindness Detection, Kaggle)

* Images used: **{s['n_images']}**, five DR grades (0 No DR ... 4 Proliferative DR).
* Class imbalance: largest/smallest class ratio is **{s['imbalance_ratio']}:1**; counts: {s['class_counts']}.
* Image sizes vary a lot ({s['n_distinct_image_sizes']} distinct sizes, width {s['width_min_max']}, height {s['height_min_max']}),
  so resizing and border cropping are required.

## Limitations
* Small dataset for deep learning, and the minority classes (Severe, Proliferative) have few images.
* Single-source data (one clinic network in India): the model may not generalise to other cameras or populations.
* Labels come from a single grading and contain some noise, especially between neighbouring grades.
* Image quality varies (blur, under/over-exposure, artefacts).

## Ethical concerns
* Fundus images are medical data; the Kaggle data is de-identified and licensed for the competition only, so it is not redistributed in this repository.
* Demographics are unknown, so performance across age, sex and ethnicity cannot be checked (fairness risk).
* Missing severe cases (false negatives) can delay treatment, so the system is a decision-support prototype only.
"""


# --------------------------------------------------------------------------- #
# 4. Stratified split
# --------------------------------------------------------------------------- #
def make_splits(df: pd.DataFrame, out_dir: Path = C.SPLIT_DIR, seed: int = C.SEED
                ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split, saved as CSV, with a proof that nothing leaks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    train_r, val_r, test_r = C.SPLIT_RATIOS
    train_df, rest = train_test_split(df, test_size=1 - train_r, stratify=df["diagnosis"], random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=test_r / (val_r + test_r), stratify=rest["diagnosis"],
                                       random_state=seed)
    train_df, val_df, test_df = (d.reset_index(drop=True) for d in (train_df, val_df, test_df))

    # No image may appear in more than one split (data-leakage guard).
    ids = [set(d["id_code"]) for d in (train_df, val_df, test_df)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "Data leakage: overlapping images!"

    for name, d in zip(("train", "val", "test"), (train_df, val_df, test_df)):
        d.to_csv(out_dir / f"{name}.csv", index=False)

    table = pd.DataFrame({n: d["diagnosis"].value_counts().sort_index()
                          for n, d in zip(("train", "val", "test"), (train_df, val_df, test_df))}).fillna(0).astype(int)
    table.index = C.CLASS_NAMES
    table.loc["Total"] = table.sum()
    table.to_csv(out_dir / "split_counts.csv")
    print(table)

    fig, ax = plt.subplots(figsize=(8, 4))
    table.drop("Total").plot(kind="bar", ax=ax)
    ax.set_title("Images per class in each split (stratified)")
    ax.set_ylabel("Images")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(out_dir / "split_distribution.png", dpi=150)
    plt.close(fig)
    return train_df, val_df, test_df


def load_splits(split_dir: Path = C.SPLIT_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reload the saved split CSVs."""
    return tuple(pd.read_csv(split_dir / f"{n}.csv") for n in ("train", "val", "test"))  # type: ignore
