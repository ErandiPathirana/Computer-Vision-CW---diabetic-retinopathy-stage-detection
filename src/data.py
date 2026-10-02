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
import re
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
# Kaggle DR datasets use different column names / layouts, so the loader understands:
#   (a) a labelled CSV (id_code,diagnosis  or  image,level  or ...) + a folder of images, and
#   (b) images stored in class folders (0,1,2,3,4  or  No_DR, Mild, Moderate, Severe, Proliferative_DR).
ID_ALIASES = ["id_code", "image", "image_id", "id", "filename", "file_name", "image_name", "name"]
LABEL_ALIASES = ["diagnosis", "level", "label", "grade", "dr_grade", "stage", "severity", "class"]


FOLDER_HINTS = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]   # typical folder names, used in messages


def _norm(text: str) -> str:
    """lower-case, non-alphanumerics -> '_' (so 'No DR', 'no-dr' and 'No_DR' all match)."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def grade_from_name(name) -> Optional[int]:
    """Map a folder name or label text to a DR grade 0-4 (None if it is not a grade name)."""
    n = _norm(name)
    if n in {"0", "1", "2", "3", "4"}:
        return int(n)
    m = re.fullmatch(r"(?:grade|class|stage|level|dr)_?([0-4])", n)
    if m:
        return int(m.group(1))
    if "proliferat" in n and not n.startswith("non"):
        return 4
    if n == "pdr":
        return 4
    if "severe" in n:
        return 3
    if "moderate" in n:
        return 2
    if "mild" in n:
        return 1
    if n in {"no_dr", "nodr", "normal", "none", "no_apparent_dr", "healthy", "no_diabetic_retinopathy"}:
        return 0
    return None


def standardize_table(d: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return a dataframe with columns id_code, diagnosis (ints 0-4), or None if d is not a label table."""
    d = d.rename(columns=lambda c: str(c).replace("\ufeff", "").strip())      # Excel adds a hidden BOM / spaces
    lower = {str(c).lower(): c for c in d.columns}
    id_col = next((lower[a] for a in ID_ALIASES if a in lower), None)
    lab_col = next((lower[a] for a in LABEL_ALIASES if a in lower), None)
    if id_col is None or lab_col is None:
        return None
    out = pd.DataFrame({"id_code": d[id_col].astype(str)})
    labels = d[lab_col]
    if pd.api.types.is_numeric_dtype(labels):
        out["diagnosis"] = labels
    else:
        out["diagnosis"] = labels.map(grade_from_name)
    out = out.dropna(subset=["diagnosis"])
    out["diagnosis"] = out["diagnosis"].astype(int)
    return out if len(out) else None


def mount_drive() -> None:
    """Mount Google Drive (Colab only)."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:  # pragma: no cover - only outside Colab
        raise RuntimeError("Google Drive can only be mounted inside Google Colab.") from exc
    drive.mount(C.DRIVE_MOUNT)


def find_drive_zip(zip_name: str = C.DRIVE_ZIP_NAME, root: str = C.DRIVE_MOUNT + "/MyDrive") -> Path:
    """Search Drive recursively for the dataset zip (exact name, case-insensitive)."""
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


def _list_images(root: Path) -> List[Path]:
    return [Path(d) / f for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(IMAGE_EXTS)]


def inspect_dataset_folder(root: Path = C.EXTRACT_DIR, top: int = 15) -> None:
    """Print what is inside the dataset folder (images per folder, CSV files and their label counts).

    Run this when something looks wrong; the output is also useful in the report.
    """
    root = Path(root)
    per_dir: Dict[str, int] = {}
    csvs = []
    for d, _, fs in os.walk(root):
        n = sum(f.lower().endswith(IMAGE_EXTS) for f in fs)
        if n:
            per_dir[str(Path(d).relative_to(root))] = n
        csvs += [Path(d) / f for f in fs if f.lower().endswith(".csv")]
    print(f"Folder: {root}")
    print(f"Total images: {sum(per_dir.values())}")
    for d, n in sorted(per_dir.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {n:>6} images in  {d or '.'}")
    if not csvs:
        print("No CSV files found.")
    for p in csvs:
        try:
            d = pd.read_csv(p)
        except Exception as exc:
            print(f"CSV {p.name}: unreadable ({exc})")
            continue
        std = standardize_table(d)
        info = (f"labels {std['diagnosis'].value_counts().sort_index().to_dict()}" if std is not None
                else "no usable id/label columns")
        print(f"CSV {p.name}: {d.shape[0]} rows, columns {list(d.columns)} -> {info}")


def _find_label_csv(search_dirs: List[Path]) -> Tuple[Path, pd.DataFrame]:
    """Find a labelled CSV and return (path, standardised id_code/diagnosis table).

    CSVs whose labels are all identical (Kaggle's sample_submission.csv is all zeros) are
    rejected. Files called train.csv are preferred, then the file with the most rows.
    """
    candidates, notes = [], []
    for base in search_dirs:
        for dirpath, _, files in os.walk(base):
            for f in files:
                if not f.lower().endswith(".csv"):
                    continue
                p = Path(dirpath) / f
                try:
                    raw = pd.read_csv(p)
                except Exception:
                    continue
                std = standardize_table(raw)
                if std is None:
                    notes.append(f"{f}: columns {list(raw.columns)} (no labels)")
                elif std["diagnosis"].nunique() < 2:
                    notes.append(f"{f}: {len(std)} rows but only one distinct label "
                                 f"({std['diagnosis'].unique().tolist()}), looks like sample_submission.csv")
                else:
                    candidates.append((p.name.lower() != "train.csv", -len(std), str(p), p, std))
    if candidates:
        best = sorted(candidates, key=lambda c: c[:3])[0]
        return best[3], best[4]
    raise RuntimeError("No labelled CSV found. CSV files seen: " + ("; ".join(notes) or "none"))


def _frame_from_class_folders(root: Path) -> pd.DataFrame:
    """Build id_code / diagnosis / path from folders named after the grades (0..4 or No_DR, Mild, ...)."""
    rows = []
    for d, _, fs in os.walk(root):
        grade = grade_from_name(Path(d).name)
        if grade is None:
            continue
        for f in sorted(fs):
            if f.lower().endswith(IMAGE_EXTS):
                rows.append((grade, str(Path(d) / f)))
    df = pd.DataFrame(rows, columns=["diagnosis", "path"])
    if df["diagnosis"].nunique() < 2:
        return pd.DataFrame()
    df["id_code"] = [f"{i:05d}_{Path(p).stem}" for i, p in enumerate(df["path"])]   # unique, file-name safe
    return df[["id_code", "diagnosis", "path"]]


def _attach_paths(table: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Find each CSV id's image anywhere under root (matches with or without file extension)."""
    by_name, by_stem = {}, {}
    for p in sorted(_list_images(root), key=lambda q: ("train" not in str(q).lower(), str(q))):
        by_name.setdefault(p.name.lower(), str(p))
        by_stem.setdefault(p.stem.lower(), str(p))
    ids = table["id_code"].astype(str)
    table = table.copy()
    table["path"] = [by_name.get(i.lower()) or by_stem.get(Path(i).stem.lower()) or "" for i in ids]
    return table


LOADER_VERSION = "2.2 (finds images by file name in all folders; checks every grade)"


def load_dataset_from_drive(zip_path: Optional[str] = None, force: bool = False,
                            source_dir: Optional[str] = None) -> pd.DataFrame:
    """Copy the dataset from Google Drive into Colab and return the validated dataframe.

    Steps: mount Drive -> unzip on the fast local disk (or copy an already-unzipped Drive folder
    given as source_dir) -> find the labels (CSV, or class folders) -> validate -> save
    RAW_DIR/train.csv (columns id_code, diagnosis, path). Stops with a clear message on problems.
    """
    print("Dataset loader version:", LOADER_VERSION)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if C.CSV_PATH.exists() and not force:
        try:
            print("Dataset already prepared (use force=True to redo).")
            return validate_dataset(pd.read_csv(C.CSV_PATH))
        except Exception as exc:
            print(f"Prepared copy is not usable ({exc}); preparing again.")

    mount_drive()
    if C.EXTRACT_DIR.exists():
        shutil.rmtree(C.EXTRACT_DIR)
    search_extra: List[Path] = []
    if source_dir:
        print("Copying folder from Drive:", source_dir)
        shutil.copytree(source_dir, C.EXTRACT_DIR)
    else:
        zpath = Path(zip_path) if zip_path else find_drive_zip()
        print("Using zip:", zpath)
        C.EXTRACT_DIR.mkdir(parents=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(C.EXTRACT_DIR)
        search_extra.append(zpath.parent)          # a labelled CSV may sit next to the zip in Drive

    inspect_dataset_folder(C.EXTRACT_DIR)

    try:
        csv_file, table = _find_label_csv([C.EXTRACT_DIR] + search_extra)
        print(f"Labels: {csv_file.name} ({len(table)} rows)")
        df = _attach_paths(table, C.EXTRACT_DIR)
    except RuntimeError as csv_problem:
        df = _frame_from_class_folders(C.EXTRACT_DIR)
        if df.empty:
            raise RuntimeError(
                f"{csv_problem}\nNo class folders (0-4 or No_DR/Mild/Moderate/Severe/Proliferative_DR) were found either.\n"
                "This dataset has NO diagnosis labels, so a classifier cannot be trained on it. A test set "
                "(Kaggle test.csv / test_images) is unlabelled. You need either the labelled 'train.csv' + "
                "'train_images', or images sorted into one folder per grade."
            ) from None
        print(f"No labelled CSV; using class folders ({len(df)} images).")

    df = validate_dataset(df)
    df.to_csv(C.CSV_PATH, index=False)
    return df


# --------------------------------------------------------------------------- #
# 2. Validation
# --------------------------------------------------------------------------- #
def _resolve_image_paths(df: pd.DataFrame, img_dir: Path) -> pd.Series:
    """Map each id_code to a file on disk (any supported extension)."""
    on_disk = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    return df["id_code"].astype(str).map(lambda i: str(on_disk[i]) if i in on_disk else "")


def validate_dataset(df: pd.DataFrame, img_dir: Path = C.IMG_DIR, check_readable: bool = True,
                     require_all_grades: bool = True) -> pd.DataFrame:
    """Check labels and images, report problems, and return a clean dataframe.

    The returned dataframe has columns: id_code, diagnosis, path.
    Raises a clear error (never an IndexError) if the data cannot be used.
    """
    if "path" not in df.columns:
        std = standardize_table(df)
        if std is None:
            raise ValueError(f"CSV needs an id column and a label column (e.g. 'id_code' and 'diagnosis'), has {list(df.columns)}")
        df = std
    df = df[[c for c in ("id_code", "diagnosis", "path") if c in df.columns]].copy()
    df["id_code"] = df["id_code"].astype(str)

    bad_labels = ~df["diagnosis"].isin(range(C.NUM_CLASSES))
    if bad_labels.any():
        raise ValueError(f"{int(bad_labels.sum())} rows have labels outside 0-{C.NUM_CLASSES - 1}.")
    df["diagnosis"] = df["diagnosis"].astype(int)
    if df["diagnosis"].nunique() < 2:
        raise ValueError(
            f"Only one class is present (all labels = {df['diagnosis'].iloc[0]}). This is usually sample_submission.csv "
            "or an unlabelled file. Use the labelled train.csv from the Kaggle APTOS Data tab.")

    dup = int(df["id_code"].duplicated().sum())
    df = df.drop_duplicates("id_code").reset_index(drop=True)

    if "path" not in df.columns:
        df["path"] = _resolve_image_paths(df, img_dir)
    df["path"] = df["path"].fillna("").astype(str)
    exists = df["path"].map(lambda p: bool(p) and os.path.isfile(p))
    missing = int((~exists).sum())
    labelled_per_grade = df["diagnosis"].value_counts()
    df = df[exists].reset_index(drop=True)

    if require_all_grades:
        # A grade that is labelled in the CSV but has no image files at all (e.g. a missing No_DR folder in the zip)
        # would silently produce a model that can never predict that grade.
        for g in range(C.NUM_CLASSES):
            if (df["diagnosis"] == g).sum() == 0:
                n_rows = int(labelled_per_grade.get(g, 0))
                why = (f"the labels list {n_rows} images for it but NO image files were found "
                       f"(is the '{FOLDER_HINTS[g]}' folder missing from the zip?)") if n_rows else "no images or labels exist for it"
                raise ValueError(f"Grade {g} ({C.CLASS_NAMES[g]}): {why}. All five grades are needed to train a "
                                 "five-stage classifier. Add the missing images to the zip and run again "
                                 "(or pass require_all_grades=False to train on fewer grades).")

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

    print(f"Validation: {len(df)} usable images | duplicate ids removed: {dup} | "
          f"labelled rows without an image file: {missing} | unreadable images: {unreadable}")
    if len(df) == 0:
        raise RuntimeError("No labelled images matched. Do the images and the labels come from the same set?")
    if df["diagnosis"].nunique() < 2:
        raise ValueError("After removing missing images only one class is left; check that the labels match the images.")
    print("Images per grade:", df["diagnosis"].value_counts().sort_index().to_dict())
    if len(df) < 500:
        print("WARNING: fewer than 500 labelled images. The full APTOS training set has 3662.")
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
    smallest = df["diagnosis"].value_counts()
    if smallest.min() < 3:
        raise ValueError(f"Grade {int(smallest.idxmin())} has only {int(smallest.min())} image(s); at least 3 per grade are "
                         "needed for a stratified train/val/test split. Check that the labels are correct.")
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
    table = table.reindex(range(C.NUM_CLASSES), fill_value=0)     # always one row per grade
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
