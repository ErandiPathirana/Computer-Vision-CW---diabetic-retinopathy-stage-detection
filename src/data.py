"""
Data downloading and preparation module.
Includes functions for EDA.
"""
import os
import zipfile
import subprocess
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from src.config import (
    RAW_DATA_DIR, 
    TRAIN_CSV, 
    TRAIN_IMAGES_DIR, 
    RESULTS_DIR, 
    CLASS_NAMES
)

def download_aptos_dataset():
    """
    Downloads and extracts the APTOS 2019 Blindness Detection dataset using Kaggle API.
    Reads Kaggle credentials from Google Colab Secrets.
    """
    try:
        from google.colab import userdata
        os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
        os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')
    except ImportError:
        print("Warning: google.colab module not found. Assuming environment variables for Kaggle API are already set.")
    except Exception as e:
        print(f"Error accessing Colab secrets: {e}")

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    dataset_name = "aptos2019-blindness-detection"
    zip_path = os.path.join(RAW_DATA_DIR, f"{dataset_name}.zip")
    
    print(f"Downloading {dataset_name} to {RAW_DATA_DIR}...")
    try:
        subprocess.run([
            "kaggle", "competitions", "download", 
            "-c", dataset_name, 
            "-p", RAW_DATA_DIR
        ], check=True)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
            
        print("Cleaning up zip file...")
        os.remove(zip_path)
        
        print("Dataset downloaded and extracted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure your Kaggle credentials are correct and you have accepted the competition rules on Kaggle.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def run_eda():
    """
    Runs exploratory data analysis on the APTOS 2019 dataset.
    Generates plots, checks image statistics, and creates a summary markdown.
    """
    eda_dir = os.path.join(RESULTS_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Error: Could not find {TRAIN_CSV}. Please download the dataset first.")
        return
        
    df = pd.read_csv(TRAIN_CSV)
    
    # 1. Class Distribution
    print("\n--- Class Distribution ---")
    counts = df['diagnosis'].value_counts().sort_index()
    percentages = (counts / len(df)) * 100
    
    for idx, (count, pct) in enumerate(zip(counts, percentages)):
        print(f"Class {idx} ({CLASS_NAMES[idx]}): {count} images ({pct:.2f}%)")
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=CLASS_NAMES, y=counts.values, hue=CLASS_NAMES, palette="viridis", legend=False)
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Diagnosis")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Sample Images per Class
    print("\n--- Sample Images per Class ---")
    samples_per_class = 3
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(12, 3 * len(CLASS_NAMES)))
    
    for class_idx in range(len(CLASS_NAMES)):
        class_df = df[df['diagnosis'] == class_idx].sample(min(samples_per_class, counts[class_idx]), random_state=42)
        for i, (_, row) in enumerate(class_df.iterrows()):
            img_path = os.path.join(TRAIN_IMAGES_DIR, f"{row['id_code']}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[class_idx, i].imshow(img)
            axes[class_idx, i].set_title(f"Class {class_idx}: {CLASS_NAMES[class_idx]}")
            axes[class_idx, i].axis("off")
            
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "sample_images.png"), dpi=150)
    plt.close()
    
    # 3. Image Size Statistics & Duplicates
    print("\n--- Image Statistics & Quality Check ---")
    widths, heights = [], []
    corrupted = []
    hashes = defaultdict(list)
    
    for _, row in df.iterrows():
        img_name = f"{row['id_code']}.png"
        img_path = os.path.join(TRAIN_IMAGES_DIR, img_name)
        
        if not os.path.exists(img_path):
            corrupted.append(img_name)
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            corrupted.append(img_name)
            continue
            
        h, w, _ = img.shape
        heights.append(h)
        widths.append(w)
        
        with open(img_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        hashes[file_hash].append(img_name)
        
    print(f"Total valid images analyzed: {len(widths)}")
    if widths:
        print(f"Image Widths  - Min: {np.min(widths)}, Max: {np.max(widths)}, Mean: {np.mean(widths):.2f}")
        print(f"Image Heights - Min: {np.min(heights)}, Max: {np.max(heights)}, Mean: {np.mean(heights):.2f}")
    
    print(f"Corrupted/Missing files: {len(corrupted)}")
    if corrupted:
        print(f"Example corrupted files: {corrupted[:5]}")
        
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    print(f"Duplicate image groups found: {len(duplicates)}")
    if duplicates:
        print(f"Example duplicates: {list(duplicates.values())[:3]}")

    # 4. Generate EDA Summary
    imbalance_ratio = counts.max() / counts.min()
    summary = f"""# Dataset EDA Summary

**Imbalance Ratio**: The dataset is highly imbalanced. The majority class ({CLASS_NAMES[counts.idxmax()]}) is approximately {imbalance_ratio:.2f}x larger than the minority class ({CLASS_NAMES[counts.idxmin()]}).

## Limitations to Address
1. **Size**: The training set contains only ~{len(df)} images. Deep learning models typically require much more data to generalize well, necessitating robust data augmentation, transfer learning, or both.
2. **Image Quality Variation**: Fundus images exhibit significant differences in lighting, contrast, field of view, and artifacts (like camera artifacts or focus blur). Preprocessing steps like Graham's method (image blending) or CLAHE will be critical.
3. **Label Noise**: Medical imaging often suffers from inter-rater variability. Some images might be borderline or misclassified, which can hinder the model's ability to learn clear decision boundaries.
4. **Single-Source Bias**: Models trained strictly on APTOS data may fail to generalize to images captured using different fundus cameras or populations from different demographics without domain adaptation techniques.
"""
    print("\n--- EDA Markdown Summary ---")
    print(summary)
    
    with open(os.path.join(eda_dir, "eda_summary.md"), "w") as f:
        f.write(summary)

def compute_class_weights_from_train():
    """
    Computes class weights exclusively from the training split 
    to penalize the majority class and boost the minority classes.
    """
    train_csv = os.path.join(RESULTS_DIR, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"{train_csv} missing. Run data_loader.py first.")
        
    df = pd.read_csv(train_csv)
    counts = df['diagnosis'].value_counts().sort_index()
    total = len(df)
    num_classes = len(CLASS_NAMES)
    
    # Standard formula: weight = total_samples / (num_classes * class_samples)
    class_weights = {}
    print("\n--- Computed Class Weights (from TRAIN only) ---")
    for i in range(num_classes):
        weight = total / (num_classes * counts[i])
        class_weights[i] = weight
        print(f"Class {i} ({CLASS_NAMES[i]}): {weight:.4f}")
        
    return class_weights

def visualize_oversampling():
    """
    Visualizes the effect of minority-class oversampling on the training set
    as a comparison to the main class-weight strategy.
    """
    train_csv = os.path.join(RESULTS_DIR, "train.csv")
    if not os.path.exists(train_csv):
        return
        
    df = pd.read_csv(train_csv)
    
    # Original counts
    orig_counts = df['diagnosis'].value_counts().sort_index()
    max_count = orig_counts.max()
    
    # Simulate oversampling by matching the majority class
    oversampled_counts = [max_count] * len(CLASS_NAMES)
    
    plot_df = pd.DataFrame({
        'Diagnosis': CLASS_NAMES * 2,
        'Count': list(orig_counts.values) + oversampled_counts,
        'Strategy': ['Original (Imbalanced)'] * len(CLASS_NAMES) + ['Simulated Oversampling'] * len(CLASS_NAMES)
    })
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Diagnosis', y='Count', hue='Strategy', palette='Set2')
    plt.title("Class Balancing Strategy: Original vs Minority Oversampling")
    plt.ylabel("Number of Images")
    
    eda_dir = os.path.join(RESULTS_DIR, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    save_path = os.path.join(eda_dir, "oversampling_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved oversampling comparison chart to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis")
    parser.add_argument("--weights", action="store_true", help="Compute class weights and show oversampling chart")
    args = parser.parse_args()
    
    if args.download:
        download_aptos_dataset()
    if args.eda:
        run_eda()
    if args.weights:
        compute_class_weights_from_train()
        visualize_oversampling()
    if not (args.download or args.eda or args.weights):
        print("Please specify --download, --eda, or --weights")
