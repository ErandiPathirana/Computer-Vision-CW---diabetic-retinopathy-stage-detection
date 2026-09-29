"""
Preprocessing pipeline for fundus images.
Includes cropping, resizing, CLAHE, denoising, and sharpening.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from src.config import (
    IMG_SIZE,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR
)

PREPROC_RESULTS_DIR = os.path.join(RESULTS_DIR, "preprocessing")

def crop_fundus(img, tol=7):
    """
    Crops the black borders around the fundus circle.
    Creates a mask for pixels above the tolerance and finds the bounding box.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    
    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # If the image is completely dark (highly unlikely), return original
    if not np.any(rows) or not np.any(cols):
        return img
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    return img[ymin:ymax+1, xmin:xmax+1]

def apply_clahe(img):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to the L channel in LAB color space to improve contrast.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # clipLimit=2.0 and tileGridSize=(8,8) are standard values for fundus imaging
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def apply_unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.5):
    """
    Applies an unsharp mask to enhance edges (blood vessels, exudates).
    """
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened

def preprocess_image(img_path, return_steps=False):
    """
    Executes the full preprocessing pipeline on a single image.
    
    Normalization Strategy:
    tf.keras.applications.EfficientNetB0 (and other EfficientNets in Keras)
    includes its own internal Rescaling layer. It expects inputs strictly 
    in the [0, 255] range. 
    Therefore, we explicitly KEEP the output as uint8 in the [0, 255] range.
    Applying standard /255.0 normalization here would cause double-scaling
    and severely damage model performance.
    """
    # Read image in BGR and convert to RGB
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Crop
    cropped = crop_fundus(img_rgb)
    
    # 2. Resize
    resized = cv2.resize(cropped, IMG_SIZE)
    
    # 3. CLAHE
    clahe = apply_clahe(resized)
    
    # 4. Mild Gaussian Denoising
    denoised = cv2.GaussianBlur(clahe, (3, 3), 0)
    
    # 5. Optional unsharp-mask edge enhancement
    sharpened = apply_unsharp_mask(denoised)
    
    if return_steps:
        return img_rgb, cropped, resized, clahe, denoised, sharpened
    
    # 6. Return (No /255.0 scaling, see docstring)
    return sharpened

def generate_comparison_figures(sample_paths):
    """
    Generates before/after comparison figures and contrast histograms.
    """
    os.makedirs(PREPROC_RESULTS_DIR, exist_ok=True)
    
    for i, img_path in enumerate(sample_paths):
        try:
            img_rgb, cropped, resized, clahe, denoised, sharpened = preprocess_image(img_path, return_steps=True)
            
            fig = plt.figure(figsize=(16, 10))
            
            # Show original
            ax1 = plt.subplot(2, 4, 1)
            ax1.imshow(img_rgb)
            ax1.set_title("1. Original")
            ax1.axis('off')
            
            # Show Cropped & Resized
            ax2 = plt.subplot(2, 4, 2)
            ax2.imshow(resized)
            ax2.set_title("2. Cropped & Resized")
            ax2.axis('off')
            
            # Show CLAHE
            ax3 = plt.subplot(2, 4, 3)
            ax3.imshow(clahe)
            ax3.set_title("3. CLAHE Applied")
            ax3.axis('off')
            
            # Show Sharpened (Final)
            ax4 = plt.subplot(2, 4, 4)
            ax4.imshow(sharpened)
            ax4.set_title("4. Denoised & Sharpened")
            ax4.axis('off')
            
            # Histogram for Original (Resized)
            ax5 = plt.subplot(2, 4, 6)
            gray_res = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
            ax5.hist(gray_res.ravel(), 256, [0, 256], color='#1f77b4', alpha=0.7)
            ax5.set_title("Histogram (Before CLAHE)")
            ax5.set_xlim(0, 255)
            
            # Histogram for CLAHE (Final)
            ax6 = plt.subplot(2, 4, 7)
            gray_clahe = cv2.cvtColor(clahe, cv2.COLOR_RGB2GRAY)
            ax6.hist(gray_clahe.ravel(), 256, [0, 256], color='#ff7f0e', alpha=0.7)
            ax6.set_title("Histogram (After CLAHE)")
            ax6.set_xlim(0, 255)
            
            plt.suptitle(f"Preprocessing Pipeline Comparison: {os.path.basename(img_path)}", fontsize=16)
            plt.tight_layout()
            
            save_path = os.path.join(PREPROC_RESULTS_DIR, f"preprocessing_steps_{i+1}.png")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"Saved comparison figure to {save_path}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

def process_and_cache_dataset(input_dir, output_dir):
    """
    Preprocesses all images in the input directory and caches them.
    This saves significant computation time during model training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img_paths = glob(os.path.join(input_dir, "*.png"))
    print(f"Found {len(img_paths)} images in {input_dir}. Starting preprocessing...")
    
    for img_path in tqdm(img_paths, desc="Preprocessing Images"):
        filename = os.path.basename(img_path)
        save_path = os.path.join(output_dir, filename)
        
        # Skip if already processed to allow resuming
        if os.path.exists(save_path):
            continue
            
        try:
            processed_img = preprocess_image(img_path)
            # OpenCV writes in BGR, so we must convert RGB back to BGR before saving
            processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, processed_img_bgr)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    print(f"Finished preprocessing and caching to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Generate comparison figures for a few sample images")
    parser.add_argument("--process-all", action="store_true", help="Preprocess and cache all training/testing images")
    args = parser.parse_args()
    
    train_input_dir = os.path.join(RAW_DATA_DIR, "train_images")
    train_output_dir = os.path.join(PROCESSED_DATA_DIR, "train_images")
    
    if args.compare:
        all_imgs = glob(os.path.join(train_input_dir, "*.png"))
        if len(all_imgs) >= 3:
            import random
            # Set a fixed seed to always grab the same images for comparison
            random.seed(42)
            sample_paths = random.sample(all_imgs, 3)
            generate_comparison_figures(sample_paths)
        else:
            print("Not enough images found in raw data directory to run comparisons.")
            
    if args.process_all:
        process_and_cache_dataset(train_input_dir, train_output_dir)
        
        test_input_dir = os.path.join(RAW_DATA_DIR, "test_images")
        test_output_dir = os.path.join(PROCESSED_DATA_DIR, "test_images")
        if os.path.exists(test_input_dir):
            process_and_cache_dataset(test_input_dir, test_output_dir)

    if not (args.compare or args.process_all):
        print("Please provide --compare or --process-all")
