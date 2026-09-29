"""
Data augmentation and tf.data pipeline module.
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    IMG_SIZE,
    BATCH_SIZE,
    PROCESSED_DATA_DIR,
    RESULTS_DIR
)

# =============================================================================
# WHY VALIDATION AND TEST SETS STAY UN-AUGMENTED
# =============================================================================
# Data augmentation is strictly a regularization technique used during training 
# to artificially expand the dataset and prevent the model from overfitting to 
# the training distribution.
# 
# Validation and test sets must remain entirely pristine and un-augmented because 
# they represent the real-world, untouched distribution of data that the model 
# will be evaluated against. Altering test/val data (e.g. adding blur, rotations)
# would give a distorted and clinically invalid measure of true model performance.
# =============================================================================

def data_augmentation_layer():
    """
    Keras Sequential model for data augmentation.
    Includes flips, rotations, zoom, and contrast jitter.
    
    Clinical Meaning Note: 
    We avoid extreme shear or drastic color shifts (like hue/saturation changes) 
    that might distort retinal structures (microaneurysms, hemorrhages, cotton wool spots) 
    and alter their clinical representation.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        # 20 degrees rotation is approx factor of 20/360 = 0.055
        tf.keras.layers.RandomRotation(factor=0.055, fill_mode='constant'),
        # Zoom 0.1
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode='constant'),
        tf.keras.layers.RandomContrast(factor=0.2),
        tf.keras.layers.RandomBrightness(factor=0.1)
    ], name="augmentation_pipeline")

def parse_image(file_path, label):
    """Reads and decodes a preprocessed image."""
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) # Keep in 0-255 range for EfficientNet
    return img, label

def build_dataset(df, is_training=False):
    """
    Builds a tf.data.Dataset from a DataFrame.
    """
    # Assuming images are already preprocessed and cached
    paths = [os.path.join(PROCESSED_DATA_DIR, "train_images", f"{x}.png") for x in df['id_code']]
    labels = df['diagnosis'].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
        
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    
    if is_training:
        aug_model = data_augmentation_layer()
        # Apply augmentation ONLY if is_training=True
        ds = ds.map(lambda x, y: (aug_model(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds

def show_augmented_examples():
    """Generates a figure showing an original image vs augmented versions."""
    train_csv = os.path.join(RESULTS_DIR, "train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found. Please run data_loader.py to split data.")
        return
        
    df = pd.read_csv(train_csv).head(1)
    img_path = os.path.join(PROCESSED_DATA_DIR, "train_images", f"{df['id_code'].iloc[0]}.png")
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found. Run preprocessing first.")
        return
        
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.expand_dims(img, 0) # Add batch dimension for the layer
    
    aug_model = data_augmentation_layer()
    
    plt.figure(figsize=(15, 6))
    
    # Original
    ax = plt.subplot(1, 5, 1)
    plt.imshow(img[0].numpy().astype("uint8"))
    plt.title("Original (Preprocessed)")
    plt.axis("off")
    
    # Augmented versions
    for i in range(2, 6):
        ax = plt.subplot(1, 5, i)
        aug_img = aug_model(img, training=True)
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title(f"Augmented {i-1}")
        plt.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.join(RESULTS_DIR, "augmentation"), exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, "augmentation", "augmented_examples.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved augmented examples to {save_path}")

if __name__ == "__main__":
    show_augmented_examples()
