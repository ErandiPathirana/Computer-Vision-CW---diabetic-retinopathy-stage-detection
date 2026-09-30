"""
Training module for Diabetic Retinopathy stage detection.
Handles the 2-phase training process, callbacks, history plotting, and hyperparameter experiments.
"""
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard,
)

from src.config import RESULTS_DIR, MODELS_DIR, LOGS_DIR, BEST_MODEL_PATH
from src.model import build_model, configure_phase_1, configure_phase_2
from src.augment import build_dataset
from src.data import compute_class_weights_from_train

TRAIN_RESULTS_DIR = os.path.join(RESULTS_DIR, "training")

def get_callbacks(model_name="best_model", phase_suffix="_phase1"):
    """
    Returns a comprehensive list of callbacks to monitor and control training.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    checkpoint_path = os.path.join(MODELS_DIR, f"{model_name}.keras")
    
    callbacks = [
        # EarlyStopping: Stops if validation loss doesn't improve for 6 epochs
        # restore_best_weights=True guarantees we keep the absolute best model
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        
        # ReduceLROnPlateau: Halves the LR if validation loss stalls for 3 epochs
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
        
        # ModelCheckpoint: Saves the best model weights dynamically
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
        
        # CSVLogger: Logs epoch-by-epoch metrics to a CSV file
        CSVLogger(os.path.join(LOGS_DIR, f"training_log{phase_suffix}.csv"), append=True),
        
        # TensorBoard: Enables rich visualizations of metrics and graphs
        TensorBoard(log_dir=os.path.join(LOGS_DIR, f"tb_logs{phase_suffix}"))
    ]
    return callbacks

def plot_history(history1, history2=None, save_prefix="combined"):
    """
    Plots training and validation metrics for Phase 1, Phase 2, or both combined.
    """
    os.makedirs(TRAIN_RESULTS_DIR, exist_ok=True)
    
    # Extract metrics
    acc = history1.history['accuracy']
    val_acc = history1.history['val_accuracy']
    loss = history1.history['loss']
    val_loss = history1.history['val_loss']
    
    # If a second history (Phase 2) is provided, append the metrics
    if history2 is not None:
        acc += history2.history['accuracy']
        val_acc += history2.history['val_accuracy']
        loss += history2.history['loss']
        val_loss += history2.history['val_loss']
        
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(16, 6))
    
    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
    
    # Add vertical dashed line to demarcate phases if combined
    if history2 is not None:
        phase1_epochs = len(history1.history['accuracy'])
        plt.axvline(x=phase1_epochs, color='gray', linestyle='--', label='Phase 2 Start')
        
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
    plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
    
    if history2 is not None:
        plt.axvline(x=phase1_epochs, color='gray', linestyle='--', label='Phase 2 Start')
        
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(TRAIN_RESULTS_DIR, f"{save_prefix}_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_prefix} history plot to {save_path}")

def train_pipeline(epochs_phase1=15, epochs_phase2=20):
    """
    Main training pipeline executing Phase 1 (head only) and Phase 2 (fine-tuning).
    """
    train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(RESULTS_DIR, "val.csv"))
    
    # Build tf.data datasets (augmentation applied exclusively to training set inside build_dataset)
    train_ds = build_dataset(train_df, is_training=True)
    val_ds = build_dataset(val_df, is_training=False)
    
    # Strictly calculated from training data to avoid leakage
    class_weights = compute_class_weights_from_train()
    
    model = build_model()
    
    # =========================================================================
    # PHASE 1: WARM UP THE HEAD
    # =========================================================================
    print("\n" + "="*50)
    print("STARTING PHASE 1: Training Classification Head")
    print("="*50)
    model = configure_phase_1(model)
    callbacks_p1 = get_callbacks(model_name="best_model_phase1", phase_suffix="_phase1")
    
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        class_weight=class_weights,
        callbacks=callbacks_p1
    )
    plot_history(history_p1, save_prefix="phase1")
    
    # =========================================================================
    # PHASE 2: FINE-TUNE THE BACKBONE
    # =========================================================================
    print("\n" + "="*50)
    print("STARTING PHASE 2: Fine-Tuning Top Backbone Layers")
    print("="*50)
    model = configure_phase_2(model, unfreeze_layers=30)
    callbacks_p2 = get_callbacks(model_name="best_model_finetuned", phase_suffix="_phase2")
    
    history_p2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        class_weight=class_weights,
        callbacks=callbacks_p2
    )
    plot_history(history_p2, save_prefix="phase2")
    
    # =========================================================================
    # COMBINED PLOTTING
    # =========================================================================
    plot_history(history_p1, history_p2, save_prefix="combined_phases")

    # =========================================================================
    # CANONICAL MODEL COPY
    # Ensure results/best_model.keras always points to the best fine-tuned
    # weights so that evaluate.py and app.py have a single reliable path.
    # =========================================================================
    finetuned_ckpt = os.path.join(MODELS_DIR, "best_model_finetuned.keras")
    if os.path.exists(finetuned_ckpt):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        shutil.copy2(finetuned_ckpt, BEST_MODEL_PATH)
        print(f"\nCanonical best model copied to {BEST_MODEL_PATH}")
    else:
        # Phase 2 didn't produce a checkpoint (e.g. val loss never improved);
        # fall back to Phase 1 checkpoint.
        phase1_ckpt = os.path.join(MODELS_DIR, "best_model_phase1.keras")
        if os.path.exists(phase1_ckpt):
            shutil.copy2(phase1_ckpt, BEST_MODEL_PATH)
            print(f"\nWarning: Phase 2 checkpoint not found. "
                  f"Canonical best model copied from Phase 1 → {BEST_MODEL_PATH}")

    print("\nTraining complete! Best models saved to", MODELS_DIR)


def run_hyperparameter_experiment():
    """
    Runs a small hyperparameter experiment over different learning rates 
    and dropout values. Logs results into a Markdown/CSV comparison table.
    """
    from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Input
    from tensorflow.keras.models import Model
    from src.model import get_backbone
    from src.config import IMG_SIZE, NUM_CLASSES, BACKBONE
    
    print("\n--- Starting Hyperparameter Experiment ---")
    train_df = pd.read_csv(os.path.join(RESULTS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(RESULTS_DIR, "val.csv"))
    
    train_ds = build_dataset(train_df, is_training=True)
    val_ds = build_dataset(val_df, is_training=False)
    class_weights = compute_class_weights_from_train()
    
    # Define experiment grid
    experiments = [
        {"lr": 1e-3, "dropout": 0.3},
        {"lr": 5e-4, "dropout": 0.3},
        {"lr": 1e-3, "dropout": 0.5},
    ]
    
    results = []
    
    for idx, exp in enumerate(experiments):
        print(f"\nExperiment {idx+1}/{len(experiments)}: LR={exp['lr']}, Dropout={exp['dropout']}")
        
        # Build custom model variant for this experiment
        inputs = Input(shape=(*IMG_SIZE, 3))
        base_model = get_backbone(BACKBONE, inputs)
        for layer in base_model.layers:
            layer.trainable = False
            
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(exp['dropout'])(x)
        outputs = Dense(NUM_CLASSES, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=exp['lr'])
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Run for a short number of epochs
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3, # Short test run
            class_weight=class_weights,
            verbose=1
        )
        
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])
        
        results.append({
            "Learning Rate": exp['lr'],
            "Dropout": exp['dropout'],
            "Best Val Loss": f"{best_val_loss:.4f}",
            "Best Val Acc": f"{best_val_acc:.4f}"
        })
        
    # Save comparison table
    results_df = pd.DataFrame(results)
    
    os.makedirs(TRAIN_RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(TRAIN_RESULTS_DIR, "hyperparameter_experiment.csv")
    md_path = os.path.join(TRAIN_RESULTS_DIR, "hyperparameter_experiment.md")
    
    results_df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write("# Hyperparameter Experiment Results\n\n")
        f.write("Tested on Phase 1 (head only) for 3 epochs each.\n\n")
        f.write(results_df.to_markdown(index=False))
        
    print(f"\nExperiment complete. Results saved to {md_path}")
    print("\n" + results_df.to_markdown(index=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Run full 2-phase training pipeline")
    parser.add_argument("--experiment", action="store_true", help="Run small hyperparameter experiment")
    args = parser.parse_args()
    
    if args.experiment:
        run_hyperparameter_experiment()
    if args.train:
        train_pipeline()
    if not (args.train or args.experiment):
        print("Please specify --train or --experiment")
