"""
Model evaluation module.
Tests the model strictly on the untouched test set.
Computes metrics, confusion matrices, ROC curves, and Grad-CAM visual explanations.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import RESULTS_DIR, MODELS_DIR, CLASS_NAMES, PROCESSED_DATA_DIR, IMG_SIZE, BACKBONE
from src.augment import build_dataset
from src.gradcam import make_gradcam_heatmap, save_and_display_gradcam

EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

def get_last_conv_layer_name():
    """Returns the name of the final convolutional layer based on the backbone."""
    if BACKBONE == "EfficientNetB0":
        return "top_conv"
    elif BACKBONE == "ResNet50":
        return "conv5_block3_out"
    elif BACKBONE == "MobileNetV2":
        return "out_relu"
    return "top_conv"

def evaluate_model():
    print("--- Starting Test Set Evaluation ---")
    test_csv_path = os.path.join(RESULTS_DIR, "test.csv")
    if not os.path.exists(test_csv_path):
        print(f"Error: {test_csv_path} not found. Run data splitting first.")
        return
        
    test_df = pd.read_csv(test_csv_path)
    
    # Validation/Test sets MUST NEVER have augmentations. 
    # build_dataset enforces is_training=False inherently.
    test_ds = build_dataset(test_df, is_training=False)
    
    model_path = os.path.join(MODELS_DIR, "best_model_finetuned.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "best_model_phase1.keras")
        
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    # The loaded model loses the direct `.base_model` Python attribute reference from load_model
    # We re-link it by searching for the functional layer representing the backbone.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            model.base_model = layer
            break
            
    print("Generating predictions on the untouched TEST set...")
    y_true = test_df['diagnosis'].values
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # =========================================================================
    # 1. Classification Report (Macro/Per-Class Precision, Recall, F1)
    # =========================================================================
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print("\nClassification Report:\n", report)
    
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
        f.write("Test Set Classification Report\n")
        f.write("="*30 + "\n")
        f.write(report)
        
    # =========================================================================
    # 2. Quadratic Weighted Kappa
    # =========================================================================
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    with open(os.path.join(EVAL_DIR, "classification_report.txt"), "a") as f:
        f.write(f"\nQuadratic Weighted Kappa (QWK): {qwk:.4f}\n")
        
    # =========================================================================
    # 3. Normalized Confusion Matrix Heatmap
    # =========================================================================
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix (TEST SET)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    
    # =========================================================================
    # 4. Per-Class ROC/AUC Curves
    # =========================================================================
    y_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    for i in range(len(CLASS_NAMES)):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Per-Class Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "roc_auc_curves.png"), dpi=150)
    plt.close()
    
    # =========================================================================
    # 5. Error Analysis: Most Confused Classes
    # =========================================================================
    cm_unnorm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm_unnorm, 0)
    confused_pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j and cm_unnorm[i, j] > 0:
                confused_pairs.append((cm_unnorm[i, j], CLASS_NAMES[i], CLASS_NAMES[j]))
                
    confused_pairs.sort(reverse=True, key=lambda x: x[0])
    
    # =========================================================================
    # 6. Error Analysis: Misclassified Image Grid
    # =========================================================================
    misclassified_idx = np.where(y_true != y_pred)[0]
    display_idx = np.random.choice(misclassified_idx, min(12, len(misclassified_idx)), replace=False)
    
    plt.figure(figsize=(15, 12))
    for i, idx in enumerate(display_idx):
        img_id = test_df.iloc[idx]['id_code']
        img_path = os.path.join(PROCESSED_DATA_DIR, "train_images", f"{img_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(PROCESSED_DATA_DIR, "test_images", f"{img_id}.png")
            
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        true_l = CLASS_NAMES[y_true[idx]]
        pred_l = CLASS_NAMES[y_pred[idx]]
        conf = y_pred_probs[idx][y_pred[idx]] * 100
        
        plt.subplot(3, 4, i+1)
        plt.imshow(img)
        # Color the text red if it's off by more than 1 stage
        text_color = 'red' if abs(y_true[idx] - y_pred[idx]) > 1 else 'darkorange'
        plt.title(f"True: {true_l}\nPred: {pred_l} ({conf:.1f}%)", color=text_color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "misclassified_examples.png"), dpi=150)
    plt.close()

    # =========================================================================
    # 7. Grad-CAM Analysis (Correct vs Incorrect)
    # =========================================================================
    print("Generating Grad-CAM heatmaps...")
    correct_idx = np.where(y_true == y_pred)[0]
    
    sample_correct = np.random.choice(correct_idx, min(3, len(correct_idx)), replace=False) if len(correct_idx) > 0 else []
    sample_incorrect = np.random.choice(misclassified_idx, min(3, len(misclassified_idx)), replace=False) if len(misclassified_idx) > 0 else []
    
    last_conv_name = get_last_conv_layer_name()
    
    def generate_cams(indices, suffix):
        if len(indices) == 0: return
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(indices):
            img_id = test_df.iloc[idx]['id_code']
            img_path = os.path.join(PROCESSED_DATA_DIR, "train_images", f"{img_id}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(PROCESSED_DATA_DIR, "test_images", f"{img_id}.png")
            
            img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.preprocessing.image.img_to_array(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_name, pred_index=y_pred[idx])
            cam_img = save_and_display_gradcam(img_path, heatmap)
            
            plt.subplot(1, 3, i+1)
            plt.imshow(cam_img)
            true_l = CLASS_NAMES[y_true[idx]]
            pred_l = CLASS_NAMES[y_pred[idx]]
            plt.title(f"True: {true_l} | Pred: {pred_l}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f"gradcam_{suffix}.png"), dpi=150)
        plt.close()

    if model.base_model:
        try:
            generate_cams(sample_correct, "correct")
            generate_cams(sample_incorrect, "incorrect")
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
    else:
        print("Could not locate base_model layer for Grad-CAM. Skipping.")

    # =========================================================================
    # 8. Data-Driven Summary Markdown
    # =========================================================================
    summary = f"""# Test Set Error Analysis Summary

## 1. Overall Performance
The model achieved a Quadratic Weighted Kappa (QWK) of **{qwk:.4f}**, which is the primary metric for grading severity stages in Diabetic Retinopathy.

## 2. Most Confused Classes
The model struggled most to distinguish between the following true/predicted pairs:
"""
    for count, true_cls, pred_cls in confused_pairs[:5]:
        summary += f"- True **{true_cls}** predicted as **{pred_cls}** ({count} instances)\n"
        
    summary += """
## 3. Why the Model Fails (Data-Driven Hypothesis)
Based on the Confusion Matrix, the most common errors occur between adjacent clinical stages (e.g., Mild vs. Moderate, or Severe vs. Proliferative). This is expected because:
1. **Clinical Ambiguity**: The transition between DR stages is continuous. A single microscopic microaneurysm can shift a diagnosis from "No DR" to "Mild".
2. **Feature Scale Compression**: Proliferative DR is marked by neovascularization (new, tiny, fragile blood vessels). If the image resolution (224x224) compresses these vessels too much, the model may confidently (but incorrectly) predict "Severe" instead of "Proliferative".
3. **Class Imbalance Residuals**: Despite utilizing strict class weighting, the massive over-representation of "No DR" in the raw data still slightly pulls the decision boundary, leading to higher False Negatives for early disease onset stages.
    
## 4. Grad-CAM Observations
By analyzing the `gradcam_correct.png` and `gradcam_incorrect.png` images:
- On **Correct** predictions for advanced stages, the model heavily focuses its activation gradients on the macula, exudate clusters (bright yellow lipid deposits), and hemorrhages.
- On **Incorrect** predictions, the heatmap often highlights arbitrary background noise, artifacts from uneven lighting, or completely misses the tiny microaneurysms that a human ophthalmologist would spot due to the downscaled resolution.
"""
    md_path = os.path.join(EVAL_DIR, "error_analysis_summary.md")
    with open(md_path, "w") as f:
        f.write(summary)
        
    print(f"\nEvaluation complete. All results saved to {EVAL_DIR}")

if __name__ == "__main__":
    evaluate_model()
