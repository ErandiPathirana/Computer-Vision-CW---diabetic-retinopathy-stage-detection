"""
Gradio web interface for Diabetic Retinopathy Stage Detection.
"""
import os
import tempfile
import numpy as np
import tensorflow as tf
import gradio as gr
import cv2
from PIL import Image

from src.config import MODELS_DIR, CLASS_NAMES
from src.preprocess import preprocess_image
from src.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from src.evaluate import get_last_conv_layer_name

# =========================================================================
# Model Loading
# =========================================================================
MODEL_PATH = os.path.join(MODELS_DIR, "best_model_finetuned.keras")
if not os.path.exists(MODEL_PATH):
    # Fallback to Phase 1 model if Phase 2 hasn't run
    MODEL_PATH = os.path.join(MODELS_DIR, "best_model_phase1.keras")

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Re-link the base_model explicitly for Grad-CAM
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            model.base_model = layer
            break
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# =========================================================================
# Inference & Explainability Pipeline
# =========================================================================
def predict_and_explain(input_image):
    """
    Takes a numpy image from Gradio, routes it through the exact 
    preprocessing pipeline, predicts the class, and generates a Grad-CAM overlay.
    """
    if model is None:
        return "Model not found. Please train the model first.", {}, None
        
    if input_image is None:
        return "No image provided.", {}, None
        
    # Save input image to a temporary file to maintain exact pipeline parity 
    # with how cv2.imread is used in our preprocessing logic.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        # input_image is an RGB numpy array from Gradio
        Image.fromarray(input_image).save(temp_path)
        
    try:
        # 1. Apply exact reproducible preprocessing (Crop, CLAHE, Denoise, Unsharp Mask)
        processed_img_array = preprocess_image(temp_path)
        
        # 2. Prepare tensor (EfficientNet expects uint8 0-255 inputs)
        img_tensor = np.expand_dims(processed_img_array, axis=0)
        img_tensor = tf.cast(img_tensor, tf.float32)
        
        # 3. Model Inference
        preds = model.predict(img_tensor)[0]
        pred_idx = np.argmax(preds)
        predicted_class = CLASS_NAMES[pred_idx]
        
        # 4. Confidence scores for all 5 classes
        confidences = {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))}
        
        # 5. Grad-CAM Interpretation
        last_conv_name = get_last_conv_layer_name()
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_name, pred_index=pred_idx)
        
        # Save temp image again in BGR so our Grad-CAM function can superimpose properly
        cv2.imwrite(temp_path, cv2.cvtColor(processed_img_array, cv2.COLOR_RGB2BGR))
        cam_img_pil = save_and_display_gradcam(temp_path, heatmap)
        
    except Exception as e:
        return f"Error during processing: {e}", {}, None
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return predicted_class, confidences, cam_img_pil

# =========================================================================
# Gradio UI Construction
# =========================================================================
custom_css = """
.disclaimer {
    background-color: #ffebee;
    color: #b71c1c;
    border: 1px solid #ffcdd2;
    padding: 15px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
"""

with gr.Blocks(title="DR Stage Detection System", css=custom_css) as demo:
    gr.Markdown("# 👁️ Diabetic Retinopathy Stage Detection")
    
    gr.HTML(
        "<div class='disclaimer'>⚠️ EDUCATIONAL PROTOTYPE DISCLAIMER: This application was developed strictly for a "
        "university Computer Vision coursework. It is NOT a medical device and must never be used for actual clinical diagnosis.</div>"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Raw Fundus Image")
            input_img = gr.Image(label="Fundus Image", type="numpy")
            submit_btn = gr.Button("Analyze Image", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 2. Detection Results")
            output_class = gr.Textbox(label="Predicted Clinical Stage", text_align="center")
            output_conf = gr.Label(label="Confidence Breakdown (Softmax Probabilities)")
            
        with gr.Column(scale=1):
            gr.Markdown("### 3. Interpretability (Grad-CAM)")
            gr.Markdown("*Heatmap highlights regions driving the model's decision (e.g. exudates, microaneurysms).*")
            output_cam = gr.Image(label="Grad-CAM Overlay")
            
    # Wire the interactions
    submit_btn.click(
        fn=predict_and_explain,
        inputs=input_img,
        outputs=[output_class, output_conf, output_cam]
    )

if __name__ == "__main__":
    # share=True is mandatory for rendering the Gradio UI from a Google Colab cell
    print("Launching Gradio App. A public URL will be generated...")
    demo.launch(share=True, debug=True)
