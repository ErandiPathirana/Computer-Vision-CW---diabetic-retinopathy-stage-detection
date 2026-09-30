"""
Simple Gradio demo that runs inside Colab (public share link) - a fallback for the video.
Uses exactly the same preprocessing and Grad-CAM code as training and evaluation.

    !python app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import keras
import numpy as np

from src import config as C
from src.gradcam import make_gradcam, overlay_heatmap
from src.preprocess import preprocess_image

_model = None


def _get_model():
    """Load the trained model once (from models/ if exported, else results/)."""
    global _model
    if _model is None:
        path = C.EXPORT_MODEL_PATH if C.EXPORT_MODEL_PATH.exists() else C.BEST_MODEL_PATH
        _model = keras.models.load_model(path)
    return _model


def predict(image: np.ndarray):
    """image: RGB uint8 array from Gradio -> (label probabilities, Grad-CAM overlay)."""
    img = preprocess_image(image)
    heat, probs, _ = make_gradcam(_get_model(), img)
    label = {C.CLASS_NAMES[i]: float(probs[i]) for i in range(C.NUM_CLASSES)}
    return label, overlay_heatmap(img, heat)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Fundus photograph"),
    outputs=[gr.Label(num_top_classes=5, label="DR stage"), gr.Image(label="Grad-CAM")],
    title="Diabetic Retinopathy stage detection (educational prototype)",
    description="Educational prototype. Not a medical device. Not for clinical use.",
)

if __name__ == "__main__":
    demo.launch(share=True)
