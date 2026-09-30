"""
Prediction logic for the web app.

Uses exactly the same preprocessing (src/preprocess.py) and Grad-CAM (src/gradcam.py)
as training and evaluation, so the app sees the same inputs as the model was trained on.
TensorFlow is imported lazily, so the rest of the app (and its tests) work without it.
"""
from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src import config as C
from src.preprocess import preprocess_image

MAX_PIXELS = 60_000_000          # refuse absurdly large images (memory protection)

TRIAGE = {
    0: ("Routine", "No signs of diabetic retinopathy detected. Routine re-screening at the normal interval."),
    1: ("Monitor", "Mild changes. Closer follow-up and repeat screening; risk factors should be reviewed with a clinician."),
    2: ("Refer", "Moderate changes. Ophthalmology review is recommended."),
    3: ("Refer soon", "Severe changes. Prompt referral to an ophthalmologist is recommended."),
    4: ("Urgent", "Proliferative disease suspected. Urgent specialist referral is recommended."),
}


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB uint8 array (validates type and size)."""
    if not (data[:8] == b"\x89PNG\r\n\x1a\n" or data[:3] == b"\xff\xd8\xff"):
        raise ValueError("Only PNG or JPEG images are accepted.")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The file could not be read as an image.")
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        raise ValueError("The image is too small (minimum 64x64 pixels).")
    if h * w > MAX_PIXELS:
        raise ValueError("The image is too large.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def assess_quality(rgb: np.ndarray) -> List[str]:
    """Advisory checks only (never block the prediction)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    warnings = []
    if gray.mean() < 20:
        warnings.append("The image is very dark; the result may be unreliable.")
    if gray.mean() > 200:
        warnings.append("The image is over-exposed; the result may be unreliable.")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
        warnings.append("The image looks blurry or low in detail; the result may be unreliable.")
    if (gray < C.BLACK_THRESHOLD).mean() < 0.02 and gray.std() < 25:
        warnings.append("This may not be a retinal fundus photograph.")
    return warnings


def png_b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode image.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def heatmap_rgb(heat: np.ndarray) -> np.ndarray:
    """Colour-coded heat-map (JET) as RGB uint8, for the opacity slider in the UI."""
    return cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * np.clip(heat, 0, 1)), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


class Predictor:
    """Loads the trained model once and answers prediction requests (thread-safe)."""

    def __init__(self, model_path: Optional[str] = None,
                 runner: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray, int]]] = None):
        """runner(img_uint8) -> (heatmap, probabilities, class index). Injected in tests; by default the real model."""
        self.model_path = Path(model_path or os.environ.get("MODEL_PATH", "") or
                               (C.EXPORT_MODEL_PATH if C.EXPORT_MODEL_PATH.exists() else C.BEST_MODEL_PATH))
        self.class_names = list(C.CLASS_NAMES)
        self._runner = runner
        self._model = None
        self._lock = threading.Lock()
        self.load_error: Optional[str] = None
        if runner is not None:
            self.load_error = None

    # ---- loading ---------------------------------------------------------- #
    @property
    def loaded(self) -> bool:
        return self._runner is not None or self._model is not None

    def load(self) -> None:
        """Load the model; on failure keep the server running and remember why."""
        if self._runner is not None:
            return
        if not self.model_path.exists():
            self.load_error = (f"Model file not found: {self.model_path}. Run the Colab notebook, download export.zip "
                               "and unzip it into the project folder so that models/best_model.keras exists.")
            return
        try:
            import keras
            from src.gradcam import make_gradcam
            self._model = keras.models.load_model(self.model_path)
            self._runner = lambda img: make_gradcam(self._model, img)
            names_file = C.MODELS_DIR / "class_names.json"
            if names_file.exists():
                self.class_names = json.loads(names_file.read_text())
            self.load_error = None
        except Exception as exc:                                   # keep the UI usable and explain the problem
            self.load_error = (f"The model could not be loaded ({type(exc).__name__}: {exc}). Install the same "
                               "TensorFlow version that was used for training in Colab.")

    def info(self) -> Dict:
        return {"model_loaded": self.loaded, "model_path": str(self.model_path), "message": self.load_error}

    # ---- prediction ------------------------------------------------------- #
    def predict(self, data: bytes) -> Dict:
        if not self.loaded:
            raise RuntimeError(self.load_error or "The model is not loaded.")
        rgb = decode_image(data)
        steps = preprocess_image(rgb, return_steps=True)            # [(name, image), ...]
        final = steps[-1][1]                                        # exactly what the model receives
        with self._lock:                                            # one prediction at a time
            heat, probs, cls = self._runner(final)
        probs = np.asarray(probs, dtype=float)
        stage = int(np.argmax(probs))
        conf = float(probs[stage])
        triage, text = TRIAGE[stage]
        overlay = cv2.addWeighted(final, 0.6, heatmap_rgb(heat), 0.4, 0)
        images = {name.lower(): png_b64(img) for name, img in steps}
        images["heatmap"] = png_b64(heatmap_rgb(heat))
        images["overlay"] = png_b64(overlay)
        return {
            "stage": stage,
            "stage_name": self.class_names[stage],
            "confidence": conf,
            "probabilities": [{"name": self.class_names[i], "probability": float(p)} for i, p in enumerate(probs)],
            "low_confidence": conf < C.LOW_CONFIDENCE_THRESHOLD,
            "threshold": C.LOW_CONFIDENCE_THRESHOLD,
            "triage": triage,
            "action": text,
            "action_note": "Illustrative only. Not clinical advice.",
            "quality_warnings": assess_quality(rgb),
            "images": images,
        }
