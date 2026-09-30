"""
Smoke tests for the Diabetic Retinopathy pipeline.

Covers:
  - Model build (build_model)
  - preprocess_image on a synthetic 512x512 image
  - Grad-CAM on the same synthetic image (shape + no-NaN check)

Run with:
    python -m pytest tests/test_smoke.py -v
or:
    python tests/test_smoke.py
"""
import os
import sys
import tempfile
import numpy as np

# Ensure the repo root is on sys.path so `from src.X import Y` works
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_image_file(h: int = 512, w: int = 512) -> str:
    """
    Creates a temporary PNG file with a realistic synthetic retinal appearance:
    - Dark circular background (simulates fundus vignette)
    - Bright disc spot (simulates optic disc)
    This is brighter than a pure-black image, so the fundus-crop step in
    preprocess_image does not produce a zero-pixel crop.
    """
    import cv2

    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Simulate the retinal fundus with a bright greenish circle
    cv2.circle(img, (w // 2, h // 2), int(min(h, w) * 0.45), (50, 120, 50), thickness=-1)
    # Optic disc
    cv2.circle(img, (int(w * 0.6), h // 2), int(min(h, w) * 0.08), (200, 200, 150), thickness=-1)
    # Some microaneurysm-like dots
    for _ in range(30):
        cx = np.random.randint(int(w * 0.2), int(w * 0.8))
        cy = np.random.randint(int(h * 0.2), int(h * 0.8))
        cv2.circle(img, (cx, cy), 3, (180, 80, 80), thickness=-1)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    # cv2 writes BGR; convert RGB → BGR before saving
    cv2.imwrite(tmp.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Test 1: preprocess_image
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def setup_method(self):
        self.img_path = _make_synthetic_image_file(512, 512)

    def teardown_method(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_output_shape(self):
        from src.config import IMG_SIZE
        from src.preprocess import preprocess_image

        result = preprocess_image(self.img_path)

        assert result.shape == (*IMG_SIZE, 3), (
            f"Expected shape {(*IMG_SIZE, 3)}, got {result.shape}"
        )

    def test_output_dtype_uint8(self):
        from src.preprocess import preprocess_image

        result = preprocess_image(self.img_path)
        assert result.dtype == np.uint8, (
            f"Expected uint8 (0-255), got dtype={result.dtype}. "
            "EfficientNetB0 requires 0-255 inputs — do not normalise here."
        )

    def test_output_range_0_255(self):
        from src.preprocess import preprocess_image

        result = preprocess_image(self.img_path)
        assert result.min() >= 0 and result.max() <= 255, (
            f"Pixel range [{result.min()}, {result.max()}] out of expected [0, 255]."
        )

    def test_return_steps_length(self):
        from src.preprocess import preprocess_image

        steps = preprocess_image(self.img_path, return_steps=True)
        assert len(steps) == 6, f"Expected 6 pipeline steps, got {len(steps)}"


# ---------------------------------------------------------------------------
# Test 2: build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    def test_model_builds(self):
        import tensorflow as tf
        from src.model import build_model
        from src.config import IMG_SIZE, NUM_CLASSES

        model = build_model()
        assert model is not None

        # Input shape
        expected_input = (None, *IMG_SIZE, 3)
        assert tuple(model.input_shape) == expected_input, (
            f"Input shape mismatch: {model.input_shape} vs {expected_input}"
        )

        # Output shape (batch, NUM_CLASSES)
        expected_output = (None, NUM_CLASSES)
        assert tuple(model.output_shape) == expected_output, (
            f"Output shape mismatch: {model.output_shape} vs {expected_output}"
        )

    def test_base_model_attribute(self):
        from src.model import build_model
        import tensorflow as tf

        model = build_model()
        assert hasattr(model, "base_model"), "build_model() must set model.base_model"
        assert isinstance(model.base_model, tf.keras.Model)

    def test_phase1_freezes_backbone(self):
        from src.model import build_model, configure_phase_1
        import tensorflow as tf

        model = build_model()
        model = configure_phase_1(model)

        backbone_trainable = [
            l.trainable for l in model.base_model.layers
            if not isinstance(l, tf.keras.layers.InputLayer)
        ]
        assert all(not t for t in backbone_trainable), (
            "Phase 1: all backbone layers should be frozen"
        )

    def test_phase2_bn_stays_frozen(self):
        from src.model import build_model, configure_phase_1, configure_phase_2
        import tensorflow as tf

        model = build_model()
        model = configure_phase_1(model)
        model = configure_phase_2(model)

        for layer in model.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                assert not layer.trainable, (
                    f"Backbone BN layer '{layer.name}' should stay frozen in Phase 2"
                )


# ---------------------------------------------------------------------------
# Test 3: Grad-CAM
# ---------------------------------------------------------------------------

class TestGradCAM:
    def setup_method(self):
        self.img_path = _make_synthetic_image_file(512, 512)
        import tensorflow as tf
        from src.model import build_model, configure_phase_1
        self.model = build_model()
        self.model = configure_phase_1(self.model)

    def teardown_method(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_heatmap_shape(self):
        from src.preprocess import preprocess_image
        from src.gradcam import make_gradcam_heatmap
        from src.config import GRADCAM_LAYER

        img = preprocess_image(self.img_path).astype(np.float32)
        img_batch = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

        heatmap = make_gradcam_heatmap(img_batch, self.model, GRADCAM_LAYER)

        assert heatmap.ndim == 2, f"Heatmap should be 2-D, got shape {heatmap.shape}"
        assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0

    def test_heatmap_no_nan(self):
        from src.preprocess import preprocess_image
        from src.gradcam import make_gradcam_heatmap
        from src.config import GRADCAM_LAYER

        img = preprocess_image(self.img_path).astype(np.float32)
        img_batch = np.expand_dims(img, axis=0)

        heatmap = make_gradcam_heatmap(img_batch, self.model, GRADCAM_LAYER)
        assert not np.isnan(heatmap).any(), "Grad-CAM heatmap contains NaN values"

    def test_heatmap_range(self):
        from src.preprocess import preprocess_image
        from src.gradcam import make_gradcam_heatmap
        from src.config import GRADCAM_LAYER

        img = preprocess_image(self.img_path).astype(np.float32)
        img_batch = np.expand_dims(img, axis=0)

        heatmap = make_gradcam_heatmap(img_batch, self.model, GRADCAM_LAYER)
        assert heatmap.min() >= 0.0, "Heatmap minimum should be >= 0"
        assert heatmap.max() <= 1.0 + 1e-6, f"Heatmap maximum {heatmap.max()} should be <= 1"

    def test_save_and_display_gradcam_returns_pil(self):
        from PIL import Image as PILImage
        from src.preprocess import preprocess_image
        from src.gradcam import make_gradcam_heatmap, save_and_display_gradcam
        from src.config import GRADCAM_LAYER
        import cv2

        img = preprocess_image(self.img_path)
        img_batch = img.astype(np.float32)[np.newaxis]

        heatmap = make_gradcam_heatmap(img_batch, self.model, GRADCAM_LAYER)

        # Overwrite temp file with BGR version (as save_and_display_gradcam reads with PIL)
        cv2.imwrite(self.img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        result = save_and_display_gradcam(self.img_path, heatmap)
        assert isinstance(result, PILImage.Image), (
            f"save_and_display_gradcam should return PIL.Image, got {type(result)}"
        )
        assert result.size[0] > 0 and result.size[1] > 0

    def test_get_base_model_from_loaded(self):
        """Simulate a disk-loaded model (no .base_model attr) and verify get_base_model still works."""
        import tensorflow as tf
        from src.gradcam import get_base_model

        # Strip the attribute to simulate post-load_model state
        model_copy = self.model
        if hasattr(model_copy, "base_model"):
            delattr(model_copy, "base_model")

        base = get_base_model(model_copy)
        assert isinstance(base, tf.keras.Model)


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    suites = [TestPreprocessImage, TestBuildModel, TestGradCAM]
    passed = 0
    failed = 0

    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite) if m.startswith("test_")]
        for method_name in methods:
            suite.setup_method()
            try:
                getattr(suite, method_name)()
                print(f"  PASS  {suite_cls.__name__}::{method_name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}::{method_name}")
                traceback.print_exc()
                failed += 1
            finally:
                suite.teardown_method()

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
