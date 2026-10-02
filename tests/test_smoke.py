"""
Smoke tests (run with:  python -m pytest tests -q   or   python -m unittest discover tests).
Tests that need TensorFlow are skipped automatically when it is not installed.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C            # noqa: E402
from src import data as D              # noqa: E402
from src import preprocess as P        # noqa: E402

HAS_TF = importlib.util.find_spec("tensorflow") is not None


def fake_fundus(h=480, w=640, seed=0):
    """A synthetic 'fundus' image: black frame, orange disc, bright spot, light noise."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 2 - 10, (190, 110, 50), -1)
    cv2.circle(img, (w // 2 + 60, h // 2 - 20), 25, (240, 200, 120), -1)
    noise = rng.randint(0, 6, img.shape, dtype=np.uint8)
    return np.where(img > 0, cv2.add(img, noise), img)


class TestConfig(unittest.TestCase):
    def test_paths_are_relative_to_repo(self):
        self.assertTrue(C.REPO_ROOT.exists())
        self.assertEqual(C.BEST_MODEL_PATH.parent, C.RESULTS_DIR)
        self.assertEqual(len(C.CLASS_NAMES), 5)

    def test_split_ratios_sum_to_one(self):
        self.assertAlmostEqual(sum(C.SPLIT_RATIOS), 1.0)


class TestPreprocess(unittest.TestCase):
    def test_output_shape_dtype_range(self):
        out = P.preprocess_image(fake_fundus())
        self.assertEqual(out.shape, (C.IMG_SIZE, C.IMG_SIZE, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertGreater(out.max(), 1)          # must be 0-255, not 0-1 (no double normalisation)

    def test_crop_removes_black_frame(self):
        img = np.zeros((300, 300, 3), np.uint8)
        img[100:200, 50:250] = 128
        self.assertEqual(P.crop_black_borders(img).shape[:2], (100, 200))

    def test_black_image_is_not_cropped_to_nothing(self):
        self.assertEqual(P.crop_black_borders(np.zeros((50, 50, 3), np.uint8)).shape, (50, 50, 3))

    def test_steps_returned(self):
        steps = P.preprocess_image(fake_fundus(), return_steps=True)
        self.assertEqual([n for n, _ in steps], ["Original", "Cropped", "Denoised", "CLAHE", "Sharpened"])

    def test_deterministic(self):
        img = fake_fundus()
        np.testing.assert_array_equal(P.preprocess_image(img), P.preprocess_image(img))

    def test_bytes_pipeline_matches_array_pipeline(self):
        img = fake_fundus()
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.testing.assert_array_equal(P.preprocess_bytes(buf.tobytes()), P.preprocess_image(img))

    def test_unreadable_bytes_raise(self):
        with self.assertRaises(ValueError):
            P.preprocess_bytes(b"not an image")

    def test_model_input_is_float_0_255(self):
        x = P.to_model_input(P.preprocess_image(fake_fundus()))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape, (1, C.IMG_SIZE, C.IMG_SIZE, 3))


class TestData(unittest.TestCase):
    def _fake_dataset(self, tmp: Path):
        img_dir = tmp / "imgs"
        img_dir.mkdir()
        rows = []
        for cls, n in enumerate([30, 10, 15, 8, 8]):
            for i in range(n):
                name = f"c{cls}_{i}"
                cv2.imwrite(str(img_dir / f"{name}.png"), cv2.cvtColor(fake_fundus(120, 160, i), cv2.COLOR_RGB2BGR))
                rows.append((name, cls))
        return pd.DataFrame(rows, columns=["id_code", "diagnosis"]), img_dir

    def test_validate_and_split_have_no_leakage(self):
        with tempfile.TemporaryDirectory() as t:
            df, img_dir = self._fake_dataset(Path(t))
            clean = D.validate_dataset(df, img_dir)
            tr, va, te = D.make_splits(clean, out_dir=Path(t) / "splits")
            self.assertEqual(len(tr) + len(va) + len(te), len(clean))
            self.assertFalse(set(tr.id_code) & set(te.id_code))
            self.assertFalse(set(tr.id_code) & set(va.id_code))
            self.assertEqual(set(tr.diagnosis), set(range(5)))    # stratified: every class in train

    def test_one_class_csv_is_rejected_and_train_csv_preferred(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            pd.DataFrame({"id_code": list("abc"), "diagnosis": [0, 0, 0]}).to_csv(root / "sample_submission.csv", index=False)
            with self.assertRaises(RuntimeError):
                D._find_label_csv([root])
            pd.DataFrame({"id_code": list("abc"), "diagnosis": [0, 1, 2]}).to_csv(root / "train.csv", index=False)
            self.assertEqual(D._find_label_csv([root]).name, "train.csv")
        with self.assertRaises(ValueError):
            D.validate_dataset(pd.DataFrame({"id_code": ["a", "b"], "diagnosis": [0, 0]}), check_readable=False)

    def test_csv_without_labels_gives_clear_error(self):
        with self.assertRaises(ValueError):
            D.validate_dataset(pd.DataFrame({"id_code": ["a", "b"]}))

    def test_bad_label_range_rejected(self):
        with self.assertRaises(ValueError):
            D.validate_dataset(pd.DataFrame({"id_code": ["a"], "diagnosis": [9]}))


@unittest.skipUnless(HAS_TF, "TensorFlow not installed")
class TestModelAndGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.model import build_model
        cls.model = build_model("efficientnetb0", weights=None)     # no download needed for tests

    def test_output_shape_and_softmax(self):
        x = np.random.randint(0, 255, (2, C.IMG_SIZE, C.IMG_SIZE, 3)).astype("float32")
        out = self.model.predict(x, verbose=0)
        self.assertEqual(out.shape, (2, 5))
        np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-4)

    def test_phase_freezing(self):
        import keras
        from src.model import configure_phase1, configure_phase2, get_base_model
        configure_phase1(self.model)
        self.assertFalse(get_base_model(self.model).trainable)
        configure_phase2(self.model, unfreeze_layers=30)
        base = get_base_model(self.model)
        self.assertTrue(any(l.trainable for l in base.layers))
        self.assertFalse(any(l.trainable for l in base.layers if isinstance(l, keras.layers.BatchNormalization)))

    def test_gradcam_shape_range_and_no_nan(self):
        from src.gradcam import make_gradcam, overlay_heatmap
        img = P.preprocess_image(fake_fundus())
        heat, probs, cls = make_gradcam(self.model, img)
        self.assertEqual(heat.shape, (C.IMG_SIZE, C.IMG_SIZE))
        self.assertFalse(np.isnan(heat).any())
        self.assertTrue(0.0 <= heat.min() and heat.max() <= 1.0)
        self.assertEqual(len(probs), 5)
        self.assertEqual(overlay_heatmap(img, heat).shape, img.shape)

    def test_save_and_reload_gives_same_prediction_and_gradcam_works(self):
        import keras
        from src.gradcam import make_gradcam
        img = P.preprocess_image(fake_fundus())
        x = P.to_model_input(img)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "m.keras"
            self.model.save(path)
            loaded = keras.models.load_model(path)
        np.testing.assert_allclose(self.model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-4)
        heat, _, _ = make_gradcam(loaded, img)          # simulates the web app loading from disk
        self.assertEqual(heat.shape, (C.IMG_SIZE, C.IMG_SIZE))


if __name__ == "__main__":
    unittest.main()
