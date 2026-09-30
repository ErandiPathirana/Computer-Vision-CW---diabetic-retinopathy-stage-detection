"""End-to-end check on synthetic images: preprocess -> tf.data -> predict -> Grad-CAM."""
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
from src import preprocess as P        # noqa: E402

HAS_TF = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_TF, "TensorFlow not installed")
class TestPipelineEndToEnd(unittest.TestCase):
    def test_dataset_pipeline_and_prediction(self):
        from src.augment import check_pixel_range, compute_class_weights, make_dataset, oversample_minorities
        from src.gradcam import make_gradcam
        from src.model import build_model
        rng = np.random.RandomState(1)
        old = C.PROCESSED_DIR
        with tempfile.TemporaryDirectory() as t:
            C.PROCESSED_DIR = Path(t) / "proc"
            raw = Path(t) / "raw"
            raw.mkdir()
            rows = []
            for i in range(20):
                img = np.zeros((200, 260, 3), np.uint8)
                cv2.circle(img, (130, 100), 90, (int(rng.randint(120, 220)), 100, 50), -1)
                cv2.imwrite(str(raw / f"i{i}.png"), img)
                rows.append((f"i{i}", str(raw / f"i{i}.png"), i % 5))
            df = pd.DataFrame(rows, columns=["id_code", "path", "diagnosis"])
            df = P.cache_preprocessed(df, workers=2)
            weights = compute_class_weights(df)
            self.assertEqual(set(weights), set(range(5)))
            self.assertGreater(len(oversample_minorities(df)), len(df) - 1)
            ds = make_dataset(df, training=True, batch_size=4)
            check_pixel_range(ds)
            model = build_model(weights=None)
            probs = model.predict(make_dataset(df, training=False, batch_size=4), verbose=0)
            self.assertEqual(probs.shape, (20, 5))
            heat, _, _ = make_gradcam(model, P.load_rgb(df.iloc[0]["proc_path"]))
            self.assertEqual(heat.shape, (C.IMG_SIZE, C.IMG_SIZE))
        C.PROCESSED_DIR = old


if __name__ == "__main__":
    unittest.main()
