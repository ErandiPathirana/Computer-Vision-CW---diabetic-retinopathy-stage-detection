"""
Web app tests. They use a stub model, so they run without TensorFlow and without a trained model.
    python -m pytest tests -q        (or: python -m unittest discover tests)
"""
import http.client
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp import inference                      # noqa: E402
from webapp.server import create_server           # noqa: E402

HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


def fundus_png(h=300, w=400) -> bytes:
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 2 - 8, (190, 110, 50), -1)
    cv2.circle(img, (w // 2 + 40, h // 2), 18, (240, 200, 120), -1)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def stub_runner(img):
    """Pretends to be the model: class 2 with 70% probability and a centred heat-map."""
    heat = np.zeros(img.shape[:2], np.float32)
    heat[60:160, 60:160] = 1.0
    return heat, np.array([0.05, 0.1, 0.7, 0.1, 0.05]), 2


class TestInference(unittest.TestCase):
    def test_quality_and_decode(self):
        with self.assertRaises(ValueError):
            inference.decode_image(b"hello world, not an image")
        with self.assertRaises(ValueError):
            inference.decode_image(b"\x89PNG\r\n\x1a\n" + b"garbage")
        rgb = inference.decode_image(fundus_png())
        self.assertEqual(rgb.shape[2], 3)
        self.assertTrue(any("very dark" in w for w in inference.assess_quality(np.zeros((100, 100, 3), np.uint8))))

    def test_predict_result_structure(self):
        res = inference.Predictor(runner=stub_runner).predict(fundus_png())
        self.assertEqual(res["stage"], 2)
        self.assertEqual(res["stage_name"], "Moderate")
        self.assertAlmostEqual(sum(p["probability"] for p in res["probabilities"]), 1.0, places=5)
        self.assertFalse(res["low_confidence"])
        for key in ("original", "cropped", "denoised", "clahe", "sharpened", "heatmap", "overlay"):
            self.assertIn(key, res["images"])
        self.assertIn("Illustrative", res["action_note"])

    def test_low_confidence_flag(self):
        flat = lambda img: (np.zeros(img.shape[:2]), np.array([0.3, 0.25, 0.2, 0.15, 0.1]), 0)
        self.assertTrue(inference.Predictor(runner=flat).predict(fundus_png())["low_confidence"])

    def test_missing_model_is_reported_not_crashing(self):
        p = inference.Predictor(model_path="does/not/exist.keras")
        p.load()
        self.assertFalse(p.loaded)
        self.assertIn("not found", p.load_error)


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / "assets").mkdir()
        (root / "samples").mkdir()
        (root / "models").mkdir()
        (root / "assets" / "note.json").write_text('{"ok": true}')
        (root / "samples" / "s1.png").write_bytes(fundus_png())
        (root / "models" / "metrics.json").write_text(json.dumps({"accuracy": 0.9}))
        cls.server = create_server("127.0.0.1", 0, inference.Predictor(runner=stub_runner),
                                   root / "assets", root / "samples", root / "models")
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.tmp.cleanup()

    def call(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=90)
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        data = r.read()
        conn.close()
        return r.status, r.getheader("Content-Type", ""), data

    def test_pages_and_health(self):
        st, ct, body = self.call("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn("text/html", ct)
        st, _, body = self.call("GET", "/api/health")
        self.assertTrue(json.loads(body)["model_loaded"])

    def test_predict_ok(self):
        st, _, body = self.call("POST", "/api/predict", fundus_png(), {"Content-Type": "image/png"})
        self.assertEqual(st, 200)
        self.assertEqual(json.loads(body)["stage_name"], "Moderate")

    def test_predict_rejects_wrong_type_and_empty_and_large(self):
        st, _, body = self.call("POST", "/api/predict", b"just text", {"Content-Type": "text/plain"})
        self.assertEqual(st, 400)
        self.assertIn("PNG or JPEG", json.loads(body)["error"])
        self.assertEqual(self.call("POST", "/api/predict", b"", {"Content-Type": "image/png"})[0], 400)
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (11 * 1024 * 1024)
        self.assertEqual(self.call("POST", "/api/predict", big, {"Content-Type": "image/png"})[0], 413)

    def test_assets_samples_metrics(self):
        self.assertEqual(json.loads(self.call("GET", "/api/assets")[2]), ["note.json"])
        self.assertEqual(self.call("GET", "/api/assets/note.json")[0], 200)
        self.assertEqual(json.loads(self.call("GET", "/api/samples")[2]), ["s1.png"])
        self.assertEqual(self.call("GET", "/api/samples/s1.png")[0], 200)
        self.assertEqual(json.loads(self.call("GET", "/api/metrics")[2])["accuracy"], 0.9)
        self.assertEqual(self.call("GET", "/api/assets/missing.png")[0], 404)

    def test_path_traversal_is_blocked(self):
        for bad in ("/api/assets/../models/metrics.json", "/api/assets/..%2fmodels%2fmetrics.json",
                    "/api/assets/%2e%2e/secret", "/api/samples/..\\x.png", "/api/assets/.hidden"):
            st = self.call("GET", bad)[0]
            self.assertIn(st, (400, 404), bad)

    @unittest.skipUnless(HAS_REPORTLAB, "reportlab not installed")
    def test_report_pdf(self):
        res = inference.Predictor(runner=stub_runner).predict(fundus_png())
        st, ct, body = self.call("POST", "/api/report", json.dumps(res).encode(), {"Content-Type": "application/json"})
        self.assertEqual(st, 200)
        self.assertEqual(ct, "application/pdf")
        self.assertTrue(body.startswith(b"%PDF"))
        self.assertEqual(self.call("POST", "/api/report", b'{"x": 1}', {"Content-Type": "application/json"})[0], 400)

    def test_unknown_route(self):
        self.assertEqual(self.call("GET", "/nope")[0], 404)


if __name__ == "__main__":
    unittest.main()
