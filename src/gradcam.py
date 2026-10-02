"""
Grad-CAM: shows which retinal regions drove the prediction.

The model is  preprocess -> backbone (nested model) -> GAP -> BN -> Dropout -> Dense.
Because the backbone is a nested model, we cannot ask Keras for an intermediate
layer output of the outer graph. Instead we run the pieces by hand inside a
GradientTape: the backbone output IS the last convolutional feature map
(7x7x1280 for EfficientNetB0), so we take gradients of the class score with
respect to it. This works on a model freshly built OR loaded from disk.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import keras

from . import config as C


from .model import get_base_model  # noqa: E402,F401  (re-exported: the backbone is the nested Model inside the DR model)


def make_gradcam(model: keras.Model, img_uint8: np.ndarray, class_idx: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute the Grad-CAM heat-map.

    Args:
        model: trained DR model.
        img_uint8: preprocessed RGB image, uint8 (IMG_SIZE x IMG_SIZE x 3), values 0-255.
        class_idx: class to explain; defaults to the predicted class.
    Returns:
        heatmap in [0, 1] resized to the image size, class probabilities, explained class.
    """
    x = tf.convert_to_tensor(img_uint8[None].astype("float32"))
    pre = model.get_layer("preprocess")
    backbone = get_base_model(model)
    gap, bn = model.get_layer("gap"), model.get_layer("head_bn")
    drop, out = model.get_layer("dropout"), model.get_layer("predictions")

    with tf.GradientTape() as tape:
        conv = backbone(pre(x), training=False)          # last conv feature map (1, h, w, channels)
        tape.watch(conv)
        h = drop(bn(gap(conv), training=False), training=False)
        probs = out(h)
        if class_idx is None:
            class_idx = int(tf.argmax(probs[0]))
        score = probs[:, class_idx]
    grads = tape.gradient(score, conv)                    # d score / d feature map
    weights = tf.reduce_mean(grads, axis=(1, 2))          # importance of each channel
    cam = tf.reduce_sum(conv * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)                        # +eps: no division by zero (all-zero map -> zeros)
    cam = np.nan_to_num(cam)
    cam = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(cam, 0.0, 1.0), probs[0].numpy(), int(class_idx)


def overlay_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a JET-coloured heat-map over the image (RGB uint8 in and out)."""
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)
