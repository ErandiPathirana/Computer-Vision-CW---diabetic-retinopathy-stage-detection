"""
Grad-CAM implementation for visual explanations.

Key design decisions
--------------------
* get_base_model() scans model.layers for the nested functional sub-model,
  so Grad-CAM works on models loaded from disk (model.base_model Python
  attribute is lost after tf.keras.models.load_model).
* Division is guarded with +1e-8 to avoid NaN when all activations are zero.
* PIL is used instead of the deprecated tf.keras.preprocessing.image API.
"""
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def get_base_model(model):
    """
    Returns the nested backbone sub-model from a DR classifier.

    Works whether model.base_model is set (freshly built) or not (loaded
    from disk). Scans model.layers for the first tf.keras.Model instance
    that is not the top-level model itself.

    Raises
    ------
    ValueError
        If no nested functional sub-model is found.
    """
    # Fast path: attribute set by build_model() in model.py
    if hasattr(model, "base_model") and model.base_model is not None:
        return model.base_model

    # Slow path: scan layers (needed after load_model)
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer is not model:
            return layer

    raise ValueError(
        "Could not locate a nested backbone sub-model inside the loaded model. "
        "Ensure the model was built with build_model() from src/model.py."
    )


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image array and model.

    Parameters
    ----------
    img_array : np.ndarray or tf.Tensor, shape (1, H, W, 3), float32, range 0-255
    model     : compiled Keras model (freshly built or loaded from disk)
    last_conv_layer_name : str
        Name of the final conv layer inside the backbone (e.g. "top_conv").
    pred_index : int or None
        Class index to explain. If None, uses argmax of predicted probabilities.

    Returns
    -------
    heatmap : np.ndarray, shape (h, w), float32, range 0-1
    """
    base_model = get_base_model(model)
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Sub-model: backbone input → (last conv output, backbone output)
    base_grad_model = tf.keras.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output],
    )

    # Reconstruct the classification head by name so it works post-load
    head_layer_names = ["head_gap", "head_bn", "head_dropout", "head_classifier"]
    head_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = head_input
    for layer in model.layers:
        if layer.name in head_layer_names:
            x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    # Compute gradients of the predicted class score w.r.t. last conv output
    with tf.GradientTape() as tape:
        last_conv_output, base_output = base_grad_model(img_array)
        tape.watch(last_conv_output)

        preds = head_model(base_output)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_output = last_conv_output[0]
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalise to [0, 1] — guard against all-zero activations (div-by-zero → NaN)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (max_val + 1e-8)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Superimposes the Grad-CAM heatmap onto the original image.

    Parameters
    ----------
    img_path : str  — path to the original (preprocessed) image
    heatmap  : np.ndarray, shape (h, w), float32, range 0-1
    alpha    : float — heatmap blending weight

    Returns
    -------
    PIL.Image.Image — the composite RGB image
    """
    # Load original image via PIL (avoids deprecated tf.keras.preprocessing.image API)
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

    # Scale heatmap to 0-255 and apply jet colormap
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]          # shape (256, 3)
    jet_heatmap = jet_colors[heatmap_uint8]            # shape (h, w, 3)

    # Resize heatmap to match original image dimensions using PIL
    jet_heatmap_pil = Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
        (img.shape[1], img.shape[0]), resample=Image.BILINEAR
    )
    jet_heatmap_resized = np.array(jet_heatmap_pil, dtype=np.float32)

    # Superimpose
    superimposed = jet_heatmap_resized * alpha + img
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed)
