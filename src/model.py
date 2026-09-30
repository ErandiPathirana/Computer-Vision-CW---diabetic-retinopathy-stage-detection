"""
CNN with transfer learning.

Why EfficientNetB0?  It reaches strong ImageNet accuracy with only ~4M parameters
(compound scaling of depth/width/resolution), so it trains quickly on a free Colab
T4 GPU and is small enough (~17 MB) to ship inside the web app. ResNet50 (larger,
~24M) and MobileNetV2 (smaller, faster, less accurate) are compared in
src/experiments.py.

Transfer learning in two phases:
  Phase 1 - backbone frozen, only the new head (GAP -> BatchNorm -> Dropout -> Dense)
            is trained with a high learning rate (1e-3).
  Phase 2 - the top N backbone layers are unfrozen and fine-tuned with a very small
            learning rate (1e-5); BatchNorm layers stay frozen so their ImageNet
            statistics are not destroyed by small batches.
"""
from __future__ import annotations

from typing import Dict

import keras
import numpy as np

from . import config as C

BACKBONES = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "mobilenetv2": keras.applications.MobileNetV2,
    "resnet50": keras.applications.ResNet50,
}


def _preprocess_layer(backbone_name: str) -> keras.layers.Layer:
    """Input scaling that each backbone expects (input images are float 0-255)."""
    if backbone_name == "mobilenetv2":                       # expects [-1, 1]
        return keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")
    if backbone_name == "resnet50":                          # expects BGR, mean-subtracted
        return keras.layers.Lambda(lambda x: keras.applications.resnet50.preprocess_input(x), name="preprocess")
    return keras.layers.Rescaling(1.0, name="preprocess")    # EfficientNet rescales internally


def build_model(backbone_name: str = "efficientnetb0", dropout: float = C.DROPOUT, weights: str | None = "imagenet",
                num_classes: int = C.NUM_CLASSES) -> keras.Model:
    """Backbone (named 'backbone') + classification head.

    Layer names are fixed ('preprocess', 'backbone', 'gap', 'head_bn', 'dropout',
    'predictions') because Grad-CAM and the web app look layers up by name.
    """
    base = BACKBONES[backbone_name](include_top=False, weights=weights, input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3),
                                    name=C.BACKBONE_LAYER_NAME)
    base.trainable = False
    inputs = keras.Input(shape=(C.IMG_SIZE, C.IMG_SIZE, 3), name="image")
    x = _preprocess_layer(backbone_name)(inputs)
    x = base(x, training=False)          # training=False keeps BatchNorm in inference mode, also during fine-tuning
    x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = keras.layers.BatchNormalization(name="head_bn")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    return keras.Model(inputs, outputs, name=f"dr_{backbone_name}")


def get_base_model(model: keras.Model) -> keras.Model:
    """Return the pretrained sub-model (works after load_model too)."""
    return model.get_layer(C.BACKBONE_LAYER_NAME)


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def configure_phase1(model: keras.Model, lr: float = C.HEAD_LR) -> None:
    """Freeze the whole backbone; train only the head."""
    get_base_model(model).trainable = False
    compile_model(model, lr)


def configure_phase2(model: keras.Model, unfreeze_layers: int = C.UNFREEZE_LAYERS, lr: float = C.FINETUNE_LR) -> None:
    """Unfreeze the last `unfreeze_layers` backbone layers (except BatchNorm) and recompile."""
    base = get_base_model(model)
    base.trainable = True
    for layer in base.layers[:-unfreeze_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, lr)


def parameter_counts(model: keras.Model) -> Dict[str, int]:
    """Trainable / frozen / total parameters (for the report)."""
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    total = int(model.count_params())
    return {"trainable": trainable, "frozen": total - trainable, "total": total}
