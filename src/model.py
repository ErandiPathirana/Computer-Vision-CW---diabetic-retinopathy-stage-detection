"""
Model architecture module.
Builds the selected backbone and handles the 2-phase fine-tuning logic.
"""
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50, MobileNetV2

from src.config import IMG_SIZE, NUM_CLASSES, BACKBONE

def get_backbone(backbone_name, input_tensor):
    """
    Returns the selected backbone model loaded with ImageNet weights.
    """
    if backbone_name == "EfficientNetB0":
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_tensor=input_tensor)
    elif backbone_name == "ResNet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)
    elif backbone_name == "MobileNetV2":
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
        
    return base_model

def build_model(backbone_name=BACKBONE):
    """
    Builds the complete model with the swappable backbone and a custom classification head.
    """
    inputs = Input(shape=(*IMG_SIZE, 3))
    
    # Load the base model and treat it as a sub-model (keeps layers grouped nicely)
    base_model = get_backbone(backbone_name, inputs)
    
    # Add custom head on top of the base model output
    x = base_model.output
    x = GlobalAveragePooling2D(name="head_gap")(x)
    x = BatchNormalization(name="head_bn")(x)
    x = Dropout(0.3, name="head_dropout")(x)
    outputs = Dense(NUM_CLASSES, activation='softmax', name="head_classifier")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f"DR_Classifier_{backbone_name}")
    
    # Store reference to base_model for easier freezing/unfreezing logic
    model.base_model = base_model
    return model

def configure_phase_1(model):
    """
    Configures Phase 1: Freezes the base model, compiles to train only the head.
    Learning rate = 1e-3.
    """
    # Freeze the entire base model
    for layer in model.base_model.layers:
        layer.trainable = False
        
    # Ensure head layers are trainable
    head_layer_names = ["head_gap", "head_bn", "head_dropout", "head_classifier"]
    for layer in model.layers:
        if layer.name in head_layer_names:
            layer.trainable = True
            
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("\n--- Model Configured for Phase 1 (Head Only, lr=1e-3) ---")
    return model

def configure_phase_2(model, unfreeze_layers=30):
    """
    Configures Phase 2: Unfreezes the top N non-BN layers of the backbone for fine-tuning.

    BatchNorm freezing rationale
    ----------------------------
    Only the *backbone's* BatchNormalization layers are kept frozen here.
    Frozen backbone BN layers use their ImageNet-trained running statistics
    (mean/variance) as fixed constants, which is critical when fine-tuning
    on a small medical dataset — updating them would corrupt the statistics
    that the backbone's weights depend on.

    The head's `head_bn` layer IS intentionally left trainable (set above)
    because it was freshly initialised and must adapt to the new feature
    distribution. This is correct and not a contradiction of the BN-freeze rule.

    Learning rate = 1e-5 (10x smaller than Phase 1 to avoid catastrophic forgetting).
    """
    # Keep head layers trainable (including head_bn — see docstring)
    head_layer_names = ["head_gap", "head_bn", "head_dropout", "head_classifier"]
    for layer in model.layers:
        if layer.name in head_layer_names:
            layer.trainable = True

    # Unfreeze the top N layers of the backbone, skipping BatchNorm layers
    unfrozen_count = 0
    for layer in reversed(model.base_model.layers):
        if unfrozen_count >= unfreeze_layers:
            layer.trainable = False
            continue

        # Keep backbone BatchNorm layers frozen to preserve ImageNet statistics
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            unfrozen_count += 1

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"\n--- Model Configured for Phase 2 (Fine-tuning top {unfreeze_layers} backbone layers, lr=1e-5) ---")
    return model

def print_model_parameters(model):
    """
    Prints a breakdown of trainable vs frozen parameters.
    """
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Total Parameters:      {total_params:,}")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"Frozen Parameters:     {non_trainable_params:,}")
    
def display_summary():
    """
    Showcase the model building and parameter changes across phases.
    """
    print(f"Building model with {BACKBONE} backbone...")
    model = build_model()
    
    # Phase 1
    model = configure_phase_1(model)
    print_model_parameters(model)
    
    # Phase 2
    model = configure_phase_2(model, unfreeze_layers=30)
    print_model_parameters(model)
    
    print("\n--- Full Model Summary (Phase 2 state) ---")
    model.summary()

if __name__ == "__main__":
    display_summary()
