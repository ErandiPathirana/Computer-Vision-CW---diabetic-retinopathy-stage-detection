"""
Grad-CAM implementation for visual explanations.
"""
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image array and model.
    Handles nested base_models robustly by isolating the backbone from the head.
    """
    base_model = model.base_model
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    # 1. Create a model mapping from backbone input to its last conv layer AND its output
    base_grad_model = tf.keras.Model(
        [base_model.inputs], 
        [last_conv_layer.output, base_model.output]
    )
    
    # 2. Reconstruct the custom classification head
    head_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = head_input
    
    # Assuming standard names given in src/model.py
    head_layer_names = ["head_gap", "head_bn", "head_dropout", "head_classifier"]
    for layer in model.layers:
        if layer.name in head_layer_names:
            x = layer(x)
            
    head_model = tf.keras.Model(head_input, x)
    
    # 3. Compute gradients of the top predicted class wrt the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_output = base_grad_model(img_array)
        tape.watch(last_conv_layer_output)
        
        preds = head_model(base_output)
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients and pool them
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """
    Superimposes the heatmap onto the original image and returns the composite image.
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)
    
    # Apply jet colormap
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Resize heatmap to match original image dimensions
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    
    return superimposed_img
