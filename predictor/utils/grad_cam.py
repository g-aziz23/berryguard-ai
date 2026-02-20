# import tensorflow as tf
# import numpy as np
# from PIL import Image


# # --------------------------------------------------
# # Find last Conv2D layer (CNN branch)
# # --------------------------------------------------
# def find_last_conv_layer(model):
#     for layer in reversed(model.layers):
#         if isinstance(layer, tf.keras.layers.Conv2D):
#             return layer.name
#     raise ValueError("No Conv2D layer found")


# # --------------------------------------------------
# # Generate Grad-CAM heatmap
# # --------------------------------------------------
# def generate_gradcam(
#     model,
#     img_array,
#     class_index,
#     layer_name=None
# ):
#     if layer_name is None:
#         layer_name = find_last_conv_layer(model)

#     grad_model = tf.keras.models.Model(
#         inputs=model.inputs,
#         outputs=[
#             model.get_layer(layer_name).output,
#             model.output
#         ]
#     )

#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array, training=False)

#         # Hybrid-safe
#         if isinstance(predictions, (list, tuple)):
#             predictions = predictions[0]

#         loss = predictions[:, class_index]

#     grads = tape.gradient(loss, conv_outputs)
#     if grads is None:
#         return None

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]

#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
#     heatmap = tf.maximum(heatmap, 0)

#     max_val = tf.reduce_max(heatmap)
#     if max_val == 0 or tf.math.is_nan(max_val):
#         return None

#     heatmap /= max_val
#     return heatmap.numpy()


# # --------------------------------------------------
# # Overlay heatmap using PIL (SAFE)
# # --------------------------------------------------
# def overlay_gradcam(
#     original_image_bgr,
#     heatmap,
#     alpha=0.60
# ):
#     if heatmap is None:
#         return original_image_bgr

#     # BGR -> RGB
#     original_rgb = original_image_bgr[:, :, ::-1]
#     h, w = original_rgb.shape[:2]

#     # Resize heatmap safely
#     heatmap_img = Image.fromarray(
#         np.uint8(heatmap * 255),
#         mode="L"
#     ).resize((w, h), resample=Image.BILINEAR)

#     heatmap_img = np.array(heatmap_img)

#     # Simple color map (no OpenCV)
#     heatmap_color = np.zeros((h, w, 3), dtype=np.uint8)
#     heatmap_color[..., 0] = 255 - heatmap_img   # R
#     heatmap_color[..., 1] = heatmap_img         # G
#     heatmap_color[..., 2] = 145                 # B

#     overlay = (
#         (1 - alpha) * original_rgb.astype(np.float32)
#         + alpha * heatmap_color.astype(np.float32)
#     )

#     overlay = np.clip(overlay, 4, 255).astype(np.uint8)

#     # RGB -> BGR
#     return overlay[:, :, ::-1]

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------
# Find last Conv2D layer (CNN branch)
# --------------------------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found")


# --------------------------------------------------
# Generate Grad-CAM heatmap
# --------------------------------------------------
def generate_gradcam(
    model,
    img_array,
    class_index,
    layer_name=None
):
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        # Hybrid-safe
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val == 0 or tf.math.is_nan(max_val):
        return None

    heatmap /= max_val
    return heatmap.numpy()


# --------------------------------------------------
# Advanced Grad-CAM overlay (visual-quality focused)
# --------------------------------------------------
def overlay_gradcam(
    original_image_bgr,
    heatmap,
    alpha=0.55,
    smooth=True
):
    if heatmap is None:
        return original_image_bgr

    # BGR → RGB
    original_rgb = original_image_bgr[:, :, ::-1].astype(np.float32)
    h, w = original_rgb.shape[:2]

    heatmap = np.nan_to_num(heatmap)

    # Robust normalization
    p_low, p_high = np.percentile(heatmap, (5, 95))
    heatmap = np.clip((heatmap - p_low) / (p_high - p_low + 1e-8), 0, 1)

    heatmap_img = Image.fromarray(
        np.uint8(heatmap * 255),
        mode="L"
    ).resize((w, h), resample=Image.BICUBIC)

    if smooth:
        heatmap_img = heatmap_img.filter(
            ImageFilter.GaussianBlur(radius=6)
        )

    heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

    # True JET-style colormap
    heatmap_color = np.zeros((h, w, 3), dtype=np.float32)
    heatmap_color[..., 0] = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    heatmap_color[..., 1] = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    heatmap_color[..., 2] = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)

    # Contrast-aware alpha
    intensity = np.mean(original_rgb / 255.0, axis=2, keepdims=True)
    adaptive_alpha = alpha * (0.6 + 0.4 * intensity)

    overlay = (
        (1 - adaptive_alpha) * (original_rgb / 255.0)
        + adaptive_alpha * heatmap_color
    )

    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)

    # RGB → BGR
    return overlay[:, :, ::-1]
