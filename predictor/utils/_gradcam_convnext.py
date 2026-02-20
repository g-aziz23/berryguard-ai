import tensorflow as tf
import numpy as np
import cv2

def gradcam_convnext(model, img_tensor, layer_name="layer_normalization"):
    """
    model      : loaded TriHybrid model
    img_tensor : (1,224,224,3) float32
    layer_name : last ConvNeXt feature layer
    return     : heatmap (224,224) float32
    """

    # ConvNeXt sub-model
    convnext = model.get_layer("convnext_tiny_backbone")

    target_layer = convnext.get_layer(layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_features, preds = grad_model(img_tensor)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_features)

    # Global Average Pooling on gradients
    weights = tf.reduce_mean(grads, axis=(1, 2))

    cam = tf.reduce_sum(
        tf.multiply(weights[:, None, None, :], conv_features),
        axis=-1,
    )

    cam = tf.nn.relu(cam)
    cam = cam[0].numpy()

    # Normalize
    cam = cam / (cam.max() + 1e-8)

    # Resize to input size
    cam = cv2.resize(cam, (224, 224))

    return cam
