import numpy as np
import cv2


def predict_image(model, img_tensor):
    preds = model(img_tensor, training=False)
    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)
    return class_idx, confidence


def preprocess(img):
    """
    img: BGR or RGB image (H,W,3)
    returns: (1,224,224,3) float32
    """
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    return img[None, ...]
