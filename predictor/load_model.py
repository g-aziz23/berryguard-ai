import os
import numpy as np
import keras

from predictor.utils.vit_backbone import ViTBackbone


# ===============================
# PATH CONFIG
# ===============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "qqhybrid_model.keras")

# ===============================
# SINGLETON MODEL
# ===============================

_model = None

def get_model():
    """
    Load model once and reuse (Django-safe)
    """
    global _model

    if _model is None:
        _model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "ViTBackbone": ViTBackbone
            },
            compile=False
        )
        _model.trainable = False

    return _model


def predict_image(img_array):
    """
    img_array: (H,W,3) or (1,H,W,3)
    returns: (class_index, confidence%)
    """
    model = get_model()

    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds) * 100)

    return class_idx, confidence
