import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32")
    return img[None, ...]


import cv2
import numpy as np

IMG_SIZE = 224

def preprocess_for_model(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = img[None, ...]   # (1,224,224,3)
    return img

import cv2
import numpy as np

IMG_SIZE = 224


def preprocess_for_model(img_path):
    """
    For classification model (EffNet + ConvNeXt + ViT)
    Output: (1, 224, 224, 3), float32
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = img[None, ...]
    return img


def preprocess_for_display(img_path):
    """
    For UI / Grad-CAM overlay base
    Output: RGB uint8 image (H,W,3)
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
