import os
import re
import uuid
import cv2
import tensorflow as tf

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# =========================
# CNN PREPROCESS & PREDICT
# =========================
from predictor.utils.preprocess import preprocess_for_model
from predictor.load_model import predict_image, get_model

# =========================
# GRAD-CAM
# =========================
from predictor.utils.grad_cam import (
    generate_gradcam,
    overlay_gradcam
)

# =========================
# YOLO
# =========================
# detect_and_draw returns:
#   (annotated_image, confidence_list)
from yolo.yolo_detector import detect_and_draw


# =========================
# CONFIG
# =========================

CLASSES = [
    "Angular Leafspot",
    "Calcium Deficiency",
    "Healthy",
    "Leaf Scorch",
    "Leaf Spot",
    "Powdery Mildew",
]

CONF_THRESHOLD = 40.0


# =========================
# HELPERS
# =========================

def extract_true_class(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"\d+", "", name)
    name = name.replace("_", " ").lower()

    for cls in CLASSES:
        if cls.lower() in name:
            return cls
    return "Unknown"


def unique_name(prefix, ext="jpg"):
    return f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"


# =========================
# VIEW
# =========================

def home(request):

    # -------------------------
    # GET
    # -------------------------
    if request.method != "POST":
        return render(request, "predictor/home.html", {
            "uploaded": False,
            "predicted": False,
        })

    # -------------------------
    # Image upload
    # -------------------------
    image = request.FILES.get("image")
    if not image:
        return render(request, "predictor/home.html", {
            "uploaded": False,
            "predicted": False,
        })

    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    img_path = fs.path(filename)
    img_url = fs.url(filename)

    true_class = extract_true_class(image.name)

    # -------------------------
    # Read original image
    # -------------------------
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return render(request, "predictor/home.html", {
            "uploaded": False,
            "predicted": False,
        })

    # -------------------------
    # YOLO detection
    # -------------------------
    # IMPORTANT:
    # detect_and_draw must return (image, confidence_list)
    yolo_result, yolo_confidences = detect_and_draw(img_bgr)

    if yolo_confidences is None:
        yolo_confidences = []

    yolo_filename = unique_name("yolo")
    cv2.imwrite(fs.path(yolo_filename), yolo_result)
    yolo_url = fs.url(yolo_filename)

    # -------------------------
    # Preprocess for CNN
    # -------------------------
    img_model = preprocess_for_model(img_path)  # (1,224,224,3)

    # -------------------------
    # CNN Classification
    # -------------------------
    class_index, confidence = predict_image(img_model)

    if confidence < CONF_THRESHOLD:
        return render(request, "predictor/home.html", {
            "uploaded": True,
            "predicted": True,
            "input_img": img_url,
            "yolo_img": yolo_url,
            "grad_img": None,
            "class_name": "Not a Strawberry Leaf",
            "true_class": true_class,
            "confidence": round(confidence, 2),

            # 🔹 graph support (even if low confidence)
            "yolo_confidences": yolo_confidences,
        })

    predicted_class = CLASSES[class_index]

    # =========================
    # GRAD-CAM
    # =========================
    model = get_model()

    heatmap = generate_gradcam(
        model=model,
        img_array=tf.convert_to_tensor(img_model, dtype=tf.float32),
        class_index=class_index,
        layer_name=None
    )

    gradcam_img = overlay_gradcam(
        img_bgr,
        heatmap,
        alpha=0.55
    )

    grad_filename = unique_name("gradcam")
    cv2.imwrite(fs.path(grad_filename), gradcam_img)
    grad_url = fs.url(grad_filename)

    # -------------------------
    # Render result
    # -------------------------
    return render(request, "predictor/home.html", {
        "uploaded": True,
        "predicted": True,
        "input_img": img_url,
        "yolo_img": yolo_url,
        "grad_img": grad_url,
        "class_name": predicted_class,
        "true_class": true_class,
        "confidence": round(confidence, 2),

        # ✅ THIS feeds the bar/line graph in HTML
        # example: [0.91, 0.87, 0.63, 0.94]
        "yolo_confidences": yolo_confidences,
    })
