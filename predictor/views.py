# import os
# import re
# import uuid
# import base64
# import requests

# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage


# # ✅ Correct HF endpoint (with trailing slash)
# HF_API_URL = "https://abdulaziz368-berryguard-ai.hf.space/api/predict/"


# CLASSES = [
#     "Angular Leafspot",
#     "Calcium Deficiency",
#     "Healthy",
#     "Leaf Scorch",
#     "Leaf Spot",
#     "Powdery Mildew",
# ]


# def extract_true_class(filename):
#     name = os.path.splitext(filename)[0]
#     name = re.sub(r"\d+", "", name)
#     name = name.replace("_", " ").lower()

#     for cls in CLASSES:
#         if cls.lower() in name:
#             return cls
#     return "Unknown"


# def unique_name(prefix, ext="jpg"):
#     return f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"


# def home(request):

#     if request.method != "POST":
#         return render(request, "predictor/home.html", {
#             "uploaded": False,
#             "predicted": False,
#         })

#     image = request.FILES.get("image")

#     if not image:
#         return render(request, "predictor/home.html", {
#             "uploaded": False,
#             "predicted": False,
#         })

#     fs = FileSystemStorage()
#     filename = fs.save(image.name, image)
#     img_path = fs.path(filename)
#     img_url = fs.url(filename)

#     true_class = extract_true_class(image.name)

#     # 🔹 Convert image to pure base64 (NO prefix)
#     with open(img_path, "rb") as f:
#         image_bytes = f.read()

#     image_base64 = base64.b64encode(image_bytes).decode("utf-8")

#     payload = {
#         "data": [image_base64]
#     }

#     try:
#         response = requests.post(HF_API_URL, json=payload, timeout=60)

#         print("HF STATUS:", response.status_code)
#         print("HF RESPONSE:", response.text)

#         result = response.json()

#         predicted_class = result["data"][0]
#         confidence = result["data"][1]
#         gradcam_image = result["data"][2]

#     except Exception as e:
#         print("ERROR:", str(e))
#         predicted_class = "Prediction Error"
#         confidence = 0
#         gradcam_image = None

#     return render(request, "predictor/home.html", {
#         "uploaded": True,
#         "predicted": True,
#         "input_img": img_url,
#         "class_name": predicted_class,
#         "true_class": true_class,
#         "confidence": round(float(confidence), 2),
#         "grad_img": gradcam_image,
#         "yolo_confidences": [],
#     })

import os
import re
import base64
import requests

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage


# ✅ Correct HF endpoint (Gradio 6.x)
HF_API_URL = "https://abdulaziz368-berryguard-ai.hf.space/gradio_api/run/predict"


CLASSES = [
    "Angular Leafspot",
    "Calcium Deficiency",
    "Healthy",
    "Leaf Scorch",
    "Leaf Spot",
    "Powdery Mildew",
]


def extract_true_class(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r"\d+", "", name)
    name = name.replace("_", " ").lower()

    for cls in CLASSES:
        if cls.lower() in name:
            return cls
    return "Unknown"


def home(request):

    if request.method != "POST":
        return render(request, "predictor/home.html")

    image = request.FILES.get("image")

    if not image:
        return render(request, "predictor/home.html")

    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    img_path = fs.path(filename)
    img_url = fs.url(filename)

    true_class = extract_true_class(image.name)

    # 🔹 Convert image to base64 with data prefix
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "data": [
            {
                "path": None,
                "url": f"data:image/jpeg;base64,{image_base64}",
                "orig_name": image.name,
                "meta": {"_type": "gradio.FileData"}
            }
        ]
    }

    try:
        response = requests.post(HF_API_URL, json=payload, timeout=120)

        print("STATUS:", response.status_code)
        print("TEXT:", response.text)

        if response.status_code != 200:
            raise Exception("HF API failed")

        result = response.json()["data"]

        predicted_class = result[0]
        confidence = float(result[1])
        gradcam_image = result[2]

    except Exception as e:
        print("ERROR:", str(e))
        predicted_class = "Prediction Error"
        confidence = 0
        gradcam_image = None

    return render(request, "predictor/home.html", {
        "input_img": img_url,
        "class_name": predicted_class,
        "confidence": round(confidence, 2),
        "grad_img": gradcam_image,
        "true_class": true_class,
    })