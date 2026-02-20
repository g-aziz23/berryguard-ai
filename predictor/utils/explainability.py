import numpy as np
import cv2

from predictor.utils._gradcam_effnet import gradcam_effnet
from predictor.utils._gradcam_convnext import gradcam_convnext


def generate_final_fusion_overlay(model, img_tensor, class_idx, orig_rgb):
    cam1 = gradcam_effnet(model, img_tensor, class_idx)
    cam2 = gradcam_convnext(model, img_tensor, class_idx)

    cam = (cam1 + cam2) / 2.0
    cam = cv2.resize(cam, (orig_rgb.shape[1], orig_rgb.shape[0]))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(orig_rgb, 0.6, heatmap, 0.4, 0)
    return overlay
