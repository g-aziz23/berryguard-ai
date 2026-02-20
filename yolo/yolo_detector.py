from ultralytics import YOLO
import cv2

YOLO_MODEL_PATH = "model\\yolo_11_best_69_45.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)


confidences = []


def detect_and_draw(image_bgr, conf=0.15):

    results = yolo_model.predict(
        source=image_bgr,
        conf=conf,
        verbose=False
    )

    annotated = image_bgr.copy()
    confidences = []

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            conf_score = float(scores[i])
            confidences.append(conf_score)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    return annotated, confidences
