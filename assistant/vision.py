# vision.py
import cv2
from ultralytics import YOLO
from typing import List, Tuple, Optional, Any

from config import YOLO_MODEL_PATH, CAMERA_INDEX
from logging_utils import get_logger

logger = get_logger()
_yolo_model = None

def get_yolo_model() -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        logger.info(f"[VISION] Loading YOLO model: {YOLO_MODEL_PATH}")
        _yolo_model = YOLO(YOLO_MODEL_PATH)
    return _yolo_model

def capture_frame(camera_index: int = CAMERA_INDEX) -> Optional[Any]:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("[VISION] Cannot open webcam.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("[VISION] Failed to read frame from webcam.")
        return None
    return frame

def detect_objects_and_annotate(frame) -> Tuple[Any, List[str]]:
    """
    Run YOLO on frame, draw boxes and return (annotated_frame, object_names).
    """
    model = get_yolo_model()
    results = model(frame)[0]
    class_names = model.names

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, str(cls_id)) if isinstance(class_names, dict) else class_names[cls_id]
        detections.append(label)

        # Draw boxes
        x1, y1, x2, y2 = box.xyxy[0]
        conf = float(box.conf[0])
        cv2.rectangle(frame,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0),
                      2)
        cv2.putText(frame,
                    f"{label} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for d in detections:
        if d not in seen:
            seen.add(d)
            unique.append(d)

    logger.info(f"[VISION] Detected: {unique}")
    return frame, unique

# Backwards compatibility wrapper: some modules import detect_objects()
def detect_objects(frame) -> List[str]:
    """
    Return list of detected object names (no annotation).
    If callers expect annotation too, use detect_objects_and_annotate.
    """
    _, objects = detect_objects_and_annotate(frame)
    return objects
