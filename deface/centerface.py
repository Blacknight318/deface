import os
import numpy as np
import cv2
from edgetpu.detection.engine import DetectionEngine

# Find file relative to the location of this code file
default_model_path = os.path.join(os.path.dirname(__file__), 'centerface_edgetpu.tflite')


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert input image to RGB if it is in RGBA or L format"""
    if img.ndim == 2:  # 1-channel grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # 4-channel RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


class CenterFace:
    def __init__(self, model_path=None, in_shape=None):
        self.in_shape = in_shape

        if model_path is None:
            model_path = default_model_path

        self.engine = DetectionEngine(model_path)

    def __call__(self, img, threshold=0.5):
        img = ensure_rgb(img)
        self.orig_shape = img.shape[:2]
        if self.in_shape is None:
            self.in_shape = self.orig_shape[::-1]

        resized_img = cv2.resize(img, self.in_shape)
        rgb_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb_resized_img, axis=0)

        detections = self.engine.detect_with_input_tensor(input_tensor, threshold=threshold, keep_aspect_ratio=True,
                                                          relative_coord=False, top_k=100)

        boxes, landmarks = [], []
        for detection in detections:
            box = detection.bounding_box.flatten().tolist()
            landmark = detection.landmarks.flatten().tolist()
            boxes.append(box)
            landmarks.append(landmark)

        if len(boxes) > 0:
            boxes = np.asarray(boxes, dtype=np.float32)
            landmarks = np.asarray(landmarks, dtype=np.float32)
        else:
            boxes = np.empty(shape=[0, 5], dtype=np.float32)
            landmarks = np.empty(shape=[0, 10], dtype=np.float32)

        return boxes, landmarks
