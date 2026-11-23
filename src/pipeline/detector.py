"""Find license plates in a full image with YOLO."""
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(weights)
        self.conf, self.imgsz = conf, imgsz

    def detect(self, img: np.ndarray) -> list[dict]:
        """BGR image -> list of {bbox: [x1, y1, x2, y2], confidence}."""
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        return [
            {"bbox": [int(v) for v in box.xyxy[0].tolist()], "confidence": float(box.conf[0])}
            for box in res.boxes
        ]