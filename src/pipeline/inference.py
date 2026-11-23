"""Full pipeline: image -> detect plates -> crop -> read -> validate."""
import time
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.detector import PlateDetector
from src.pipeline.recognizer import PlateRecognizer
from src.pipeline.validator import validate

DET_WEIGHTS = "models/detection/yolo11n_best.pt"
OCR_WEIGHTS = "models/ocr/crnn_v3_seq64.pt"


def crop_plate(img: np.ndarray, bbox: list[int], margin: float = 0.05) -> np.ndarray:
    """Cut the plate with a small margin, clipped to the image borders."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return img[max(0, y1 - my):min(h, y2 + my), max(0, x1 - mx):min(w, x2 + mx)]


class LPRPipeline:
    def __init__(self, det_weights: str = DET_WEIGHTS, ocr_weights: str = OCR_WEIGHTS):
        self.detector = PlateDetector(det_weights)
        self.recognizer = PlateRecognizer(ocr_weights)

    def read_crop(self, crop: np.ndarray) -> tuple[str, float]:
        """Vertical plate (rotated photo)? Try both 90° rotations, keep the most confident."""
        if crop.shape[0] > crop.shape[1] * 1.2:
            candidates = [self.recognizer.read(cv2.rotate(crop, r))
                          for r in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)]
            return max(candidates, key=lambda c: c[1])
        return self.recognizer.read(crop)

    def predict(self, img: np.ndarray) -> dict:
        t0 = time.perf_counter()
        boxes = self.detector.detect(img)
        t1 = time.perf_counter()

        detections = []
        for b in boxes:
            crop = crop_plate(img, b["bbox"])
            if crop.size == 0:
                continue
            raw, ocr_conf = self.read_crop(crop)
            v = validate(raw)
            detections.append({
                "bbox": b["bbox"],
                "detection_confidence": round(b["confidence"], 4),
                "plate_text_raw": raw,
                "plate_text": v["text"],
                "display": v["display"],
                "format": v["format"],
                "status": v["status"],
                "ocr_confidence": round(ocr_conf, 4),
            })
        t2 = time.perf_counter()
        return {
            "success": len(detections) > 0,
            "detections": detections,
            "timings_ms": {"detection": round((t1 - t0) * 1000, 1),
                           "ocr": round((t2 - t1) * 1000, 1),
                           "total": round((t2 - t0) * 1000, 1)},
        }


def draw(img: np.ndarray, result: dict) -> np.ndarray:
    """Draw boxes + text (ASCII only: OpenCV can't draw Arabic)."""
    out = img.copy()
    for d in result["detections"]:
        x1, y1, x2, y2 = d["bbox"]
        color = (0, 200, 0) if d["status"] == "valid" else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d['plate_text_raw']} ({d['ocr_confidence']:.2f})", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


if __name__ == "__main__":
    pipe = LPRPipeline()
    out_dir = Path("runs/pipeline/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(Path("data/raw/test/images").glob("*.jpg"))[:20]
    for p in images:
        img = cv2.imread(str(p))
        r = pipe.predict(img)
        texts = [f"{d['plate_text_raw']}({d['status']})" for d in r["detections"]] or ["NO PLATE"]
        print(f"{p.name:>8} | {r['timings_ms']['total']:6.1f} ms | {', '.join(texts)}")
        cv2.imwrite(str(out_dir / p.name), draw(img, r))
    print(f"\nAnnotated images saved in {out_dir}")