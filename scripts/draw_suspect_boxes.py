"""Draw the suspicious bboxes so we can inspect them by eye."""
from pathlib import Path

import cv2
import pandas as pd

RAW = Path("data/raw/detection")
OUT = Path("data/interim/suspect_boxes")
OUT.mkdir(parents=True, exist_ok=True)

audit = pd.read_csv("data/interim/audit_detection.csv")
labels = pd.read_csv(RAW / "license_plates_detection_train.csv")
suspects = labels[labels.img_id.isin(audit.img_id)]

for r in suspects.itertuples():
    img = cv2.imread(str(RAW / "images" / r.img_id))
    cv2.rectangle(img, (r.xmin, r.ymin), (r.xmax, r.ymax), (0, 0, 255), 3)
    ratio = (r.xmax - r.xmin) / (r.ymax - r.ymin)
    cv2.putText(img, f"ratio={ratio:.2f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imwrite(str(OUT / r.img_id), img)

print(f"{len(suspects)} images saved in {OUT}")