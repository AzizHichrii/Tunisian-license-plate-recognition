"""Validate YOLO labels and draw a few samples to check by eye."""
import random
from pathlib import Path

import cv2

ROOT = Path("data/processed/detection")
OUT = Path("data/interim/yolo_check")
OUT.mkdir(parents=True, exist_ok=True)

errors, all_imgs = [], []
for split in ["train", "val", "test"]:
    for img in (ROOT / "images" / split).glob("*.jpg"):
        lbl = ROOT / "labels" / split / f"{img.stem}.txt"
        if not lbl.exists():
            errors.append(f"{img.name}: missing label")
            continue
        for line in lbl.read_text().splitlines():
            cls, *vals = line.split()
            vals = list(map(float, vals))
            if cls != "0" or len(vals) != 4 or not all(0 <= v <= 1 for v in vals):
                errors.append(f"{img.name}: bad line '{line}'")
        all_imgs.append((img, lbl))

print(f"{len(all_imgs)} images checked, {len(errors)} errors")
for e in errors[:10]:
    print("  ", e)

random.seed(0)
for img_path, lbl in random.sample(all_imgs, 12):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    _, xc, yc, bw, bh = map(float, lbl.read_text().split())
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.imwrite(str(OUT / img_path.name), img)
print(f"12 samples drawn in {OUT}")