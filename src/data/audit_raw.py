"""Audit raw data: find problems without modifying anything."""
import re
from pathlib import Path

import pandas as pd
from PIL import Image

RAW = Path("data/raw")
OUT = Path("data/interim")
PLATE_RE = re.compile(r"^\d{1,3}T\d{1,4}$|^\d{1,6}N$")


def open_image(path: Path):
    """Return (width, height) or None if missing/corrupted."""
    if not path.exists():
        return None
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return im.size
    except Exception:
        return None


def audit_detection() -> pd.DataFrame:
    df = pd.read_csv(RAW / "detection/license_plates_detection_train.csv")
    issues = []
    for r in df.itertuples():
        size = open_image(RAW / "detection/images" / r.img_id)
        if size is None:
            issues.append((r.img_id, "missing_or_corrupted"))
            continue
        w, h = size
        if not (0 <= r.xmin < r.xmax <= w and 0 <= r.ymin < r.ymax <= h):
            issues.append((r.img_id, f"bbox_out_of_bounds ({w}x{h})"))
        if (r.xmax - r.xmin) < (r.ymax - r.ymin):
            issues.append((r.img_id, "bbox_taller_than_wide"))
    labeled = set(df.img_id)
    for p in (RAW / "detection/images").glob("*.jpg"):
        if p.name not in labeled:
            issues.append((p.name, "image_without_label"))
    for img_id in df[df.img_id.duplicated()].img_id:
        issues.append((img_id, "duplicate_label_row"))
    return pd.DataFrame(issues, columns=["img_id", "issue"])


def audit_recognition() -> pd.DataFrame:
    df = pd.read_csv(RAW / "recognition/license_plates_recognition_train.csv", dtype=str)
    issues = []
    for r in df.itertuples():
        if open_image(RAW / "recognition/images" / r.img_id) is None:
            issues.append((r.img_id, "missing_or_corrupted"))
        if not PLATE_RE.match(str(r.text)):
            issues.append((r.img_id, f"bad_text_format ({r.text})"))
    labeled = set(df.img_id)
    for p in (RAW / "recognition/images").glob("*.jpg"):
        if p.name not in labeled:
            issues.append((p.name, "image_without_label"))
    return pd.DataFrame(issues, columns=["img_id", "issue"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("detection", audit_detection), ("recognition", audit_recognition)]:
        rep = fn()
        rep.to_csv(OUT / f"audit_{name}.csv", index=False)
        print(f"\n=== {name.upper()} : {len(rep)} issues ===")
        if len(rep):
            print(rep.issue.str.split(" ").str[0].value_counts().to_string())