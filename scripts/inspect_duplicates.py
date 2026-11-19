"""Compare labels inside each duplicate group + save side-by-side views."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")
BOX = ["xmin", "ymin", "xmax", "ymax"]
TOL = 5  # pixels: boxes this close are considered the same


def side_by_side(paths, captions, out_path, h=200):
    """Put images next to each other (same height) with a caption."""
    tiles = []
    for p, cap in zip(paths, captions):
        img = cv2.imread(str(p))
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        cv2.putText(img, cap, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(img)
    cv2.imwrite(str(out_path), np.hstack(tiles))


def check_detection():
    dups = pd.read_csv(INT / "duplicates_detection.csv")
    labels = pd.read_csv(RAW / "detection/license_plates_detection_train.csv")
    df = dups.merge(labels, on="img_id")
    out = INT / "dup_view_detection"
    out.mkdir(exist_ok=True)
    conflicts = 0
    for g, grp in df.groupby("group"):
        spread = (grp[BOX].max() - grp[BOX].min()).max()
        if spread > TOL:
            conflicts += 1
            paths, caps = [], []
            for r in grp.itertuples():
                img = cv2.imread(str(RAW / "detection/images" / r.img_id))
                cv2.rectangle(img, (r.xmin, r.ymin), (r.xmax, r.ymax), (0, 0, 255), 3)
                tmp = out / f"_tmp_{r.img_id}"
                cv2.imwrite(str(tmp), img)
                paths.append(tmp)
                caps.append(r.img_id)
            side_by_side(paths, caps, out / f"group_{g}.jpg", h=300)
            for t in paths:
                t.unlink()
    print(f"detection: {df.group.nunique()} groups -> "
          f"{df.group.nunique() - conflicts} same box, {conflicts} different box")


def check_recognition():
    dups = pd.read_csv(INT / "duplicates_recognition.csv")
    labels = pd.read_csv(RAW / "recognition/license_plates_recognition_train.csv", dtype=str)
    df = dups.merge(labels, on="img_id")
    out = INT / "dup_view_recognition"
    out.mkdir(exist_ok=True)
    same = 0
    for g, grp in df.groupby("group"):
        same += grp.text.nunique() == 1
        side_by_side([RAW / "recognition/images" / i for i in grp.img_id],
                     [f"{i} {t}" for i, t in zip(grp.img_id, grp.text)],
                     out / f"group_{g}.jpg")
    print(f"recognition: {df.group.nunique()} groups -> "
          f"{same} same text, {df.group.nunique() - same} different text")


if __name__ == "__main__":
    check_detection()
    check_recognition()