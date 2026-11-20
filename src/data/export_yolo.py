"""Export detection data to YOLO format (images/ + labels/ + data.yaml)."""
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW_IMG = Path("data/raw/detection/images")
OUT = Path("data/processed/detection")
CLASS_ID = 0  # only one class: license_plate


def to_yolo(r, w: int, h: int) -> str:
    """Pixel box (xmin,ymin,xmax,ymax) -> YOLO line (normalized 0..1)."""
    xc = (r.xmin + r.xmax) / 2 / w
    yc = (r.ymin + r.ymax) / 2 / h
    bw = (r.xmax - r.xmin) / w
    bh = (r.ymax - r.ymin) / h
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"


def main() -> None:
    df = pd.read_csv(OUT / "splits.csv")

    # Start from empty folders so re-running gives the same result
    for sub in ["images", "labels"]:
        shutil.rmtree(OUT / sub, ignore_errors=True)
        for split in ["train", "val", "test"]:
            (OUT / sub / split).mkdir(parents=True)

    for r in tqdm(df.itertuples(), total=len(df)):
        src = RAW_IMG / r.img_id
        with Image.open(src) as im:
            w, h = im.size
        shutil.copy2(src, OUT / "images" / r.split / r.img_id)
        label_path = OUT / "labels" / r.split / f"{Path(r.img_id).stem}.txt"
        label_path.write_text(to_yolo(r, w, h))

    yaml = (
        f"path: {OUT.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: license_plate\n"
    )
    (OUT / "data.yaml").write_text(yaml)
    print(df.split.value_counts().to_string())
    print(f"data.yaml written in {OUT}")


if __name__ == "__main__":
    main()