"""Remove duplicate images: keep one per duplicate group."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
INT = Path("data/interim")

CONFIG = {
    "detection": RAW / "detection/license_plates_detection_train.csv",
    "recognition": RAW / "recognition/license_plates_recognition_train.csv",
}


def img_num(img_id: str) -> int:
    """'123.jpg' -> 123 (to sort images in numeric order)."""
    return int(Path(img_id).stem)


def clean(name: str, labels_path: Path) -> None:
    labels = pd.read_csv(labels_path, dtype={"text": str})
    dups = pd.read_csv(INT / f"duplicates_{name}.csv")

    # In each group: keep the image with the smallest number, drop the others
    dups["num"] = dups.img_id.map(img_num)
    keep = dups.sort_values("num").groupby("group").img_id.first()
    drop = set(dups.img_id) - set(keep)

    clean_df = labels[~labels.img_id.isin(drop)].reset_index(drop=True)
    removed = dups[dups.img_id.isin(drop)][["group", "img_id"]]

    clean_df.to_csv(INT / f"{name}_clean.csv", index=False)
    removed.to_csv(INT / f"{name}_removed.csv", index=False)
    print(f"{name}: {len(labels)} -> {len(clean_df)} (removed {len(drop)})")


if __name__ == "__main__":
    for name, path in CONFIG.items():
        clean(name, path)