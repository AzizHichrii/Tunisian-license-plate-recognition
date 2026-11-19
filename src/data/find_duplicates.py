"""Find exact and near-duplicate images using perceptual hashing."""
import hashlib
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/interim")
MAX_DIST = 5  # hamming distance: 0 = identical, <=5 = almost identical


def hash_images(folder: Path) -> pd.DataFrame:
    rows = []
    for p in tqdm(sorted(folder.glob("*.jpg")), desc=folder.parent.name):
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        with Image.open(p) as im:
            ph = imagehash.phash(im)
        rows.append((p.name, md5, ph))
    return pd.DataFrame(rows, columns=["img_id", "md5", "phash"])


def group_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Union-find: images linked by similarity end up in the same group."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = df.phash.tolist()
    md5s = df.md5.tolist()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if md5s[i] == md5s[j] or hashes[i] - hashes[j] <= MAX_DIST:
                parent[find(j)] = find(i)

    df = df.assign(group=[find(i) for i in range(len(df))])
    sizes = df.group.map(df.group.value_counts())
    return df[sizes > 1].sort_values("group")[["group", "img_id", "md5"]]


if __name__ == "__main__":
    for name in ["detection", "recognition"]:
        dups = group_duplicates(hash_images(RAW / name / "images"))
        dups.to_csv(OUT / f"duplicates_{name}.csv", index=False)
        n_groups = dups.group.nunique()
        exact = dups.duplicated("md5", keep=False).sum()
        print(f"{name}: {n_groups} groups, {len(dups)} images involved, {exact} exact copies")