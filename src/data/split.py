"""Split clean data into train / val / test (70 / 15 / 15)."""
from pathlib import Path

import numpy as np
import pandas as pd

INT = Path("data/interim")
OUT = Path("data/processed")
SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def split_groups(groups: pd.Series) -> pd.Series:
    """Assign each GROUP to a split, so a group never crosses two splits."""
    rng = np.random.default_rng(SEED)
    unique = groups.unique()
    rng.shuffle(unique)
    sizes = groups.value_counts()

    total, done, assign = len(groups), 0, {}
    for g in unique:
        frac = done / total
        if frac < RATIOS["train"]:
            assign[g] = "train"
        elif frac < RATIOS["train"] + RATIOS["val"]:
            assign[g] = "val"
        else:
            assign[g] = "test"
        done += sizes[g]
    return groups.map(assign)


def run(name: str, group_col: str) -> None:
    df = pd.read_csv(INT / f"{name}_clean.csv", dtype={"text": str})
    df["split"] = split_groups(df[group_col])
    (OUT / name).mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name / "splits.csv", index=False)

    counts = df.split.value_counts().reindex(RATIOS)
    pct = (counts / len(df) * 100).round(1)
    print(f"{name}: " + ", ".join(f"{s}={counts[s]} ({pct[s]}%)" for s in RATIOS))


if __name__ == "__main__":
    run("detection", group_col="img_id")   # each image is its own group
    run("recognition", group_col="text")   # same plate text = same group