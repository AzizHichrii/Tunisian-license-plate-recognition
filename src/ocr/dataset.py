"""PyTorch dataset for CRNN: plate crops -> tensors + encoded labels."""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ALPHABET = "0123456789NT"
BLANK = 0
CHAR2IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}   # 0 = blank
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}

IMG_DIR = Path("data/raw/recognition/images")
SPLITS = Path("data/processed/recognition/splits.csv")
IMG_H, IMG_W = 32, 128


def encode(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text]


def greedy_decode(indices: list[int]) -> str:
    """CTC decoding: merge repeated indices, then drop blanks.
    e.g. [3,3,0,3,12,12,0] -> '2' '2' 'T' -> '22T'"""
    out, prev = [], None
    for i in indices:
        if i != prev and i != BLANK:
            out.append(IDX2CHAR[i])
        prev = i
    return "".join(out)


def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale, resize keeping ratio, pad right to IMG_W, scale to [-1, 1]."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    new_w = min(IMG_W, max(1, int(w * IMG_H / h)))
    img = cv2.resize(img, (new_w, IMG_H))
    canvas = np.full((IMG_H, IMG_W), int(img.mean()), dtype=np.uint8)
    canvas[:, :new_w] = img
    return canvas.astype(np.float32) / 127.5 - 1.0


def augment(img: np.ndarray) -> np.ndarray:
    """Light, realistic augmentation (train only). Never flip: text would mirror."""
    alpha, beta = random.uniform(0.7, 1.3), random.uniform(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)          # contrast/brightness
    if random.random() < 0.5:
        h, w = img.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)  # small tilt
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)                       # blur
    return img


class PlateDataset(Dataset):
    def __init__(self, split: str, train: bool = False):
        df = pd.read_csv(SPLITS, dtype={"text": str})
        self.df = df[df.split == split].reset_index(drop=True)
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = cv2.imread(str(IMG_DIR / row.img_id))
        if self.train:
            img = augment(img)
        x = torch.from_numpy(preprocess(img)).unsqueeze(0)    # (1, H, W)
        y = torch.tensor(encode(row.text), dtype=torch.long)
        return x, y, row.text


def collate(batch):
    """CTC needs all labels concatenated + the length of each one."""
    xs, ys, texts = zip(*batch)
    return (torch.stack(xs), torch.cat(ys),
            torch.tensor([len(y) for y in ys]), list(texts))


if __name__ == "__main__":
    ds = PlateDataset("train", train=True)
    x, y, text = ds[0]
    print(f"train size={len(ds)} | x={tuple(x.shape)} | text={text} -> {y.tolist()}")
    print("decode check:", greedy_decode([2, 2, 0, 2, 12, 12, 0, 5]), "(expect 11T4)")

    tiles = [((ds[i][0][0].numpy() + 1) * 127.5).astype(np.uint8) for i in range(8)]
    out = Path("data/interim/ocr_samples.jpg")
    cv2.imwrite(str(out), np.vstack(tiles))
    print(f"8 preprocessed samples saved to {out}")