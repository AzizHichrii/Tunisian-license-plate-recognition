"""Read a plate crop with the trained CRNN."""
import numpy as np
import torch

from src.ocr.dataset import BLANK, IDX2CHAR, preprocess
from src.ocr.model import CRNN


class PlateRecognizer:
    def __init__(self, weights: str, seq_len: int = 64, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CRNN(seq_len=seq_len).to(self.device)
        self.model.load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """BGR crop -> (text, confidence 0..1)."""
        x = torch.from_numpy(preprocess(crop)).unsqueeze(0).unsqueeze(0).to(self.device)
        probs = self.model(x)[:, 0, :].exp()          # (T, classes)
        best_p, best_i = probs.max(1)

        chars, confs, prev = [], [], None
        for p, i in zip(best_p.tolist(), best_i.tolist()):
            if i != prev and i != BLANK:
                chars.append(IDX2CHAR[i])
                confs.append(p)
            prev = i
        return "".join(chars), (min(confs) if confs else 0.0)


if __name__ == "__main__":
    import cv2
    import pandas as pd

    from src.pipeline.validator import validate

    rec = PlateRecognizer("models/ocr/crnn_v3_seq64.pt")
    df = pd.read_csv("data/processed/recognition/splits.csv", dtype={"text": str})
    for row in df[df.split == "val"].head(5).itertuples():
        text, conf = rec.read(cv2.imread(f"data/raw/recognition/images/{row.img_id}"))
        v = validate(text)
        print(f"{row.img_id:>8} | true={row.text:<9} pred={text:<9} conf={conf:.2f} | {v['status']} | {v['display']}")