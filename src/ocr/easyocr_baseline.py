"""EasyOCR baseline on the OCR validation set, tracked in MLflow."""
import time
from pathlib import Path

import easyocr
import mlflow
import pandas as pd

from src.detection.train import MLFLOW_URI, data_version, git_commit
from src.ocr.metrics import cer, exact_match

IMG_DIR = Path("data/raw/recognition/images")
SPLITS = Path("data/processed/recognition/splits.csv")
SPLIT = "val"  # never test until the end


def to_plate_text(results) -> str:
    """EasyOCR boxes -> 'leftTright' (2 groups) or 'digitsN' (1 group)."""
    groups = sorted(results, key=lambda r: r[0][0][0])  # sort by left x
    digits = [text for _, text, _ in groups if text]
    if len(digits) >= 2:
        return f"{digits[0]}T{digits[-1]}"
    if len(digits) == 1:
        return f"{digits[0]}N"
    return ""


def main() -> None:
    df = pd.read_csv(SPLITS, dtype={"text": str})
    df = df[df.split == SPLIT].reset_index(drop=True)
    reader = easyocr.Reader(["en"], gpu=True)

    preds, raws, times = [], [], []
    for img_id in df.img_id:
        t0 = time.perf_counter()
        res = reader.readtext(str(IMG_DIR / img_id), allowlist="0123456789")
        times.append((time.perf_counter() - t0) * 1000)
        raws.append(" | ".join(text for _, text, _ in res))
        preds.append(to_plate_text(res))

    df["raw"], df["pred"] = raws, preds
    df["correct"] = df.pred == df.text
    metrics = {
        "val_CER": cer(preds, df.text.tolist()),
        "val_exact_match": exact_match(preds, df.text.tolist()),
        "inference_ms": sum(times) / len(times),
    }

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("TLPR_OCR")
    with mlflow.start_run(run_name="easyocr_baseline"):
        mlflow.log_params({"engine": "easyocr", "lang": "en", "allowlist": "digits",
                           "split": SPLIT, "git_commit": git_commit(),
                           "data_md5": data_version()})
        mlflow.log_metrics(metrics)
        out = Path("runs/ocr/easyocr_val_predictions.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        mlflow.log_artifact(str(out), "predictions")

    print({k: round(v, 4) for k, v in metrics.items()})
    print("\nErrors (first 10):")
    print(df[~df.correct][["img_id", "text", "pred", "raw"]].head(10).to_string())


if __name__ == "__main__":
    main()