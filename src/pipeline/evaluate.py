"""Final, one-time evaluation of the selected models on the TEST split."""
from pathlib import Path

import cv2
import mlflow
import pandas as pd
from ultralytics import YOLO

from src.detection.train import MLFLOW_URI, data_version, git_commit
from src.ocr.metrics import cer, exact_match
from src.pipeline.inference import DET_WEIGHTS, OCR_WEIGHTS
from src.pipeline.recognizer import PlateRecognizer
from src.pipeline.validator import validate

OUT = Path("runs/final_eval")


def eval_detection() -> dict:
    m = YOLO(DET_WEIGHTS).val(data="data/processed/detection/data.yaml", split="test",
                              imgsz=640, batch=8, device=0,
                              project=str(OUT.resolve()), name="detection", exist_ok=True)
    return {"det_test_precision": m.box.mp, "det_test_recall": m.box.mr,
            "det_test_mAP50": m.box.map50, "det_test_mAP50-95": m.box.map,
            "det_inference_ms": m.speed["inference"]}


def eval_ocr() -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv("data/processed/recognition/splits.csv", dtype={"text": str})
    df = df[df.split == "test"].reset_index(drop=True)
    rec = PlateRecognizer(OCR_WEIGHTS)

    reads = [rec.read(cv2.imread(f"data/raw/recognition/images/{i}")) for i in df.img_id]
    df["pred"] = [t for t, _ in reads]
    df["confidence"] = [c for _, c in reads]
    df["status"] = [validate(t)["status"] for t in df.pred]
    df["correct"] = df.pred == df.text

    high = df[df.confidence >= 0.9]
    metrics = {
        "ocr_test_exact_match": exact_match(df.pred.tolist(), df.text.tolist()),
        "ocr_test_CER": cer(df.pred.tolist(), df.text.tolist()),
        "ocr_test_valid_format_rate": (df.status == "valid").mean(),
        "ocr_test_share_conf_ge_0.9": len(high) / len(df),
        "ocr_test_exact_match_conf_ge_0.9": high.correct.mean() if len(high) else 0.0,
    }
    return metrics, df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("TLPR_End_To_End")
    with mlflow.start_run(run_name="final_test_eval"):
        mlflow.log_params({"detector": DET_WEIGHTS, "ocr": OCR_WEIGHTS, "split": "test",
                           "git_commit": git_commit(), "data_md5": data_version()})
        det = eval_detection()
        ocr, preds = eval_ocr()
        mlflow.log_metrics({**det, **ocr})

        preds.to_csv(OUT / "ocr_test_predictions.csv", index=False)
        mlflow.log_artifact(str(OUT / "ocr_test_predictions.csv"), "predictions")

    print("\n=== DETECTION (test, 124 images) ===")
    for k, v in det.items():
        print(f"  {k:<32} {v:.4f}")
    print("=== OCR (test, 133 crops) ===")
    for k, v in ocr.items():
        print(f"  {k:<32} {v:.4f}")


if __name__ == "__main__":
    main()