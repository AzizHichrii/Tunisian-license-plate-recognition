"""Train a YOLO plate detector and track everything in MLflow."""
import argparse
import re
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from ultralytics import YOLO

MLFLOW_URI = "http://127.0.0.1:5000"


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()


def data_version() -> str:
    """md5 of data/processed from the DVC 'ticket' file."""
    dvc = yaml.safe_load(Path("data/processed.dvc").read_text())
    return dvc["outs"][0]["md5"]


def clean_name(name: str) -> str:
    """MLflow refuses some characters: 'metrics/mAP50(B)' -> 'metrics/mAP50'."""
    return re.sub(r"[^\w\-./ ]", "", name.strip().replace("(B)", ""))

def main(cfg_path: str, epochs: int | None, model_name: str | None = None) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if epochs:
        cfg["epochs"] = epochs
    if model_name:
        cfg["model"] = model_name
    run_name = f"{Path(cfg['model']).stem}_img{cfg['imgsz']}_e{cfg['epochs']}"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_params({"git_commit": git_commit(), "data_md5": data_version()})

        # 1) Train
        model = YOLO(cfg["model"])
        model.train(
            data=cfg["data"], imgsz=cfg["imgsz"], epochs=cfg["epochs"],
            batch=cfg["batch"], patience=cfg["patience"], seed=cfg["seed"],
            device=cfg["device"], workers=cfg["workers"],
            project=str(Path(cfg["project"]).resolve()), name=run_name, exist_ok=True,
        )
        save_dir = Path(model.trainer.save_dir)

        # 2) Log the curves epoch by epoch (from results.csv)
        hist = pd.read_csv(save_dir / "results.csv")
        hist.columns = [c.strip() for c in hist.columns]
        for _, row in hist.iterrows():
            step = int(row["epoch"])
            metrics = {clean_name(k): float(v) for k, v in row.items() if k != "epoch"}
            mlflow.log_metrics(metrics, step=step)

        # 3) Evaluate the best weights on the validation set
        best = YOLO(save_dir / "weights" / "best.pt")
        m = best.val(data=cfg["data"], split="val", imgsz=cfg["imgsz"],
                     batch=cfg["batch"], device=cfg["device"],
                     project=str(Path(cfg["project"]).resolve()), name=f"{run_name}_val", exist_ok=True)
        mlflow.log_metrics({
            "val_precision": m.box.mp,
            "val_recall": m.box.mr,
            "val_mAP50": m.box.map50,
            "val_mAP50-95": m.box.map,
            "inference_ms": m.speed["inference"],
            "best_epoch": int(hist.loc[hist["metrics/mAP50-95(B)"].idxmax(), "epoch"]),
        })

        # 4) Save the model + charts as artifacts
        mlflow.log_artifact(str(save_dir / "weights" / "best.pt"), "weights")
        for f in ["results.png", "confusion_matrix.png", "BoxPR_curve.png", "args.yaml"]:
            if (save_dir / f).exists():
                mlflow.log_artifact(str(save_dir / f), "plots")

        print(f"\nDone: mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="override model, e.g. yolo11n.pt")
    p.add_argument("--config", default="configs/detection.yaml")
    p.add_argument("--epochs", type=int, help="override epochs (quick test)")
    a = p.parse_args()
    main(a.config, a.epochs, a.model)