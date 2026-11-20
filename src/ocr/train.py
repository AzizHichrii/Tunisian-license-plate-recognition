"""Train the CRNN with CTC loss, track everything in MLflow."""
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.detection.train import MLFLOW_URI, data_version, git_commit
from src.ocr.dataset import PlateDataset, collate, greedy_decode
from src.ocr.metrics import cer, exact_match
from src.ocr.model import CRNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, ctc):
    """Return val loss, predictions and targets."""
    model.eval()
    losses, preds, targets = [], [], []
    for x, y, y_len, texts in loader:
        logp = model(x.to(DEVICE))
        in_len = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
        losses.append(ctc(logp, y, in_len, y_len).item())
        best = logp.argmax(2).T.cpu().tolist()          # (B, T)
        preds += [greedy_decode(seq) for seq in best]
        targets += texts
    return float(np.mean(losses)), preds, targets


def main(cfg_path: str = "configs/ocr.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    set_seed(cfg["seed"])
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)

    train_dl = DataLoader(PlateDataset("train", train=True), batch_size=cfg["batch"],
                          shuffle=True, collate_fn=collate)
    val_dl = DataLoader(PlateDataset("val"), batch_size=cfg["batch"], collate_fn=collate)

    model = CRNN(hidden=cfg["hidden"]).to(DEVICE)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=f"crnn_h{cfg['hidden']}_bs{cfg['batch']}"):
        mlflow.log_params({**cfg, "git_commit": git_commit(), "data_md5": data_version(),
                           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)})

        best_em, best_cer, wait = -1.0, 1.0, 0
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            train_losses = []
            for x, y, y_len, _ in train_dl:
                logp = model(x.to(DEVICE))
                in_len = torch.full((x.size(0),), logp.size(0), dtype=torch.long)
                loss = ctc(logp, y, in_len, y_len)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # avoid exploding gradients
                opt.step()
                train_losses.append(loss.item())

            val_loss, preds, targets = evaluate(model, val_dl, ctc)
            em, c = exact_match(preds, targets), cer(preds, targets)
            sched.step(em)
            mlflow.log_metrics({"train_loss": float(np.mean(train_losses)), "val_loss": val_loss,
                                "val_exact_match": em, "val_CER": c,
                                "lr": opt.param_groups[0]["lr"]}, step=epoch)
            print(f"epoch {epoch:3d} | train {np.mean(train_losses):.3f} | val {val_loss:.3f} "
                  f"| EM {em:.3f} | CER {c:.3f}")

            if em > best_em or (em == best_em and c < best_cer):
                best_em, best_cer, wait = em, c, 0
                torch.save(model.state_dict(), out / "best.pt")
                mlflow.log_metric("best_epoch", epoch)
            else:
                wait += 1
                if wait >= cfg["patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation with the best weights
        model.load_state_dict(torch.load(out / "best.pt", map_location=DEVICE))
        t0 = time.perf_counter()
        _, preds, targets = evaluate(model, val_dl, ctc)
        ms = (time.perf_counter() - t0) * 1000 / len(targets)
        df = pd.DataFrame({"img_id": val_dl.dataset.df.img_id, "text": targets, "pred": preds})
        df.to_csv(out / "val_predictions.csv", index=False)

        mlflow.log_metrics({"best_val_exact_match": exact_match(preds, targets),
                            "best_val_CER": cer(preds, targets), "inference_ms": ms})
        mlflow.log_artifact(str(out / "best.pt"), "weights")
        mlflow.log_artifact(str(out / "val_predictions.csv"), "predictions")
        print(f"\nBEST: exact_match={best_em:.3f} CER={best_cer:.3f} | {ms:.2f} ms/plate")


if __name__ == "__main__":
    main()