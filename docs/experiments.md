# Experiments log

## Data (version `data-v1.0`, DVC)
| | Raw | After dedup | Train | Val | Test |
|---|---|---|---|---|---|
| Detection | 900 | 832 (68 duplicates removed) | 583 | 125 | 124 |
| OCR | 900 | 888 (12 duplicates removed) | 625 | 130 | 133 |

- Split 70/15/15, seed 42. OCR split grouped by plate text (no plate in two splits).
- EDA: images ≤ 850×547 px, median plate = 1.7% of image → `imgsz=640`.
- OCR alphabet: `0123456789NT`, length 5–8.

### Known limitations
- Only 5 plates in `N` format (vs 883 `T`).
- Same car can appear in several detection photos (not groupable, no text in detection labels).
- 1 box per image: background plates are not labeled.

## Detection — YOLO (val set)
Config: imgsz 640, 50 epochs, batch 8, patience 15, seed 42, RTX 3050 4 GB.

| Model | mAP50-95 | mAP50 | Precision | Recall | Inference (ms) | Train time | Best epoch |
|---|---|---|---|---|---|---|---|
| yolov8n | 0.787 | 0.993 | 1.000 | 0.975 | 10.8 | 11 min | 37 |
| **yolo11n** | **0.801** | 0.994 | 0.998 | **0.984** | 13.1 | 12 min | 46 |
| yolo26n | 0.795 | 0.995 | 0.990 | 0.984 | 12.6 | 17 min | 40 |

**Selected: yolo11n** — best recall and box precision (mAP50-95). Speed difference negligible.
Note: differences ~0.01 on 125 val images may be partly noise.