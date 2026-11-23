"""API settings. Override with environment variables (useful for Docker)."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    det_weights: str = os.getenv("DET_WEIGHTS", "models/detection/yolo11n_best.pt")
    ocr_weights: str = os.getenv("OCR_WEIGHTS", "models/ocr/crnn_v3_seq64.pt")
    max_upload_mb: float = float(os.getenv("MAX_UPLOAD_MB", "5"))
    uncertain_below: float = float(os.getenv("UNCERTAIN_BELOW", "0.9"))  # from test eval
    allowed_types: tuple = ("image/jpeg", "image/png")
    cors_origins: list = field(default_factory=lambda: os.getenv(
        "CORS_ORIGINS", "http://localhost:5173").split(","))


settings = Settings()