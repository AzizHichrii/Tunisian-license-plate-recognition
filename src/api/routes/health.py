"""Service status and active model versions."""
from fastapi import APIRouter, Request

from src.api.config import settings
from src.api.schemas import HealthResponse

router = APIRouter(tags=["status"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    pipe = getattr(request.app.state, "pipeline", None)
    return HealthResponse(status="ok" if pipe else "loading", models_loaded=pipe is not None,
                          device=pipe.recognizer.device if pipe else "unknown")


@router.get("/models")
def models():
    return {"detector": settings.det_weights, "ocr": settings.ocr_weights,
            "uncertain_below": settings.uncertain_below}