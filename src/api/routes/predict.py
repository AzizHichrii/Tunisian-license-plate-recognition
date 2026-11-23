"""POST /predict: upload an image, get the plates back."""
import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.api.config import settings
from src.api.schemas import PredictResponse

router = APIRouter(tags=["inference"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, file: UploadFile = File(...)):
    if file.content_type not in settings.allowed_types:
        raise HTTPException(415, "Only JPEG or PNG images are accepted.")

    data = file.file.read(int(settings.max_upload_mb * 1024 * 1024) + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image too large (max {settings.max_upload_mb} MB).")

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:  # real content check: don't trust the file name/extension
        raise HTTPException(400, "File is not a valid image.")

    result = request.app.state.pipeline.predict(img)
    for d in result["detections"]:
        d["uncertain"] = d["ocr_confidence"] < settings.uncertain_below
    return PredictResponse(num_plates=len(result["detections"]), **result)