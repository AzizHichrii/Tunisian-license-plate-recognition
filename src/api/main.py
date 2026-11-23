"""FastAPI app: loads the models once at startup."""
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.routes import health, predict
from src.pipeline.inference import LPRPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipe = LPRPipeline(settings.det_weights, settings.ocr_weights)
    pipe.predict(np.zeros((640, 640, 3), np.uint8))           # warm-up YOLO (GPU init)
    pipe.recognizer.read(np.zeros((40, 160, 3), np.uint8))    # warm-up CRNN
    app.state.pipeline = pipe
    yield


app = FastAPI(title="Tunisian License Plate Recognition API", version="1.0.0",
              description="Detect Tunisian plates (YOLO11n) and read them (CRNN).",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(predict.router)