"""FastAPI app: loads the models once at startup."""
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.db import init_db
from src.api.routes import auth, feedback, health, predict, valley
from src.pipeline.inference import LPRPipeline

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()                                                 # create tables if missing
    pipe = LPRPipeline(settings.det_weights, settings.ocr_weights)
    pipe.predict(np.zeros((640, 640, 3), np.uint8))           # warm-up YOLO (GPU init)
    pipe.recognizer.read(np.zeros((40, 160, 3), np.uint8))    # warm-up CRNN
    app.state.pipeline = pipe
    try:                                                      # warm-up Valley's encoder (bge-m3)
        from src.rag.store import embed
        embed(["warm-up"])
    except Exception as e:                                    # scan must keep working even if RAG fails
        log.warning("Valley embedder warm-up failed: %s", e)
    yield


app = FastAPI(title="PlateReco API", version="1.1.0",
              description="Tunisian plate recognition (YOLO11n + CRNN) and Valley, a RAG assistant "
                          "about cars and Tunisian road law.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(auth.router)
app.include_router(feedback.router)
app.include_router(valley.router)
