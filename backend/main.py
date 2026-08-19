"""CapsuleAI backend entry point.

Run with::

    uvicorn backend.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analysis import router as analysis_router
from backend.api.upload import router as upload_router
from backend.config import settings, setup_logging
from backend.cv.detector import CapsuleDetector, model_exists

logger = logging.getLogger("application")

MODEL_LOAD_FAILURE_MSG = (
    "The YOLO model is NOT loaded. Set YOLO_MODEL_PATH in .env and ensure the "
    "model file exists, or run `python scripts/download_demo_model.py` to fetch "
    "a demo model."
)


def _load_detector() -> CapsuleDetector | None:
    """Load the YOLO detector once. Returns None (non-fatal) if unavailable."""
    if not model_exists(settings.yolo_model_path):
        logger.error("model missing: %s | %s", settings.yolo_model_path, MODEL_LOAD_FAILURE_MSG)
        return None
    try:
        return CapsuleDetector(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.confidence_threshold,
            device=settings.device,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("failed to load YOLO model: %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.ensure_directories()
    app.state.detector = _load_detector()
    logger.info(
        "CapsuleAI starting | env=%s | device=%s | cuda_available=%s | model=%s",
        settings.app_env,
        (app.state.detector.device if app.state.detector else "n/a"),
        torch.cuda.is_available(),
        settings.yolo_model_path.name,
    )
    yield


app = FastAPI(
    title="CapsuleAI",
    description=(
        "AI-Assisted Capsule Endoscopy Analysis and Multi-Agent Report "
        "Generation System (Milestone 1: image detection)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)
app.include_router(analysis_router)


@app.get("/api/health", tags=["system"])
async def health():
    """Liveness/readiness probe with model and inference device info."""
    detector = getattr(app.state, "detector", None)
    return {
        "status": "ok",
        "app_env": settings.app_env,
        "device": detector.device if detector else ("cpu" if not torch.cuda.is_available() else "cuda"),
        "cuda_available": torch.cuda.is_available(),
        "inference_device": "NVIDIA GPU" if (detector and detector.device == "cuda") else "CPU",
        "model_loaded": detector is not None,
        "model_path": settings.yolo_model_path.as_posix(),
        "model_name": detector.model_path.name if detector else None,
        "num_classes": len(detector.class_names) if detector else None,
        "torch_version": torch.__version__,
        "message": None if detector else MODEL_LOAD_FAILURE_MSG,
    }
