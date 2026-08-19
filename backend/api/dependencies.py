"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, Request

from backend.cv.detector import CapsuleDetector


def get_detector(request: Request) -> Optional[CapsuleDetector]:
    """Return the app-wide CapsuleDetector from ``app.state``.

    If the model could not be loaded at startup this returns None and the
    caller must respond with a clear 503 error.
    """
    return getattr(request.app.state, "detector", None)


def require_detector(request: Request) -> CapsuleDetector:
    detector = get_detector(request)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The YOLO model is not available. Check that the model file exists "
                "at the configured YOLO_MODEL_PATH and restart the server."
            ),
        )
    return detector
