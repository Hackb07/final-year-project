"""Pydantic schemas for Milestone 1 (image detection results).

These schemas will be extended with the structured Finding schema in
Milestone 3. Bounding boxes are always produced by the CV model - never by
an LLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """One object detected by the YOLO model."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: List[float] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2] pixels")

    frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None


class ImageAnalysisResponse(BaseModel):
    """Full response of the ``POST /api/analyze/image`` endpoint."""

    status: str = "completed"
    analysis_id: str
    input_file: str
    input_mime: str
    image_width: int
    image_height: int
    device: str
    model_name: str
    model_classes: Dict[str, str]
    inference_ms: float
    count: int
    detections: List[DetectionResult]
    original_image: str
    annotated_image: str
    message: Optional[str] = None


class UploadResponse(BaseModel):
    """Response of ``POST /api/upload/image``."""

    file_id: str
    filename: str
    stored_path: str
    size_bytes: int
