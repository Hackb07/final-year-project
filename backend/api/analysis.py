"""Analysis endpoints (Milestone 1: image inference + annotated output)."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.dependencies import require_detector
from backend.api.upload import _sanitize_filename, validate_upload
from backend.config import settings
from backend.cv.detector import CapsuleDetector
from backend.cv.postprocess import decode_image, draw_detections, save_image
from backend.schemas.findings import DetectionResult, ImageAnalysisResponse

logger = logging.getLogger("application")

router = APIRouter(prefix="/api/analyze", tags=["analyze"])


@router.post("/image", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    detector: CapsuleDetector = Depends(require_detector),
):
    """Run the YOLO model on an uploaded image and return detections.

    The original image and an annotated copy (bounding boxes, labels,
    confidence) are saved to the outputs directory.
    """
    validate_upload(file)
    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum allowed size ({settings.max_upload_size_mb} MB).",
        )

    try:
        image_bgr = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    analysis_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
    stem = uuid.uuid4().hex
    ext = ".jpg"

    original_rel = settings.original_dir / f"{stem}{ext}"
    annotated_rel = settings.annotated_dir / f"{stem}_annotated{ext}"

    try:
        save_image(image_bgr, original_rel)

        result = detector.predict(image_bgr)
        detections = result.detections

        annotated = draw_detections(image_bgr, detections)
        save_image(annotated, annotated_rel)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    detections_out = [
        DetectionResult(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox=[round(float(v), 2) for v in d.bbox],
        )
        for d in detections
    ]

    message = None
    if not detections_out:
        message = (
            "No supported abnormality was detected by the computer vision "
            "model in the analyzed content."
        )

    logger.info(
        "image analyzed | analysis_id=%s | file=%s | detections=%d | %.1f ms | device=%s",
        analysis_id, file.filename, len(detections_out), result.inference_ms, result.device,
    )

    return ImageAnalysisResponse(
        analysis_id=analysis_id,
        input_file=_sanitize_filename(file.filename or ""),
        input_mime=file.content_type or "application/octet-stream",
        image_width=result.image_width,
        image_height=result.image_height,
        device=result.device,
        model_name=detector.model_path.name,
        model_classes={str(k): v for k, v in detector.class_names.items()},
        inference_ms=round(result.inference_ms, 2),
        count=len(detections_out),
        detections=detections_out,
        original_image=original_rel.as_posix(),
        annotated_image=annotated_rel.as_posix(),
        message=message,
    )
