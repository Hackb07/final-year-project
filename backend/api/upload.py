"""Upload endpoints (Milestone 1: images only)."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.schemas.findings import UploadResponse

logger = logging.getLogger("application")

router = APIRouter(prefix="/api", tags=["upload"])

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(filename: str) -> str:
    """Strip path components and unsafe characters from an uploaded filename."""
    name = Path(filename or "upload").name
    name = _SAFE_NAME.sub("_", name).strip("._")
    return name or "upload"


def validate_upload(file: UploadFile) -> None:
    """Validate file size and extension. Raises HTTPException on failure."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_image_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext or 'unknown'}'. "
                f"Allowed: {', '.join(settings.allowed_image_extensions)}"
            ),
        )
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File exceeds the maximum allowed size "
                f"({settings.max_upload_size_mb} MB)."
            ),
        )


@router.post("/upload/image", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Validate and store an uploaded image without analyzing it."""
    validate_upload(file)

    data = await file.read()
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum allowed size ({settings.max_upload_size_mb} MB).",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    ext = Path(file.filename or "").suffix.lower()
    file_id = uuid.uuid4().hex
    safe_name = _sanitize_filename(file.filename or "")
    stored_path = settings.upload_dir / f"{file_id}{ext}"
    stored_path.write_bytes(data)

    logger.info("image uploaded | file_id=%s | original=%s | size=%d", file_id, safe_name, len(data))

    return UploadResponse(
        file_id=file_id,
        filename=safe_name,
        stored_path=stored_path.as_posix(),
        size_bytes=len(data),
    )
