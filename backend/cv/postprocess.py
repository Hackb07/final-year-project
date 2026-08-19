"""Post-processing helpers: image decoding/validation and OpenCV annotation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from backend.cv.detector import Detection

logger = logging.getLogger("cv")

# A consistent palette so the same class always gets the same box color.
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 165, 255),    # orange
    (0, 0, 255),      # red
    (255, 0, 0),      # blue
    (0, 255, 0),      # green
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (200, 200, 200),  # light gray
]

BOX_THICKNESS = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR numpy array.

    Raises ValueError for undecodable/invalid image data.
    """
    if not data:
        raise ValueError("Empty file content provided.")
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File is not a readable image (unsupported or corrupted).")
    return image


def save_image(image: np.ndarray, path: Path) -> None:
    """Persist a BGR image. Raises RuntimeError if OpenCV cannot write it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"OpenCV failed to write image to '{path}'.")
    logger.debug("saved image -> %s (%d bytes)", path, path.stat().st_size)


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Detection],
    show_bbox: bool = True,
    show_label: bool = True,
    label_prefix: str = "",
) -> np.ndarray:
    """Return a copy of ``image_bgr`` annotated with bounding boxes + labels.

    The annotated image includes the bounding box, the disease/abnormality
    label, the confidence, and (optionally) a caller-supplied prefix such as
    a finding id.
    """
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in det.bbox)
        color = color_for_class(det.class_id)

        if show_bbox:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        if show_label:
            label = f"{det.class_name.upper()} {det.confidence:.0%}"
            if label_prefix:
                label = f"{label_prefix} {label}"
            (tw, th), baseline = cv2.getTextSize(label, FONT, 0.6, 2)
            ty = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                annotated,
                (x1, ty),
                (x1 + tw + 6, ty + th + baseline + 4),
                color,
                -1,
            )
            cv2.putText(
                annotated, label, (x1 + 3, ty + th + 2),
                FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA,
            )
    return annotated
