"""Reusable YOLO11 detection wrapper for capsule endoscopy analysis.

The computer vision model is the *source of truth* for visual detection.
It is responsible for producing bounding boxes and confidence scores;
LLM agents never detect or draw boxes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("cv")


@dataclass
class Detection:
    """A single detection produced by the YOLO model."""

    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixels

    # Populated for video analysis (Milestone 2+).
    frame_number: Optional[int] = None
    timestamp_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox"] = [round(float(v), 2) for v in self.bbox]
        data["confidence"] = round(float(self.confidence), 4)
        return data


@dataclass
class PredictResult:
    """Result of running detection on one image/frame."""

    detections: List[Detection]
    inference_ms: float
    image_width: int
    image_height: int
    device: str


class CapsuleDetector:
    """Wrapper around an Ultralytics YOLO11 model.

    Usage::

        detector = CapsuleDetector(settings.yolo_model_path)
        detections, elapsed = detector.predict(bgr_image)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.40,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO model not found at '{self.model_path}'. "
                "Set YOLO_MODEL_PATH in .env or run "
                "`python scripts/download_demo_model.py`."
            )

        self.device = self._resolve_device(device)
        self.model = self._load_model()
        self.class_names: Dict[int, str] = self._read_class_names()

        logger.info(
            "CapsuleDetector initialized | model=%s | device=%s | classes=%d | conf=%.2f",
            self.model_path.name,
            self.device,
            len(self.class_names),
            self.conf_threshold,
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _resolve_device(self, device: Optional[str]) -> str:
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_model(self) -> YOLO:
        try:
            return YOLO(str(self.model_path))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_path}': {exc}"
            ) from exc

    def _read_class_names(self) -> Dict[int, str]:
        """Read class names from the model itself when available.

        Never assumes a fixed set of classes. Falls back to ``class_{id}``.
        """
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        return {}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, image: np.ndarray) -> PredictResult:
        """Run detection on a single BGR image (as returned by cv2.imread).

        Args:
            image: BGR numpy array.

        Returns:
            PredictResult with detections and metadata.
        """
        height, width = image.shape[:2]
        start = time.perf_counter()
        try:
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections = self._parse_results(results)
        logger.debug(
            "predict() done | image=%dx%d | detections=%d | %.1f ms",
            width, height, len(detections), elapsed_ms,
        )
        return PredictResult(
            detections=detections,
            inference_ms=elapsed_ms,
            image_width=width,
            image_height=height,
            device=self.device,
        )

    def predict_video(self, video_path: Union[str, Path], frame_interval: int = 5):
        """Frame extraction + detection for video input (Milestone 2).

        Stub kept separate so Milestone 1 stays focused on image inference.
        """
        raise NotImplementedError("Video inference lands in Milestone 2.")

    # ------------------------------------------------------------------ #
    # Result parsing
    # ------------------------------------------------------------------ #
    def _parse_results(self, results) -> List[Detection]:
        detections: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox=xyxy,
                    )
                )
        # Highest confidence first.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections


def model_exists(model_path: Union[str, Path]) -> bool:
    return Path(model_path).exists()
