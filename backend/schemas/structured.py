"""Structured medical findings schema.

This is the intermediate representation between the Computer Vision agent
(raw YOLO detections) and the LLM agents (report writer, recommendation
agent, reviewer).  Every downstream consumer operates on this schema, never
on the raw model output directly.

Architecture (final-year presentation):
    Computer Vision (YOLO)
        -> Structured Medical Findings (this module)
        -> LLM Report Writer Agent
        -> Recommendation Agent
        -> Verification / Review Agent
        -> PDF Report
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from backend.cv.detector import Detection


class StructuredFinding(BaseModel):
    """A single clinical finding derived from a CV detection."""

    finding_id: str = Field(description="Unique ID for this finding within the case.")
    class_name: str = Field(description="Disease/abnormality class predicted by the CV model.")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score.")
    bbox: List[float] = Field(
        min_length=4, max_length=4,
        description="Bounding box [x1, y1, x2, y2] in pixels.",
    )
    anatomical_location: str = Field(
        default="unknown",
        description="Anatomical region, e.g. 'esophagus', 'cecum', or 'unknown'.",
    )
    severity: str = Field(
        default="unknown",
        description="Severity assessment, e.g. 'mild', 'moderate', 'severe', or 'unknown'.",
    )


class StructuredCaseReport(BaseModel):
    """Full structured output of the Computer Vision agent for one image/frame.

    This is the JSON that is passed verbatim to the LLM agents.
    """

    case_id: str = Field(description="Unique identifier for this analysis case.")
    frame_id: Optional[int] = Field(
        default=None,
        description="Video frame number, or null for still images.",
    )
    timestamp: str = Field(description="ISO-8601 timestamp of when the report was created.")
    model_name: str = Field(description="Name of the CV model used for detection.")
    device: str = Field(description="Inference device (e.g. 'cuda', 'cpu').")
    image_width: int = Field(description="Width of the input image in pixels.")
    image_height: int = Field(description="Height of the input image in pixels.")
    findings: List[StructuredFinding] = Field(description="List of clinical findings detected.")
    total_findings: int = Field(description="Total number of findings.")

    def findings_json(self) -> str:
        """Compact JSON for LLM agent consumption."""
        return self.model_dump_json(indent=2)

    def findings_summary(self) -> str:
        """One-line human-readable summary."""
        if not self.findings:
            return "No findings detected."
        classes = ", ".join(sorted({f.class_name for f in self.findings}))
        return f"{self.total_findings} finding(s): {classes}"


# --------------------------------------------------------------------------- #
# Converter: raw YOLO detections -> StructuredCaseReport
# --------------------------------------------------------------------------- #

# Heuristic anatomical location mapping based on class name.
_LOCATION_MAP = {
    "normal-z-line": "gastroesophageal junction",
    "esophagitis": "esophagus",
    "normal-pylorus": "pylorus",
    "normal-cecum": "cecum",
    "polyps": "colon",
    "dyed-lifted-polyps": "colon (lifted lesion)",
    "dyed-resection-margins": "colon (resection margin)",
    "ulcerative-colities": "colon (ulcerative colitis)",
}


def structure_detections(
    detections: List[Detection],
    case_id: str = "",
    frame_id: Optional[int] = None,
    model_name: str = "",
    device: str = "",
    image_width: int = 0,
    image_height: int = 0,
) -> StructuredCaseReport:
    """Convert raw YOLO detections into a StructuredCaseReport."""
    case_id = case_id or f"CASE_{uuid.uuid4().hex[:8].upper()}"

    findings = [
        StructuredFinding(
            finding_id=f"FND_{uuid.uuid4().hex[:6].upper()}",
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox=[round(float(v), 2) for v in d.bbox],
            anatomical_location=_LOCATION_MAP.get(
                d.class_name.lower(), "unknown"
            ),
            severity="unknown",
        )
        for d in detections
    ]

    return StructuredCaseReport(
        case_id=case_id,
        frame_id=frame_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        model_name=model_name,
        device=device,
        image_width=image_width,
        image_height=image_height,
        findings=findings,
        total_findings=len(findings),
    )
