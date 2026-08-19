"""Gradio web UI to test capcell-ai: YOLO detection + LLM report agents.

Run with::

    uv run python webui.py

Set GROQ_API_KEY in `.env` to enable the report/review agents (free key at
https://console.groq.com). Without it, detection still works and the agents
are skipped with a notice.

Architecture (presentation-ready):
    Computer Vision (YOLO)
        -> Structured Medical Findings (JSON)
        -> LLM Report Writer Agent
        -> LLM Recommendation Agent
        -> LLM Verification / Review Agent
        -> PDF Report (Suskbs)
"""

from __future__ import annotations

import logging
import uuid
from typing import Tuple

import gradio as gr
import numpy as np

from backend.agents import CapsuleAgents, llm_available
from backend.config import settings, setup_logging
from backend.cv.detector import CapsuleDetector, model_exists
from backend.cv.postprocess import draw_detections
from backend.reporting import COMPANY_NAME, build_pdf_report, report_output_dir
from backend.schemas.structured import structure_detections

setup_logging()
logger = logging.getLogger("application")


def load_detector() -> CapsuleDetector:
    if not model_exists(settings.yolo_model_path):
        raise SystemExit(
            f"Model not found at '{settings.yolo_model_path}'. "
            "Set YOLO_MODEL_PATH in .env or place the trained weights there."
        )
    return CapsuleDetector(
        model_path=settings.yolo_model_path,
        conf_threshold=settings.confidence_threshold,
        device=settings.device,
    )


detector = load_detector()
agents = CapsuleAgents() if llm_available() else None


def run_pipeline(
    image: np.ndarray,
) -> Tuple[np.ndarray, str, str, str, str, str, str, str]:
    """Detection -> Structured Findings -> Agents -> PDF."""
    if image is None:
        return (
            None,
            "Please upload an image first.",
            "",
            "No report.",
            "No recommendations.",
            "No review.",
            "No trace.",
            None,
        )

    # --- CV Agent: YOLO detection ---
    result = detector.predict(image)
    annotated = draw_detections(image, result.detections)

    # --- Structure findings ---
    case_id = f"CASE_{uuid.uuid4().hex[:8].upper()}"
    case_report = structure_detections(
        detections=result.detections,
        case_id=case_id,
        frame_id=None,
        model_name=detector.model_path.name,
        device=result.device,
        image_width=result.image_width,
        image_height=result.image_height,
    )

    status = (
        f"{case_report.total_findings} finding(s) | "
        f"inference {result.inference_ms:.1f} ms | device {result.device}"
    )

    findings_json = case_report.findings_json()

    # --- LLM Agents (if available) ---
    if agents is None:
        notice = (
            "**LLM agents disabled.** Set `GROQ_API_KEY` in `.env` "
            "(free key at https://console.groq.com), then restart."
        )
        pdf_path = build_pdf_report(
            case_report=case_report,
            report_md=notice,
            recommendations_md=notice,
            review_md=notice,
            output_dir=report_output_dir(),
            company=COMPANY_NAME,
            annotated_image=annotated,
        )
        return (
            annotated, status, findings_json,
            notice, notice, notice, "No trace.", str(pdf_path),
        )

    logger.info(
        "running agents | model=%s | case=%s",
        settings.groq_model, case_id,
    )
    agent_result = agents.analyze(case_report)

    trace_md = "\n\n".join(
        f"### {t.name} [{t.status}]"
        + (f"\n\n{t.output}" if t.output else "")
        + (f"\n\n**error:** {t.error}" if t.error else "")
        for t in agent_result.traces
    )

    # --- PDF Report (Suskbs) ---
    pdf_path = build_pdf_report(
        case_report=case_report,
        report_md=agent_result.report,
        recommendations_md=agent_result.recommendations,
        review_md=agent_result.review,
        output_dir=report_output_dir(),
        company=COMPANY_NAME,
        annotated_image=annotated,
    )

    return (
        annotated,
        status,
        findings_json,
        agent_result.report,
        agent_result.recommendations,
        agent_result.review,
        trace_md,
        str(pdf_path),
    )


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #

with gr.Blocks(title="CapsuleAI - Detection + Agents Test") as demo:
    gr.Markdown(
        f"""
# CapsuleAI - Detection + LLM Agents Test

Model: **`{detector.model_path.name}`** | Device: **`{detector.device}`** |
Classes: **{', '.join(detector.class_names.values())}** |
Confidence threshold: **{detector.conf_threshold:.2f}** |
LLM: **{settings.groq_model if agents else 'DISABLED (no GROQ_API_KEY)'}**

**Architecture:** CV Agent (YOLO) -> Structured Findings (JSON) ->
Report Writer -> Recommendation Agent -> Review Agent -> PDF
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(
                label="Upload capsule endoscopy image",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            run_btn = gr.Button("Run Detection + Agents", variant="primary")
        with gr.Column(scale=1):
            image_out = gr.Image(label="Annotated result", type="numpy")
            status = gr.Textbox(label="Status", interactive=False)

    with gr.Tab("Structured Findings (JSON)"):
        findings_json_out = gr.Code(
            label="Computer Vision Agent Output",
            language="json",
        )

    with gr.Tab("Clinical Report"):
        report_md_out = gr.Markdown(label="ReportWriter Agent")

    with gr.Tab("Recommendations"):
        recs_md_out = gr.Markdown(label="Recommendation Agent")

    with gr.Tab("Consistency Review"):
        review_md_out = gr.Markdown(label="Reviewer Agent")

    with gr.Tab("Agent Trace"):
        trace_md_out = gr.Markdown(label="Agent Pipeline Trace")

    pdf_file = gr.File(label=f"PDF Report ({COMPANY_NAME})")

    run_btn.click(
        run_pipeline,
        inputs=image_in,
        outputs=[
            image_out, status, findings_json_out,
            report_md_out, recs_md_out, review_md_out,
            trace_md_out, pdf_file,
        ],
    )


if __name__ == "__main__":
    demo.launch()
