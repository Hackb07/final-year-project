"""Branded PDF report generation for capcell-ai."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import markdown as md_lib
import numpy as np
from xhtml2pdf import pisa

from backend.config import settings
from backend.schemas.structured import StructuredCaseReport, StructuredFinding

logger = logging.getLogger("application")

COMPANY_NAME = "CAPCELL-AI"
COMPANY_TAGLINE = "AI-Assisted Capsule Endoscopy Analysis"
REPORT_TITLE = "Endoscopy Analysis Report"

PRIMARY = "#0B5D5E"   # deep teal
DARK = "#1F2933"      # slate
MUTED = "#6B7280"     # gray
LIGHT = "#F4F7F7"     # near-white tint

PDF_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
    size: A4;
    margin: 2.1cm 1.9cm 2.3cm 1.9cm;
    @frame footer {{
      -pdf-frame-content: footer;
      left: 1.9cm; right: 1.9cm; top: 27.7cm; height: 1.2cm;
    }}
  }}

  body {{
    font-family: Helvetica;
    font-size: 10pt;
    color: {dark};
    line-height: 1.55;
  }}

  /* ---- Letterhead ---- */
  .letterhead {{
    border-bottom: 3px solid {primary};
    padding-bottom: 10pt;
    margin-bottom: 14pt;
  }}
  .company {{
    font-size: 24pt;
    font-weight: bold;
    color: {primary};
    letter-spacing: 2pt;
  }}
  .tagline {{ font-size: 9pt; color: {muted}; margin-top: 2pt; }}
  .doctitle {{
    font-size: 14pt;
    font-weight: bold;
    color: {dark};
    margin-top: 8pt;
    text-transform: uppercase;
  }}

  /* ---- Metadata grid ---- */
  table.meta {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 14pt;
  }}
  table.meta td {{
    padding: 4pt 8pt;
    font-size: 8.5pt;
    border: 1px solid #E5E7EB;
    background: {light};
  }}
  table.meta .k {{ color: {muted}; font-weight: bold; width: 18%; }}
  table.meta .v {{ color: {dark}; }}

  /* ---- Section headings ---- */
  h2.section {{
    color: {primary};
    font-size: 12pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5pt;
    border-bottom: 1px solid #D1E0E0;
    padding-bottom: 4pt;
    margin: 16pt 0 8pt 0;
  }}

  /* ---- Findings table ---- */
  table.det {{
    width: 100%;
    border-collapse: collapse;
    margin: 6pt 0 4pt 0;
  }}
  table.det th {{
    background: {primary};
    color: #FFFFFF;
    padding: 6pt 8pt;
    text-align: left;
    font-size: 8.5pt;
    border: 1px solid {primary};
  }}
  table.det td {{
    padding: 6pt 8pt;
    font-size: 8.5pt;
    border: 1px solid #E5E7EB;
    color: {dark};
  }}
  .nofindings {{
    font-style: italic;
    color: {muted};
    background: {light};
    border: 1px solid #E5E7EB;
    padding: 10pt;
  }}

  /* ---- Result image ---- */
  .figure {{ text-align: center; margin: 8pt 0; }}
  .figure img {{ max-width: 78%; }}
  .figure .caption {{
    font-size: 8pt;
    color: {muted};
    margin-top: 4pt;
  }}

  /* ---- Markdown content ---- */
  .content h1, .content h2, .content h3, .content h4 {{
    color: {primary};
    margin: 10pt 0 4pt 0;
  }}
  .content h1 {{ font-size: 11.5pt; }}
  .content h2 {{ font-size: 11pt; }}
  .content h3, .content h4 {{ font-size: 10.5pt; }}
  .content ul, .content ol {{ margin: 4pt 0 4pt 14pt; }}
  .content li {{ margin-bottom: 2pt; }}
  .content strong {{ color: {dark}; }}
  .content table {{ width: 100%; border-collapse: collapse; margin: 6pt 0; }}
  .content table th {{
    background: {light};
    color: {primary};
    border: 1px solid #E5E7EB;
    padding: 5pt 7pt;
    font-size: 9pt;
    text-align: left;
  }}
  .content table td {{
    border: 1px solid #E5E7EB;
    padding: 5pt 7pt;
    font-size: 9pt;
  }}

  /* ---- Structured JSON block ---- */
  .jsonblock {{
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    padding: 10pt;
    font-size: 8pt;
    font-family: Courier;
    line-height: 1.45;
    white-space: pre;
    color: {dark};
    margin: 6pt 0;
  }}

  /* ---- Disclaimer box ---- */
  .disclaimer {{
    margin-top: 18pt;
    padding: 10pt 12pt;
    background: {light};
    border-left: 4px solid {primary};
    font-size: 8pt;
    color: {muted};
    line-height: 1.5;
  }}

  .attestation {{
    margin-top: 22pt;
    font-size: 8.5pt;
    color: {muted};
    border-top: 1px solid #E5E7EB;
    padding-top: 8pt;
  }}

  /* ---- Footer ---- */
  #footer {{ font-size: 8pt; color: {muted}; text-align: center; }}
  #footer .brand {{ color: {primary}; font-weight: bold; }}
</style>
</head>
<body>

  <div class="letterhead">
    <div class="company">{company}</div>
    <div class="tagline">{tagline}</div>
    <div class="doctitle">{report_title}</div>
  </div>

  <table class="meta">
    <tr>
      <td class="k">Case ID</td><td class="v">{case_id}</td>
      <td class="k">Report Date</td><td class="v">{timestamp}</td>
    </tr>
    <tr>
      <td class="k">Detection Model</td><td class="v">{model_name}</td>
      <td class="k">Inference Device</td><td class="v">{device}</td>
    </tr>
    <tr>
      <td class="k">Total Findings</td><td class="v">{total_findings}</td>
      <td class="k">Image Size</td><td class="v">{image_size}</td>
    </tr>
  </table>

  <h2 class="section">CV Findings (Structured)</h2>
  {findings_table}

  <div class="figure">
    <img src="{annotated_image}"/>
    <div class="caption">Fig. 1 - Annotated frame with CV model detections (bounding boxes and class labels).</div>
  </div>

  <h2 class="section">Clinical Report</h2>
  <div class="content">{report_html}</div>

  <h2 class="section">Recommendations</h2>
  <div class="content">{recommendations_html}</div>

  <h2 class="section">Consistency Review</h2>
  <div class="content">{review_html}</div>

  <div class="disclaimer">
    <strong>Disclaimer.</strong> This report was generated by an AI-assisted
    decision-support system for evaluation purposes only. It is not a substitute
    for a physician&rsquo;s clinical judgement. Findings must be confirmed by a
    qualified endoscopist. This document is confidential and proprietary to
    {company}.
  </div>

  <div class="attestation">
    Generated by {company} &middot; {tagline} &middot; Document {case_id}
  </div>

  <div id="footer">
    <span class="brand">{company}</span> &nbsp;|&nbsp; {report_title} &nbsp;|&nbsp;
    Page <pdf:pagenumber/> of <pdf:pagecount/>
  </div>

</body>
</html>
"""


def _findings_table(findings: List[StructuredFinding]) -> str:
    if not findings:
        return (
            '<p class="nofindings">No detections were found above the '
            "confidence threshold in the analyzed content.</p>"
        )
    rows = "".join(
        f"<tr>"
        f"<td>{f.finding_id}</td>"
        f"<td>{f.class_name}</td>"
        f"<td>{f.confidence:.2f}</td>"
        f"<td>{f.anatomical_location}</td>"
        f"<td>{f.severity}</td>"
        f"<td>{round(f.bbox[0],1)}, {round(f.bbox[1],1)}, "
        f"{round(f.bbox[2],1)}, {round(f.bbox[3],1)}</td>"
        f"</tr>"
        for f in findings
    )
    return (
        "<table class='det'>"
        "<tr><th>Finding ID</th><th>Class</th><th>Confidence</th>"
        "<th>Anatomical Location</th><th>Severity</th><th>BBox (x1,y1,x2,y2)</th></tr>"
        f"{rows}</table>"
    )


def build_pdf_report(
    case_report: StructuredCaseReport,
    report_md: str,
    recommendations_md: str,
    review_md: str,
    output_dir: Path,
    company: str = COMPANY_NAME,
    annotated_image: Optional[np.ndarray] = None,
) -> Path:
    """Render the structured report into a branded PDF and return its path."""
    case_id = case_report.case_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_src = ""
    if annotated_image is not None:
        img_path = output_dir / f"{case_id}_annotated.jpg"
        ok = cv2.imwrite(str(img_path), annotated_image)
        if ok:
            annotated_src = str(img_path.resolve()).replace("\\", "/")
        else:
            logger.warning("cv2.imwrite failed for annotated image: %s", img_path)

    report_html = md_lib.markdown(
        report_md or "_No report was generated._", extensions=["tables"]
    )
    recommendations_html = md_lib.markdown(
        recommendations_md or "_No recommendations were generated._", extensions=["tables"]
    )
    review_html = md_lib.markdown(
        review_md or "_No review was generated._", extensions=["tables"]
    )

    html = PDF_TEMPLATE.format(
        company=company,
        tagline=COMPANY_TAGLINE,
        report_title=REPORT_TITLE,
        case_id=case_id,
        timestamp=case_report.timestamp,
        model_name=case_report.model_name,
        device=case_report.device,
        total_findings=case_report.total_findings,
        image_size=f"{case_report.image_width} x {case_report.image_height}",
        findings_table=_findings_table(case_report.findings),
        annotated_image=annotated_src,
        report_html=report_html,
        recommendations_html=recommendations_html,
        review_html=review_html,
        primary=PRIMARY,
        dark=DARK,
        muted=MUTED,
        light=LIGHT,
    )

    pdf_path = output_dir / f"{case_id}.pdf"
    with open(pdf_path, "wb") as fh:
        status = pisa.CreatePDF(html, dest=fh, encoding="utf-8")
        if status.err:
            raise RuntimeError(
                f"xhtml2pdf failed to build '{pdf_path}' (err={status.err})."
            )

    logger.info("PDF report written -> %s (%d bytes)", pdf_path, pdf_path.stat().st_size)
    return pdf_path


def report_output_dir() -> Path:
    """Directory where generated PDF reports are saved."""
    path = settings.output_dir / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path
