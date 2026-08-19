"""CapsuleAI testing agents: CV -> Structured Findings -> Report -> Recs -> Review

Pipeline (presentation-ready architecture)::

    Computer Vision (YOLO)
        -> Structured Medical Findings  (backend.schemas.structured)
        -> Report Writer Agent          (this module)
        -> Recommendation Agent         (this module)
        -> Verification / Review Agent  (this module)
        -> PDF Report                   (backend.reporting)

Each agent is a stateless function that receives structured JSON and
returns markdown text. The LLM never detects objects or draws boxes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.agents.llm import GroqClient
from backend.schemas.structured import StructuredCaseReport

logger = logging.getLogger("application")

# --------------------------------------------------------------------------- #
# System prompts (one per agent)
# --------------------------------------------------------------------------- #

REPORT_SYSTEM = (
    "You are a senior gastroenterology registrar writing a capsule endoscopy "
    "report. You receive a JSON object with structured CV detections (findings "
    "with class, confidence, bounding box, anatomical location and severity). "
    "Base every statement ONLY on that JSON. Do not invent findings, locations, "
    "or patient details. Flag uncertainty explicitly. Return a clean markdown "
    "report with sections: Summary, Findings, Clinical Impression."
)

RECOMMENDATION_SYSTEM = (
    "You are a clinical decision-support specialist for capsule endoscopy. "
    "Given a structured findings JSON and a clinical report, generate a concise "
    "list of actionable clinical recommendations. Group them by priority "
    "(Immediate, Short-term, Routine). Be evidence-based. If the findings are "
    "normal or low-risk, state that no urgent action is required. Return "
    "clean markdown."
)

REVIEW_SYSTEM = (
    "You are a peer-review auditor for an AI-assisted capsule endoscopy system. "
    "You receive (1) the structured findings JSON from the CV model, (2) a "
    "clinical report, and (3) a set of clinical recommendations. Check for "
    "inconsistencies between all three: findings omitted from the report, "
    "confidence values misquoted, recommendations not supported by findings, "
    "or incorrect counts. Reply with a concise markdown review. End with an "
    "explicit verdict line: 'VERDICT: PASS' or 'VERDICT: FAIL'."
)


# --------------------------------------------------------------------------- #
# Agent trace
# --------------------------------------------------------------------------- #

@dataclass
class AgentTrace:
    """One agent's run output (for display in the testing UI)."""

    name: str
    status: str  # ok | skipped | error
    output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "output": self.output, "error": self.error}


@dataclass
class AgentRunResult:
    """Full result of running the agent pipeline."""

    case_report: StructuredCaseReport
    report: str
    recommendations: str
    review: str
    traces: List[AgentTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "case_report": self.case_report.model_dump(),
            "report": self.report,
            "recommendations": self.recommendations,
            "review": self.review,
            "traces": [t.to_dict() for t in self.traces],
        }


# --------------------------------------------------------------------------- #
# Individual agents
# --------------------------------------------------------------------------- #

class ReportWriterAgent:
    """Generates the clinical narrative from structured CV findings."""

    name = "ReportWriter"

    def __init__(self, client: GroqClient) -> None:
        self.client = client

    def run(self, findings_json: str) -> str:
        user = (
            "Below is the structured output from the Computer Vision agent for "
            "a capsule endoscopy frame. Write a clinical report.\n\n"
            f"```json\n{findings_json}\n```\n\n"
            "Return a markdown report with: **Summary**, **Findings** (table "
            "format), **Clinical Impression**. Keep it under 200 words."
        )
        return self.client.chat(user, system=REPORT_SYSTEM)


class RecommendationAgent:
    """Generates clinical recommendations from structured findings + report."""

    name = "Recommendations"

    def __init__(self, client: GroqClient) -> None:
        self.client = client

    def run(self, findings_json: str, report: str) -> str:
        user = (
            "Structured CV findings:\n"
            f"```json\n{findings_json}\n```\n\n"
            "Clinical report:\n"
            f"---\n{report}\n---\n\n"
            "Generate clinical recommendations grouped by priority: "
            "**Immediate**, **Short-term**, **Routine**."
        )
        return self.client.chat(user, system=RECOMMENDATION_SYSTEM)


class ReviewAgent:
    """Audits report + recommendations against structured findings."""

    name = "Reviewer"

    def __init__(self, client: GroqClient) -> None:
        self.client = client

    def run(self, findings_json: str, report: str, recommendations: str) -> str:
        user = (
            "Structured CV findings (ground truth):\n"
            f"```json\n{findings_json}\n```\n\n"
            "Clinical report:\n"
            f"---\n{report}\n---\n\n"
            "Clinical recommendations:\n"
            f"---\n{recommendations}\n---\n\n"
            "Audit for consistency and return your review with a verdict."
        )
        return self.client.chat(user, system=REVIEW_SYSTEM)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class CapsuleAgents:
    """Runs the full agent pipeline over a StructuredCaseReport."""

    def __init__(self, client: Optional[GroqClient] = None) -> None:
        self.client = client or GroqClient()

    def analyze(self, case_report: StructuredCaseReport) -> AgentRunResult:
        findings_json = case_report.findings_json()
        traces: List[AgentTrace] = []

        writer = ReportWriterAgent(self.client)
        recommender = RecommendationAgent(self.client)
        reviewer = ReviewAgent(self.client)

        # --- Report Writer ---
        report = ""
        try:
            report = writer.run(findings_json)
            traces.append(AgentTrace(name=writer.name, status="ok", output=report))
        except Exception as exc:
            logger.exception("ReportWriter agent failed")
            traces.append(AgentTrace(name=writer.name, status="error", error=str(exc)))
            raise

        # --- Recommendation Agent ---
        recommendations = ""
        try:
            recommendations = recommender.run(findings_json, report)
            traces.append(AgentTrace(name=recommender.name, status="ok", output=recommendations))
        except Exception as exc:
            logger.exception("Recommendation agent failed")
            traces.append(AgentTrace(name=recommender.name, status="error", error=str(exc)))
            recommendations = f"*Recommendation agent failed:* {exc}"

        # --- Reviewer Agent ---
        review = ""
        try:
            review = reviewer.run(findings_json, report, recommendations)
            traces.append(AgentTrace(name=reviewer.name, status="ok", output=review))
        except Exception as exc:
            logger.exception("Reviewer agent failed")
            traces.append(AgentTrace(name=reviewer.name, status="error", error=str(exc)))
            review = f"*Review agent failed:* {exc}"

        return AgentRunResult(
            case_report=case_report,
            report=report,
            recommendations=recommendations,
            review=review,
            traces=traces,
        )


def run_agents(
    case_report: StructuredCaseReport,
    client: Optional[GroqClient] = None,
) -> AgentRunResult:
    """Convenience wrapper."""
    return CapsuleAgents(client=client).analyze(case_report)
