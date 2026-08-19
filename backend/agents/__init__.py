"""Multi-agent layer for CapsuleAI (testing/demo scope).

Pipeline:
    Computer Vision (YOLO) -> Structured Findings -> LLM agents -> PDF
"""

from __future__ import annotations

from backend.agents.agents import CapsuleAgents, AgentRunResult, run_agents
from backend.agents.llm import GroqClient, llm_available

__all__ = [
    "AgentRunResult",
    "CapsuleAgents",
    "GroqClient",
    "llm_available",
    "run_agents",
]
