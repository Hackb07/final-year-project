"""Minimal Groq (OpenAI-compatible) LLM client used by the agents."""

from __future__ import annotations

import logging
from typing import List, Optional

from backend.config import settings

logger = logging.getLogger("application")

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for capsule endoscopy "
    "analysis. Be concise, factual, and evidence-grounded. Never invent "
    "findings that were not provided. Use plain medical language and markdown "
    "formatting."
)


class GroqClient:
    """Thin wrapper over the OpenAI SDK pointed at Groq's API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self.base_url = base_url or settings.groq_base_url

        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env (get a free key at "
                "https://console.groq.com) or set the environment variable."
            )

        from openai import OpenAI

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=2,
        )

    def complete(
        self,
        messages: List[dict],
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Run a single chat completion and return the assistant text."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - network/provider errors
            raise RuntimeError(f"Groq LLM call failed: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()

    def chat(self, user: str, system: str = SYSTEM_PROMPT) -> str:
        return self.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )


def llm_available() -> bool:
    return bool(settings.groq_api_key)
