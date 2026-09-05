"""Thin OpenRouter client with full usage capture.

Returns per-call token breakdowns (prompt, completion, reasoning) and
cost in credits, so downstream runners can compute real-world cost
rather than relying on stated per-token prices.

Usage:
    from or_client import ORClient
    client = ORClient()
    result = client.chat(model="z-ai/glm-5.3-flash", messages=[...])
    # result.content, result.tool_calls, result.usage, result.wallclock_s
"""
from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_TIMEOUT = 300  # 5 min — reasoning models can be slow at max effort


@dataclass
class CallResult:
    """One OpenRouter chat completion call."""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    wallclock_s: float = 0.0
    retries: int = 0
    error: str | None = None
    finish_reason: str | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def reasoning_tokens(self) -> int:
        details = self.usage.get("completion_tokens_details") or {}
        return details.get("reasoning_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def cost(self) -> float:
        return self.usage.get("cost", 0.0)


class ORClient:
    """OpenRouter chat client with automatic usage capture."""

    def __init__(self, api_key: str | None = None, max_retries: int = 3):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set — export it before running."
            )
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning_effort: str = "max",
        temperature: float = 0.0,
    ) -> CallResult:
        """Single non-streaming chat completion.

        Returns CallResult with usage breakdown. Retries on 429/5xx
        with exponential backoff. Records retry count.

        Args:
            model: OpenRouter model slug (e.g. "z-ai/glm-5.3-flash")
            messages: Chat messages list
            tools: Optional tools list (OpenAI function format)
            reasoning_effort: One of "max", "high", "low" (both models
                support exactly these three values per OpenRouter API)
            temperature: 0.0 for best-effort determinism
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "reasoning": {"effort": reasoning_effort},
        }
        if tools:
            payload["tools"] = tools

        last_err: str | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.session.post(
                    OR_URL, json=payload, timeout=OR_TIMEOUT
                )
                wallclock = time.time() - t0

                if resp.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {resp.status_code}"
                    if attempt < self.max_retries:
                        wait = 2 ** attempt
                        logger.warning(
                            "OR %s, retry %d/%d after %ds",
                            last_err, attempt + 1, self.max_retries, wait,
                        )
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                data = resp.json()

                choice = data.get("choices", [{}])[0]
                msg = choice.get("message", {})
                usage = data.get("usage", {})

                tool_calls = []
                raw_tc = msg.get("tool_calls") or []
                for tc in raw_tc:
                    fn = tc.get("function", {})
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        args = {"_raw": args_str, "_malformed": True}
                    tool_calls.append({
                        "id": tc.get("id"),
                        "name": fn.get("name"),
                        "args": args,
                    })

                return CallResult(
                    content=msg.get("content") or "",
                    tool_calls=tool_calls,
                    usage=usage,
                    wallclock_s=wallclock,
                    retries=attempt,
                    finish_reason=choice.get("finish_reason"),
                )

            except requests.RequestException as e:
                wallclock = time.time() - t0
                last_err = str(e)
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "OR request error, retry %d/%d after %ds: %s",
                        attempt + 1, self.max_retries, wait, e,
                    )
                    time.sleep(wait)
                    continue
                return CallResult(
                    error=str(e),
                    wallclock_s=wallclock,
                    retries=attempt,
                )

        return CallResult(error=last_err or "unknown", retries=self.max_retries)
