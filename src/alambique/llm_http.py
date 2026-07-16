"""Shared OpenCode HTTP calls with transient-failure retries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

from alambique.memory_config import (
    LLM_RETRY_BASE_DELAY_SECONDS,
    LLM_RETRY_MAX_ATTEMPTS,
)

logger = logging.getLogger("alambique.llm_http")

OPENCODE_MESSAGES_URL = "https://opencode.ai/zen/go/v1/chat/completions"
LlmOutcomeCallback = Callable[[bool, str | None, bool], None]


def extract_chat_completion_text(data: dict) -> str:
    """Pull assistant text from an OpenAI-compatible chat.completion payload.

    Handles null content (common when models only fill reasoning_content),
    missing choices, and multipart content lists.
    """
    choices = data.get("choices") or []
    if not choices:
        return ""
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if text is None:
                    text = part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


async def post_opencode_message(
    client: httpx.AsyncClient,
    api_key: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    on_outcome: LlmOutcomeCallback | None = None,
    log_prefix: str = "LLM",
) -> str:
    """POST to OpenCode Go chat/completions (OpenAI-compatible) with retries on 5xx and network errors."""
    last_error: Exception | None = None

    for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            response = await client.post(
                OPENCODE_MESSAGES_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            text = extract_chat_completion_text(data if isinstance(data, dict) else {})
            if attempt > 1:
                logger.info("%s respondió tras %d intentos", log_prefix, attempt)
            if on_outcome:
                on_outcome(True, None, attempt > 1)
            return text
        except httpx.HTTPStatusError as exc:
            last_error = exc
            if exc.response.status_code < 500:
                raise
            if attempt >= LLM_RETRY_MAX_ATTEMPTS:
                raise
            delay = LLM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s error %d (intento %d/%d): %s — reintento en %.1fs",
                log_prefix,
                exc.response.status_code,
                attempt,
                LLM_RETRY_MAX_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt >= LLM_RETRY_MAX_ATTEMPTS:
                raise
            delay = LLM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "%s error de red (intento %d/%d): %s — reintento en %.1fs",
                log_prefix,
                attempt,
                LLM_RETRY_MAX_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    if last_error:
        raise last_error
    return ""