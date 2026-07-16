"""Tests for recall engine — formatting and prompt generation."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alambique.recall import (
    RecallClient,
    _format_session_list,
    RECALL_PROMPT,
    PERSONALITY_PROMPT,
)


# ── Session formatting ───────────────────────────────────────────


class TestFormatSessionList:
    def test_empty(self):
        assert _format_session_list([]) == "(sin sesiones relacionadas)"

    def test_with_summary(self):
        sessions = [{"id": "sess_abc", "summary": "Charla sobre Python"}]
        result = _format_session_list(sessions)
        assert "sess_abc" in result
        assert "Charla sobre Python" in result

    def test_falls_back_to_snippet(self):
        sessions = [{"id": "sess_xyz", "snippet": "fragmento"}]
        result = _format_session_list(sessions)
        assert "fragmento" in result

    def test_missing_id(self):
        sessions = [{"summary": "algo"}]
        result = _format_session_list(sessions)
        assert "algo" in result


# ── Recall Prompt ────────────────────────────────────────────────


class TestRecallPrompt:
    def test_contains_placeholders(self):
        assert "{agent_name}" in RECALL_PROMPT
        assert "{query}" in RECALL_PROMPT
        assert "{top_sessions}" in RECALL_PROMPT

    def test_format_valid(self):
        formatted = RECALL_PROMPT.format(
            agent_name="lucy",
            query="juegos cooperativos",
            top_sessions="- [sess_1] Charla sobre gaming",
        )
        assert "lucy" in formatted
        assert "juegos cooperativos" in formatted
        assert "Charla sobre gaming" in formatted

    def test_max_4_frases(self):
        assert "4 frases" in RECALL_PROMPT or "máximo 4" in RECALL_PROMPT

    def test_no_inventes(self):
        assert "No inventes" in RECALL_PROMPT

    def test_respond_only_text(self):
        assert "SOLO con el texto" in RECALL_PROMPT
        assert "sin formato JSON" in RECALL_PROMPT


# ── Personality Prompt ───────────────────────────────────────────


class TestPersonalityPrompt:
    def test_contains_placeholders(self):
        assert "{traits}" in PERSONALITY_PROMPT
        assert "{moods}" in PERSONALITY_PROMPT

    def test_format_valid(self):
        formatted = PERSONALITY_PROMPT.format(
            traits="  [1.0] sarcastic: Es sarcástica",
            moods="  [0.9] hoy_depre: Sin bromas",
        )
        assert "Lucy" in formatted
        assert "sarcastic" in formatted
        assert "hoy_depre" in formatted

    def test_second_person(self):
        assert "segunda persona" in PERSONALITY_PROMPT
        assert "Eres" in PERSONALITY_PROMPT

    def test_null_when_insufficient(self):
        assert "devuelve null" in PERSONALITY_PROMPT.lower() or "null." in PERSONALITY_PROMPT

    def test_max_4_frases(self):
        assert "4 frases" in PERSONALITY_PROMPT or "máximo 4" in PERSONALITY_PROMPT

    def test_moods_at_end(self):
        assert "paréntesis" in PERSONALITY_PROMPT
        assert "hoy:" in PERSONALITY_PROMPT


# ── LLM retries ──────────────────────────────────────────────────


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/messages")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


def _ok_response(text: str = "resumen ok", *, content_null: bool = False) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    content = None if content_null else text
    mock.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }
    return mock


class TestRecallClientRetries:
    @pytest.mark.asyncio
    async def test_retries_on_500_then_succeeds(self):
        client = RecallClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(
            side_effect=[
                _http_status_error(500),
                _ok_response("tras reintento"),
            ]
        )

        with patch("alambique.llm_http.asyncio.sleep", new_callable=AsyncMock):
            result = await client._call_llm("prompt")

        assert result == "tras reintento"
        assert client._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_attempts(self):
        client = RecallClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=_http_status_error(500))

        with patch("alambique.llm_http.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await client._call_llm("prompt")

        assert client._client.post.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_4xx(self):
        client = RecallClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=_http_status_error(401))

        with pytest.raises(httpx.HTTPStatusError):
            await client._call_llm("prompt")

        client._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        client = RecallClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(
            side_effect=[
                httpx.ReadTimeout("timeout"),
                _ok_response(),
            ]
        )

        with patch("alambique.llm_http.asyncio.sleep", new_callable=AsyncMock):
            result = await client._call_llm("prompt")

        assert result == "resumen ok"
        assert client._client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_null_content_returns_empty_string(self):
        """OpenCode sometimes returns content: null (reasoning-only); must not crash on .strip()."""
        client = RecallClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(return_value=_ok_response(content_null=True))

        result = await client._call_llm("prompt")
        assert result == ""
        assert (result or "").strip() == ""
