"""Tests for consolidator — formatting, JSON parsing, prompt generation."""

import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alambique.consolidator import (
    _format_conversation,
    _parse_llm_json,
    CONSOLIDATION_PROMPT,
    fetch_api_key,
    get_api_key,
)
from alambique.models import Message


# ── Formatting ───────────────────────────────────────────────────


class TestPromptAnchoringRules:
    def test_prompt_mentions_anchor_default_omit(self):
        assert "ANCLAJE AL TRANSCRIPT" in CONSOLIDATION_PROMPT
        assert "obliga a actualizarlo" in CONSOLIDATION_PROMPT
        assert "Update fantasma" in CONSOLIDATION_PROMPT


class TestFormatConversation:
    def test_format_single_message(self):
        msgs = [Message(session_id="s", role="user", content="Hola")]
        result = _format_conversation(msgs)
        assert result == "[user]: Hola"

    def test_format_multiple_messages(self):
        msgs = [
            Message(session_id="s", role="user", content="Hola"),
            Message(session_id="s", role="assistant", content="¿Qué tal?"),
        ]
        result = _format_conversation(msgs)
        assert result == "[user]: Hola\n[assistant]: ¿Qué tal?"

    def test_format_empty(self):
        result = _format_conversation([])
        assert result == ""

    def test_format_preserves_newlines_in_content(self):
        msgs = [Message(session_id="s", role="user", content="línea 1\nlínea 2")]
        result = _format_conversation(msgs)
        assert "[user]: línea 1\nlínea 2" in result


# ── JSON Parsing ─────────────────────────────────────────────────


class TestParseLLMJson:
    def test_clean_json(self):
        content = '{"threads": [], "relationship_capsules": [], "echoes": []}'
        result = _parse_llm_json(content)
        assert result["threads"] == []
        assert "relationship_capsules" in result

    def test_json_with_markdown_fence(self):
        content = '```json\n{"threads": [], "session_state": "x"}\n```'
        result = _parse_llm_json(content)
        assert result.get("session_state") == "x" or result.get("threads") == []

    def test_json_with_fence_only(self):
        content = '```\n{"threads": [], "current_state_example": "x"}\n```'
        result = _parse_llm_json(content)
        assert "threads" in result

    def test_json_with_preceding_text(self):
        content = 'Aquí está el resultado:\n\n{"threads": [{}], "relationship_capsules": []}'
        result = _parse_llm_json(content)
        assert "threads" in result

    def test_json_with_trailing_text(self):
        content = '{"threads": [], "relationship_capsules": []}\nEspero que sirva.'
        result = _parse_llm_json(content)
        assert result["threads"] == []

    def test_json_nested_objects(self):
        content = """{
            "threads": [
                {
                    "action": "create",
                    "key": "philosophy_embodiment",
                    "title": "Filosofía y embodiment",
                    "current_state": "Explorando cuerpo robótico",
                    "tone_guidance": "reflexivo",
                    "salience": 0.9
                }
            ],
            "relationship_capsules": [],
            "echoes": []
        }"""
        result = _parse_llm_json(content)
        assert len(result["threads"]) == 1
        assert result["threads"][0]["key"] == "philosophy_embodiment"

    def test_finds_json_with_embedded_braces(self):
        content = '{"threads": [{"action": "create", "key": "llaves_{}", "title": "test"}], "relationship_capsules": []}'
        result = _parse_llm_json(content)
        assert result["threads"][0]["key"] == "llaves_{}"

    def test_invalid_json(self):
        # Now robust: returns empty dict on failure (logs warning)
        result = _parse_llm_json("not json at all")
        assert result == {}


# ── Prompt ───────────────────────────────────────────────────────


class TestConsolidationPrompt:
    def test_prompt_contains_all_placeholders(self):
        # New redesign placeholders (thematic threads)
        assert "{conversation}" in CONSOLIDATION_PROMPT
        assert "{existing_threads}" in CONSOLIDATION_PROMPT
        assert "{existing_capsules}" in CONSOLIDATION_PROMPT

    def test_prompt_format_valid(self):
        formatted = CONSOLIDATION_PROMPT.format(
            conversation="[user]: Hola",
            existing_threads="- memory_redesign: ...",
            existing_capsules="Relación cálida...",
        )
        assert "[user]: Hola" in formatted
        assert "memory_redesign" in formatted or "Threads existentes" in formatted

    def test_prompt_includes_action_types(self):
        # Threads/capsules/echoes use create/update/merge
        assert "create" in CONSOLIDATION_PROMPT
        assert "update" in CONSOLIDATION_PROMPT
        assert "merge" in CONSOLIDATION_PROMPT

    def test_prompt_includes_new_thematic_concepts(self):
        assert "hilo" in CONSOLIDATION_PROMPT.lower() or "Threads" in CONSOLIDATION_PROMPT
        assert "current_state" in CONSOLIDATION_PROMPT
        assert "tone_guidance" in CONSOLIDATION_PROMPT
        assert "echoes" in CONSOLIDATION_PROMPT.lower() or "ECHOES" in CONSOLIDATION_PROMPT
        assert "relationship_capsules" in CONSOLIDATION_PROMPT.lower() or "RelationshipCapsules" in CONSOLIDATION_PROMPT
        assert "salience" in CONSOLIDATION_PROMPT
        assert "reason" in CONSOLIDATION_PROMPT  # obligatorio para auditoría
        assert "EJEMPLOS NEGATIVOS" in CONSOLIDATION_PROMPT or "evita esto" in CONSOLIDATION_PROMPT.lower()

    def test_prompt_format_does_not_break_on_json_example(self):
        # Critical: the example JSON must be escaped {{ }} so .format succeeds
        formatted = CONSOLIDATION_PROMPT.format(
            conversation="test",
            existing_threads="",
            existing_capsules="",
        )
        assert '"threads"' in formatted
        assert '"current_state"' in formatted

    def test_prompt_spanish_and_quality_focus(self):
        # Still requires valid JSON and Spanish context
        assert "JSON válido" in CONSOLIDATION_PROMPT or "JSON estricto" in CONSOLIDATION_PROMPT
        assert "Calidad sobre cantidad" in CONSOLIDATION_PROMPT

    def test_prompt_includes_open_questions_and_description(self):
        assert "open_questions" in CONSOLIDATION_PROMPT
        assert "description" in CONSOLIDATION_PROMPT
        assert "EJEMPLOS NEGATIVOS" in CONSOLIDATION_PROMPT




# ── get_api_key ──────────────────────────────────────────────────


class TestGetApiKey:
    @patch("alambique.consolidator.subprocess.run")
    def test_returns_key_on_success(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="secret-key\n", stderr="")
        assert get_api_key() == "secret-key"

    @patch("alambique.consolidator.subprocess.run")
    def test_returns_none_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        assert get_api_key() is None

    @patch("alambique.consolidator.subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_pass_not_installed(self, mock_run):
        assert get_api_key() is None

    @patch("alambique.consolidator.subprocess.run")
    def test_pass_uses_human_timeout(self, mock_run):
        import subprocess

        mock_run.return_value = MagicMock(returncode=0, stdout="secret-key\n", stderr="")
        get_api_key()
        _, kwargs = mock_run.call_args
        assert kwargs["timeout"] == 120
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] is None

    @patch("alambique.consolidator.subprocess.run", side_effect=subprocess.TimeoutExpired("pass", 120))
    def test_returns_none_on_pass_timeout(self, mock_run):
        assert get_api_key() is None

    @patch("alambique.consolidator.subprocess.run")
    def test_fetch_api_key_reports_timeout_reason(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired("pass", 120)
        result = fetch_api_key()
        assert result.key is None
        assert "120s" in (result.error or "")
        assert "pinentry" in (result.error or "").lower()

    @patch("alambique.consolidator.subprocess.run")
    def test_fetch_api_key_reports_env_source(self, mock_run, monkeypatch):
        monkeypatch.setenv("ALAMBIQUE_API_KEY", "from-env")
        result = fetch_api_key()
        assert result.key == "from-env"
        assert result.source == "ALAMBIQUE_API_KEY"
        mock_run.assert_not_called()


class TestConsolidatorRetries:
    @pytest.mark.asyncio
    async def test_retries_on_500_then_parses_response(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        import httpx

        from alambique.consolidator import ConsolidatorClient
        from alambique.models import Message

        request = httpx.Request("POST", "https://opencode.ai/zen/go/v1/chat/completions")
        error_500 = httpx.HTTPStatusError(
            "error",
            request=request,
            response=httpx.Response(500, request=request),
        )
        # OpenAI-compatible chat.completion payload (current OpenCode Go path)
        payload_text = (
            '```json\n{"threads":[{"key":"test_thread","title":"Test",'
            '"current_state":"ok state long enough for the model",'
            '"tone_guidance":"neutral","search_text":"test","action":"update",'
            '"salience":0.7,"reason":"retry test"}],'
            '"relationship_capsules":[{"scope":"general","content":"updated capsule"}],'
            '"echoes":[{"thread_key":"test_thread","content":"echo1","salience":0.6}],'
            '"lucy_initiative":null}\n```'
        )
        ok = MagicMock()
        ok.raise_for_status = MagicMock()
        ok.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": payload_text,
                    }
                }
            ]
        }

        client = ConsolidatorClient("test-key")
        client._client = AsyncMock()
        client._client.post = AsyncMock(side_effect=[error_500, ok])

        messages = [
            Message(session_id="sess_1", role="user", content="hola", timestamp="2026-07-12")
        ]

        with patch("alambique.llm_http.asyncio.sleep", new_callable=AsyncMock):
            result = await client.consolidate("Lucy", messages)

        assert len(result.threads) == 1
        assert result.threads[0]["current_state"] == "ok state long enough for the model"
        assert result.lucy_initiative is None
        assert client._client.post.call_count == 2
