"""Tests for consolidator — formatting, JSON parsing, prompt generation."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alambique.consolidator import (
    _format_conversation,
    _format_existing_facts,
    _parse_llm_json,
    CONSOLIDATION_PROMPT,
    get_api_key,
)
from alambique.models import Fact, FactCategory, Message


# ── Formatting ───────────────────────────────────────────────────


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


class TestFormatExistingFacts:
    def test_format_empty(self):
        assert _format_existing_facts([]) == "(no hay hechos previos)"

    def test_format_single_fact(self):
        facts = [
            Fact(
                id=1,
                key="nombre",
                value="Víctor",
                category=FactCategory.PERSONAL,
            )
        ]
        result = _format_existing_facts(facts)
        assert "[id=1]" in result
        assert "(personal)" in result
        assert "nombre: Víctor" in result

    def test_format_multiple_facts(self):
        facts = [
            Fact(id=1, key="a", value="v1", category=FactCategory.PERSONAL),
            Fact(id=2, key="b", value="v2", category=FactCategory.PERSONALITY),
        ]
        result = _format_existing_facts(facts)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "[id=1]" in lines[0]
        assert "[id=2]" in lines[1]


# ── JSON Parsing ─────────────────────────────────────────────────


class TestParseLLMJson:
    def test_clean_json(self):
        content = '{"facts": [], "session_summary": "nada"}'
        result = _parse_llm_json(content)
        assert result["facts"] == []
        assert result["session_summary"] == "nada"

    def test_json_with_markdown_fence(self):
        content = '```json\n{"facts": [], "session_summary": "x"}\n```'
        result = _parse_llm_json(content)
        assert result["session_summary"] == "x"

    def test_json_with_fence_only(self):
        content = '```\n{"facts": [], "session_summary": "x"}\n```'
        result = _parse_llm_json(content)
        assert result["session_summary"] == "x"

    def test_json_with_preceding_text(self):
        content = 'Aquí está el resultado:\n\n{"facts": [], "session_summary": "ok"}'
        result = _parse_llm_json(content)
        assert result["session_summary"] == "ok"

    def test_json_with_trailing_text(self):
        content = '{"facts": [], "session_summary": "ok"}\nEspero que sirva.'
        result = _parse_llm_json(content)
        assert result["session_summary"] == "ok"

    def test_json_nested_objects(self):
        content = """{
            "facts": [
                {
                    "action": "create",
                    "namespace": "shared",
                    "key": "nombre",
                    "value": "Víctor",
                    "category": "personal",
                    "confidence": 1.0,
                    "ttl": null,
                    "related_fact_id": null,
                    "reason": "explicit"
                }
            ],
            "session_summary": "intro"
        }"""
        result = _parse_llm_json(content)
        assert len(result["facts"]) == 1
        assert result["facts"][0]["key"] == "nombre"
        assert result["facts"][0]["confidence"] == 1.0
        assert result["facts"][0]["ttl"] is None

    def test_finds_json_with_embedded_braces(self):
        content = '{"facts": [{"action": "create", "key": "llaves_{}", "value": "test"}], "session_summary": "ok"}'
        result = _parse_llm_json(content)
        assert result["facts"][0]["key"] == "llaves_{}"

    def test_invalid_json(self):
        with pytest.raises(ValueError):
            _parse_llm_json("not json at all")


# ── Prompt ───────────────────────────────────────────────────────


class TestConsolidationPrompt:
    def test_prompt_contains_all_placeholders(self):
        assert "{agent_name}" in CONSOLIDATION_PROMPT
        assert "{date}" in CONSOLIDATION_PROMPT
        assert "{existing_facts}" in CONSOLIDATION_PROMPT
        assert "{conversation}" in CONSOLIDATION_PROMPT
        assert "{state_default_ttl}" in CONSOLIDATION_PROMPT

    def test_prompt_format_valid(self):
        formatted = CONSOLIDATION_PROMPT.format(
            agent_name="lucy",
            date="2026-06-30",
            existing_facts="[id=1] nombre: Víctor",
            conversation="[user]: Hola",
            state_default_ttl=86400,
        )
        assert "lucy" in formatted
        assert "2026-06-30" in formatted
        assert "[id=1]" in formatted
        assert "[user]: Hola" in formatted

    def test_prompt_includes_action_types(self):
        assert "create" in CONSOLIDATION_PROMPT
        assert "update" in CONSOLIDATION_PROMPT
        assert "merge" in CONSOLIDATION_PROMPT
        assert "contradict" in CONSOLIDATION_PROMPT
        assert "discard" in CONSOLIDATION_PROMPT

    def test_prompt_includes_categories(self):
        assert "personal" in CONSOLIDATION_PROMPT
        assert "preference" in CONSOLIDATION_PROMPT
        assert "possessions" in CONSOLIDATION_PROMPT
        assert "personality" in CONSOLIDATION_PROMPT
        assert "state" in CONSOLIDATION_PROMPT
        assert "pato rosa" in CONSOLIDATION_PROMPT

    def test_prompt_solo_user_messages_for_personality(self):
        assert 'SOLO mensajes role="user"' in CONSOLIDATION_PROMPT
        assert "role=\"user\"" in CONSOLIDATION_PROMPT

    def test_prompt_spanish_response_required(self):
        assert "SIEMPRE responde en español" in CONSOLIDATION_PROMPT

    def test_prompt_session_summary_temático(self):
        assert "TEMÁTICO" in CONSOLIDATION_PROMPT




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
