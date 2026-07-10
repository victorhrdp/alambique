"""Tests for recall engine — formatting and prompt generation."""

from alambique.recall import (
    _format_fact_list,
    _format_session_list,
    _format_facts_for_personality,
    RECALL_PROMPT,
    PERSONALITY_PROMPT,
)
from alambique.models import Fact, FactCategory


# ── Fact formatting ──────────────────────────────────────────────


class TestFormatFactList:
    def test_empty(self):
        assert _format_fact_list([]) == "(sin hechos relevantes)"

    def test_single_fact(self):
        facts = [{"category": "personal", "key": "nombre", "value": "Víctor"}]
        result = _format_fact_list(facts)
        assert "- [personal] nombre: Víctor" in result

    def test_multiple_facts(self):
        facts = [
            {"category": "personal", "key": "a", "value": "v1"},
            {"category": "possessions", "key": "b", "value": "v2"},
        ]
        result = _format_fact_list(facts)
        lines = result.split("\n")
        assert len(lines) == 2

    def test_missing_fields(self):
        facts = [{}]
        result = _format_fact_list(facts)
        assert "[?]" in result


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


# ── Personality formatting ───────────────────────────────────────


class TestFormatFactsForPersonality:
    def test_empty(self):
        assert _format_facts_for_personality([]) == ""

    def test_single_trait(self):
        facts = [
            Fact(key="sarcastic",
                value="Es sarcástica en sus respuestas",
                category=FactCategory.PERSONALITY,
                confidence=0.9,
            )
        ]
        result = _format_facts_for_personality(facts)
        assert "[0.9] sarcastic:" in result
        assert "Es sarcástica" in result

    def test_multiple_traits_sorted_by_confidence(self):
        facts = [
            Fact(key="t1", value="v1", category=FactCategory.PERSONALITY, confidence=0.5),
            Fact(key="t2", value="v2", category=FactCategory.PERSONALITY, confidence=1.0),
            Fact(key="t3", value="v3", category=FactCategory.PERSONALITY, confidence=0.8),
        ]
        result = _format_facts_for_personality(facts)
        lines = result.split("\n")
        assert len(lines) == 3
        # Should be in input order (no explicit sort here, up to caller)
        assert "[0.5]" in lines[0]
        assert "[1.0]" in lines[1]

    def test_state_formatting(self):
        facts = [
            Fact(key="hoy_depre",
                value="Hoy está deprimido, sin bromas",
                category=FactCategory.STATE,
                ttl=86400,
                confidence=1.0,
            )
        ]
        result = _format_facts_for_personality(facts)
        assert "hoy_depre" in result
        assert "deprimido" in result


# ── Recall Prompt ────────────────────────────────────────────────


class TestRecallPrompt:
    def test_contains_placeholders(self):
        assert "{agent_name}" in RECALL_PROMPT
        assert "{query}" in RECALL_PROMPT
        assert "{top_facts}" in RECALL_PROMPT
        assert "{top_sessions}" in RECALL_PROMPT

    def test_format_valid(self):
        formatted = RECALL_PROMPT.format(
            agent_name="lucy",
            query="juegos cooperativos",
            top_facts="- [personal] nombre: Víctor",
            top_sessions="- [sess_1] Charla sobre gaming",
        )
        assert "lucy" in formatted
        assert "juegos cooperativos" in formatted
        assert "nombre: Víctor" in formatted
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
