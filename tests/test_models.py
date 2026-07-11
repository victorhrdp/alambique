"""Tests for Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from alambique.models import (
    Consolidation,
    ConsolidationAction,
    ConsolidationFactItem,
    ConsolidationResponse,
    Fact,
    FactCategory,
    MemoryRecallOutput,
    Message,
    Session,
    SessionEndOutput,
    SessionStartOutput,
    SessionStatus,
)


class TestEnums:
    def test_session_status_values(self):
        assert SessionStatus.OPEN == "open"
        assert SessionStatus.CLOSED == "closed"
        assert SessionStatus.TRUNCATED == "truncated"

    def test_fact_category_values(self):
        assert FactCategory.PERSONALITY == "personality"
        assert FactCategory.STATE == "state"
        assert FactCategory.PERSONAL == "personal"
        assert FactCategory.PREFERENCE == "preference"
        assert FactCategory.POSSESSIONS == "possessions"

    def test_consolidation_action_values(self):
        assert ConsolidationAction.CREATE == "create"
        assert ConsolidationAction.UPDATE == "update"
        assert ConsolidationAction.MERGE == "merge"
        assert ConsolidationAction.CONTRADICT == "contradict"
        assert ConsolidationAction.DISCARD == "discard"


class TestDomainModels:
    def test_session_defaults(self):
        s = Session(id="sess_abc")
        assert s.status == SessionStatus.OPEN
        assert s.consolidated is False
        assert s.summary is None
        assert s.ended_at is None
        assert s.created_at.tzinfo is not None

    def test_message_serializes_tool_calls(self):
        m = Message(
            session_id="sess_abc",
            role="assistant",
            content="c",
            tool_calls='[{"name": "echo"}]',
        )
        assert m.tool_calls == '[{"name": "echo"}]'

    def test_fact_defaults(self):
        f = Fact(
            key="sarcastic",
            value="Es sarcástica",
            category=FactCategory.PERSONALITY,
        )
        assert f.confidence == 1.0
        assert f.access_count == 0
        assert f.ttl is None
        assert f.created_at.tzinfo is not None

    def test_fact_category_validation(self):
        f = Fact(
            key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
        )
        assert f.category == FactCategory.PERSONAL

    def test_consolidation_creation(self):
        c = Consolidation(
            session_id="sess_abc",
            action=ConsolidationAction.CREATE,
            fact_id=1,
            new_value="test",
            reason="new fact",
        )
        assert c.action == ConsolidationAction.CREATE


class TestMCPOutputModels:
    def test_session_start_ok(self):
        o = SessionStartOutput(
            session_id="sess_abc",
            status="ok",
            persona="Eres Lucy...",
            is_new=False,
        )
        assert o.session_id == "sess_abc"
        assert o.persona == "Eres Lucy..."

    def test_session_start_new_agent(self):
        o = SessionStartOutput(
            session_id="sess_abc",
            status="ok",
            is_new=True,
        )
        assert o.is_new is True
        assert o.persona is None

    def test_session_end(self):
        o = SessionEndOutput(queued=True, pending_consolidation=2)
        assert o.queued is True
        assert o.pending_consolidation == 2

    def test_memory_recall_empty(self):
        o = MemoryRecallOutput(summary="No hay información")
        assert o.facts == []
        assert o.related_sessions == []

    def test_memory_recall_with_data(self):
        o = MemoryRecallOutput(
            summary="Hechos encontrados",
            facts=[{"id": 1, "key": "k", "value": "v", "category": "personal", "confidence": 1.0}],
            related_sessions=[{"id": "sess_x", "snippet": "..."}],
        )
        assert len(o.facts) == 1
        assert len(o.related_sessions) == 1


class TestConsolidationModels:
    def test_consolidation_response_empty(self):
        r = ConsolidationResponse(
            facts=[],
            session_summary="Nothing to report",
        )
        assert r.facts == []
        assert r.session_summary == "Nothing to report"

    def test_consolidation_fact_item_create(self):
        item = ConsolidationFactItem(
            action=ConsolidationAction.CREATE,
            key="fact_key",
            value="fact value",
            category=FactCategory.PERSONAL,
            confidence=1.0,
            reason="new",
        )
        assert item.related_fact_id is None

    def test_consolidation_fact_item_update(self):
        item = ConsolidationFactItem(
            action=ConsolidationAction.UPDATE,
            key="fact_key",
            value="new value",
            category=FactCategory.PREFERENCE,
            confidence=0.9,
            related_fact_id=42,
            reason="updated",
        )
        assert item.related_fact_id == 42


class TestDatetimeUTC:
    """Verify all models use timezone-aware datetimes."""

    def test_fact_datetime_aware(self):
        f = Fact(key="k", value="v", category=FactCategory.PERSONAL)
        assert f.created_at.tzinfo is not None
        assert f.last_accessed.tzinfo is not None

    def test_message_datetime_aware(self):
        m = Message(session_id="s", role="user", content="hi")
        assert m.timestamp.tzinfo is not None

    def test_session_datetime_aware(self):
        s = Session(id="s")
        assert s.created_at.tzinfo is not None


class TestFactForgettingCurve:
    def test_personal_category_does_not_decay(self):
        from datetime import datetime, timedelta, timezone
        # A fact created 100 days ago of category PERSONAL should keep its original confidence
        past = datetime.now(timezone.utc) - timedelta(days=100)
        f = Fact(
            key="born_date",
            value="10/04/1978",
            category=FactCategory.PERSONAL,
            confidence=1.0,
            created_at=past,
        )
        assert f.get_decayed_confidence() == 1.0

    def test_preference_decays_over_time(self):
        from datetime import datetime, timedelta, timezone
        # A preference fact created 30 days ago with no accesses should decay
        past = datetime.now(timezone.utc) - timedelta(days=30)
        f = Fact(
            key="game_style",
            value="DRPGs",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            created_at=past,
            access_count=0,
        )
        decayed = f.get_decayed_confidence()
        assert decayed < 1.0
        # The floor for preference is 0.5, so it shouldn't fall below that
        assert decayed >= 0.5

    def test_reinforcement_mitigates_decay(self):
        from datetime import datetime, timedelta, timezone
        # Two preference facts created 15 days ago: one with high access, one with 0 access
        past = datetime.now(timezone.utc) - timedelta(days=15)
        f_no_access = Fact(
            key="game_style_no",
            value="FPS",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            created_at=past,
            access_count=0,
        )
        f_high_access = Fact(
            key="game_style_high",
            value="DRPGs",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            created_at=past,
            access_count=100,
        )
        decayed_no = f_no_access.get_decayed_confidence()
        decayed_high = f_high_access.get_decayed_confidence()
        # The fact with high access count should have significantly higher confidence
        assert decayed_high > decayed_no
        assert decayed_high > 0.75

    def test_state_category_does_not_decay(self):
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(days=30)
        f = Fact(
            key="neck_pain",
            value="duele el cuello",
            category=FactCategory.STATE,
            confidence=1.0,
            ttl=86400,
            created_at=past,
        )
        assert f.get_decayed_confidence() == 1.0

    def test_possessions_decays_slowly(self):
        from datetime import datetime, timedelta, timezone

        past = datetime.now(timezone.utc) - timedelta(days=90)
        f = Fact(
            key="keyboard",
            value="YMDK ID75",
            category=FactCategory.POSSESSIONS,
            confidence=1.0,
            created_at=past,
            access_count=0,
        )
        decayed = f.get_decayed_confidence()
        assert decayed < 1.0
        assert decayed >= 0.8

    def test_is_ttl_expired_and_is_active(self):
        from datetime import datetime, timedelta, timezone

        old = datetime.now(timezone.utc) - timedelta(hours=25)
        f = Fact(
            key="tired",
            value="cansado",
            category=FactCategory.STATE,
            confidence=1.0,
            ttl=86400,
            created_at=old,
        )
        assert f.is_ttl_expired()
        assert not f.is_active()
