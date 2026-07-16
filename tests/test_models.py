"""Tests for Pydantic models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from alambique.models import (
    Consolidation,
    ConsolidationAction,
    ConsolidationResponse,
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
        assert o.related_sessions == []
        assert o.related_threads == []
        assert o.related_capsules == []

    def test_memory_recall_with_data(self):
        o = MemoryRecallOutput(
            summary="Hilos encontrados",
            related_sessions=[{"id": "sess_x", "snippet": "..."}],
            related_threads=[{"key": "philo", "snippet": "..."}],
            related_capsules=[{"scope": "general", "snippet": "..."}],
        )
        assert len(o.related_sessions) == 1
        assert len(o.related_threads) == 1
        assert len(o.related_capsules) == 1


class TestConsolidationModels:
    def test_consolidation_response_empty(self):
        r = ConsolidationResponse()
        # facts removed from consolidator (legacy); only threads/capsules/echoes now
        assert r.threads == []


class TestDatetimeUTC:
    """Verify all models use timezone-aware datetimes."""

    def test_message_datetime_aware(self):
        m = Message(session_id="s", role="user", content="hi")
        assert m.timestamp.tzinfo is not None

    def test_session_datetime_aware(self):
        s = Session(id="s")
        assert s.created_at.tzinfo is not None

