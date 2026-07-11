"""Tests for ToolHandler — all 8 MCP tools with mocked external deps."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alambique.database import Database
from alambique.memory_maintenance import parse_embedding_blob
from alambique.models import (
    ConsolidationAction,
    ConsolidationFactItem,
    ConsolidationResponse,
    Fact,
    FactCategory,
    Message,
    SessionStartOutput,
    SessionEndOutput,
    SessionStatus,
    MemoryContextOutput,
)
from alambique.tools import (
    ToolHandler,
    consolidation_search_text,
    messages_for_consolidation,
)


def append_msg(tools, session_id, role, content):
    tools.db.append_message(Message(session_id=session_id, role=role, content=content))


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test_tools.db")
    d.connect()
    yield d
    d.close()


@pytest.fixture
def mock_ollama():
    o = MagicMock()
    o.embed = AsyncMock(return_value=[0.1] * 1024)
    o.health = AsyncMock(return_value=True)
    o.ensure_model = AsyncMock(return_value=True)
    o.close = AsyncMock()
    return o


@pytest.fixture
def tools(db, mock_ollama):
    return ToolHandler(db, mock_ollama, api_key=None)


@pytest.fixture
def tools_online(db, mock_ollama):
    """ToolHandler with API key (online mode)."""
    # We patch recall and consolidator at the ToolHandler property level
    return ToolHandler(db, mock_ollama, api_key="test-key")


# ── session_start ─────────────────────────────────────────────────


class TestSessionStart:
    def test_starts_session_without_args(self, tools):
        result = asyncio.run(tools.session_start())
        assert isinstance(result, SessionStartOutput)
        assert result.status == "ok"
        assert result.session_id is not None
        assert result.session_id.startswith("sess_")

    def test_new_database(self, tools):
        result = asyncio.run(tools.session_start())
        assert result.status == "ok"
        assert result.is_new is True
        assert result.persona is None
        assert result.session_id.startswith("sess_")

    def test_existing_agent_offline_no_persona(self, tools):
        asyncio.run(tools.session_start())
        result = asyncio.run(tools.session_start())
        assert result.status == "ok"
        assert result.is_new is False
        assert result.persona is None  # offline = no persona composition

    def test_existing_agent_new_session_id_each_time(self, tools):
        r1 = asyncio.run(tools.session_start())
        r2 = asyncio.run(tools.session_start())
        assert r1.session_id != r2.session_id

    def test_session_stored_in_db(self, tools):
        r = asyncio.run(tools.session_start())
        s = tools.db.get_session(r.session_id)
        assert s is not None
        assert s.status == SessionStatus.OPEN

    def test_online_with_facts_composes_persona(self, tools_online, mock_ollama):
        db = tools_online.db
        f = Fact(
            key="sarcastic",
            value="Es sarcástica",
            category=FactCategory.PERSONALITY,
            confidence=0.9,
        )
        db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value="Eres Lucy. Sarcástica y genial.")
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.persona == "Eres Lucy. Sarcástica y genial."
        assert result.is_new is True

    def test_persona_seed_on_new_agent(self, tools_online):
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(
            return_value="Eres Lucy, la AI más inteligente."
        )
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(
            tools_online.session_start(persona_seed="Eres Lucy, la AI más inteligente.",
            )
        )
        assert result.is_new is True
        assert result.persona == "Eres Lucy, la AI más inteligente."
        traits = tools_online.db.get_facts(
            categories=(FactCategory.PERSONALITY,)
        )
        assert len(traits) == 1
        assert traits[0].key == "persona_seed"

    def test_persona_seed_ignored_when_traits_exist(self, tools_online):
        tools_online.db.insert_fact(
            Fact(
                key="existing",
                value="Personalidad previa",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value="Personalidad previa compuesta.")
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        asyncio.run(
            tools_online.session_start(persona_seed="Semilla que no debe guardarse")
        )
        traits = tools_online.db.get_facts(
            categories=(FactCategory.PERSONALITY,)
        )
        assert len(traits) == 1
        assert traits[0].key == "existing"

    def test_offline_uses_trait_without_llm(self, tools):
        tools.db.insert_fact(
            Fact(key="persona_seed",
                value="Eres Lucy offline.",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        result = asyncio.run(tools.session_start())
        assert result.persona == "Eres Lucy offline."
        assert result.degraded is True
        assert "offline_mode" in result.warnings
        assert "persona_offline_fallback" in result.warnings

    def test_online_no_facts_returns_null_persona(self, tools_online):
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value=None)
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.is_new is True
        assert result.persona is None

    def test_persona_composition_handles_error(self, tools_online):
        db = tools_online.db
        asyncio.run(tools_online.session_start())
        f = Fact(key="trait",
            value="algo",
            category=FactCategory.PERSONALITY,
            confidence=0.9,
        )
        db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(side_effect=Exception("API down"))
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.persona == "algo"  # fallback al rasgo si falla el LLM


# ── session_end ──────────────────────────────────────────────────


class TestSessionStartBinding:
    def test_session_start_without_client_warns(self, tools, mock_ollama):
        tools._compose_session_persona = AsyncMock(return_value=("persona", []))
        result = asyncio.run(tools.session_start())
        assert "binding_missing_client" in result.warnings
        assert result.degraded is True
        assert result.conversation_id is None
        assert result.session_id is not None

    def test_session_start_grok_binding_failed_no_orphan(self, tools, mock_ollama):
        from pathlib import Path

        tools._compose_session_persona = AsyncMock(return_value=("persona", []))
        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text("[]", encoding="utf-8")

        try:
            before = len(tools.db.get_all_sessions())
            result = asyncio.run(
                tools.session_start(
                    client="grok",
                    workspace="/tmp/alambique-grok-missing-binding",
                )
            )
            assert result.status == "error"
            assert result.session_id is None
            assert "binding_failed" in result.warnings
            assert len(tools.db.get_all_sessions()) == before
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")

    def test_session_start_reuses_existing_binding(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-reuse-001"
        workspace = "/tmp/alambique-grok-reuse"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"
        session_dir.mkdir(parents=True, exist_ok=True)
        transcript_file.write_text(
            json.dumps({"type": "assistant", "content": "hola"}) + "\n",
            encoding="utf-8",
        )

        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                [{"session_id": test_conv_id, "pid": 1, "cwd": workspace, "opened_at": "now"}]
            ),
            encoding="utf-8",
        )

        try:
            tools._compose_session_persona = AsyncMock(return_value=("persona", []))
            first = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            second = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            assert first.session_id == second.session_id
            assert second.session_reused is True
            assert "session_reused" in second.warnings
            open_bound = tools.db.get_open_sessions_by_binding("grok", test_conv_id)
            assert len(open_bound) == 1
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_session_start_binds_grok_conversation(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-bind-001"
        workspace = "/tmp/alambique-grok-bind"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "content": [{"type": "text", "text": "<user_query>\nBind me\n</user_query>"}],
                    }
                )
                + "\n"
            )

        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                [{"session_id": test_conv_id, "pid": 1, "cwd": workspace, "opened_at": "now"}]
            ),
            encoding="utf-8",
        )

        try:
            result = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            stored = tools.db.get_session(result.session_id)
            assert stored.client == "grok"
            assert stored.conversation_id == test_conv_id
            assert result.conversation_id == test_conv_id
            assert result.client == "grok"
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_session_end_uses_stored_binding(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-bind-end-002"
        workspace = "/tmp/alambique-grok-bind-end"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        lines = [
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_query>\nHola\n</user_query>"}],
            },
            {"type": "assistant", "content": "Adiós"},
        ]
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                [{"session_id": test_conv_id, "pid": 1, "cwd": workspace, "opened_at": "now"}]
            ),
            encoding="utf-8",
        )

        try:
            started = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            asyncio.run(tools.session_end(started.session_id))

            msgs = tools.db.get_session_messages(started.session_id)
            assert len(msgs) == 2
            assert msgs[0].content == "Hola"
            assert msgs[1].content == "Adiós"
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)


class TestSessionEnd:
    def test_close_session(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.session_end(r.session_id))
        assert isinstance(out, SessionEndOutput)
        assert out.queued is True

        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.CLOSED
        assert s.ended_at is not None

    def test_truncated_session(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.session_end(r.session_id, truncated=True))
        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.TRUNCATED

    def test_closed_session_pending_consolidation(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.session_end(r.session_id))
        pending = tools.db.get_pending_consolidations()
        assert len(pending) == 1

    def test_session_end_with_transcript_sync(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json

        test_conv_id = "test-conv-sync-123"
        log_dir = Path.home() / ".gemini" / "antigravity-cli" / "brain" / test_conv_id / ".system_generated" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = log_dir / "transcript_full.jsonl"

        lines = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "<USER_REQUEST>\nHola Lucy.\n</USER_REQUEST>\n"},
            {"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "content": "Checking status", "tool_calls": [{"name": "some_tool"}]},
            {"step_index": 2, "source": "MODEL", "type": "PLANNER_RESPONSE", "content": "¡Hola, Víctor! ¿Cómo estás?", "tool_calls": None}
        ]
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", test_conv_id)

        try:
            r = asyncio.run(tools.session_start())
            # End session - should trigger transcript sync
            asyncio.run(tools.session_end(r.session_id))

            # Verify database messages are synchronized
            msgs = tools.db.get_session_messages(r.session_id)
            assert len(msgs) == 2
            assert msgs[0].role == "user"
            assert msgs[0].content == "Hola Lucy."
            assert msgs[1].role == "assistant"
            assert msgs[1].content == "¡Hola, Víctor! ¿Cómo estás?"
        finally:
            if log_dir.parent.parent.exists():
                shutil.rmtree(log_dir.parent.parent)

    def test_session_end_with_transcript_sync_client_mismatch(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json

        test_conv_id = "test-conv-sync-456"
        log_dir = Path.home() / ".gemini" / "antigravity-cli" / "brain" / test_conv_id / ".system_generated" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        transcript_file = log_dir / "transcript_full.jsonl"

        lines = [
            {"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": "hello"}
        ]
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        monkeypatch.setenv("ANTIGRAVITY_CONVERSATION_ID", test_conv_id)

        try:
            r = asyncio.run(tools.session_start())
            # End session with mismatched client
            asyncio.run(tools.session_end(r.session_id, client="other_cli"))

            # Verify database messages are NOT synchronized
            msgs = tools.db.get_session_messages(r.session_id)
            assert len(msgs) == 0
        finally:
            if log_dir.parent.parent.exists():
                shutil.rmtree(log_dir.parent.parent)

    def test_session_end_with_grok_transcript_sync(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-sync-789"
        cwd = "/tmp/alambique-grok-test"
        encoded_cwd = quote(cwd, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        lines = [
            {"type": "system", "content": "bootstrap"},
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_info>meta</user_info>"}],
            },
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_query>\nHola Lucy.\n</user_query>"}],
            },
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "call-1", "name": "Read", "arguments": "{}"}],
            },
            {"type": "tool_result", "tool_call_id": "call-1", "content": "ok"},
            {"type": "assistant", "content": "¡Hola, Víctor! ¿Cómo estás?"},
        ]
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        try:
            r = asyncio.run(tools.session_start())
            asyncio.run(
                tools.session_end(
                    r.session_id,
                    conversation_id=test_conv_id,
                    client="grok",
                )
            )

            msgs = tools.db.get_session_messages(r.session_id)
            assert len(msgs) == 2
            assert msgs[0].role == "user"
            assert msgs[0].content == "Hola Lucy."
            assert msgs[1].role == "assistant"
            assert msgs[1].content == "¡Hola, Víctor! ¿Cómo estás?"
        finally:
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_session_end_with_grok_transcript_sync_env_fallback(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-sync-env-001"
        cwd = "/tmp/alambique-grok-env-test"
        encoded_cwd = quote(cwd, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "user",
                        "content": [{"type": "text", "text": "<user_query>\nPing\n</user_query>"}],
                    }
                )
                + "\n"
            )
            f.write(json.dumps({"type": "assistant", "content": "Pong"}) + "\n")

        monkeypatch.setenv("GROK_SESSION_ID", test_conv_id)

        try:
            r = asyncio.run(tools.session_start())
            asyncio.run(tools.session_end(r.session_id, client="grok"))

            msgs = tools.db.get_session_messages(r.session_id)
            assert len(msgs) == 2
            assert msgs[0].content == "Ping"
            assert msgs[1].content == "Pong"
        finally:
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)


# ── memory_recall ────────────────────────────────────────────────


class TestMemoryRecall:
    def test_recall_with_vector_results(self, tools, mock_ollama):
        db = tools.db
        asyncio.run(tools.session_start())
        f = Fact(key="juegos",
            value="Víctor juega It Takes Two con Enrique",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid = db.insert_fact(f)
        db.record_fact_access(fid)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid, "rowid": fid, "distance": 0.1}],  # vec0_facts
                [],  # vec0_sessions
            ]
            result = asyncio.run(tools.memory_recall("juegos cooperativos"))

        assert len(result.facts) == 1
        assert result.facts[0]["key"] == "juegos"

    def test_recall_fallback_on_vector_error(self, tools, mock_ollama):
        mock_ollama.embed.side_effect = Exception("Ollama down")

        asyncio.run(tools.session_start())
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        tools.db.insert_fact(f)

        result = asyncio.run(tools.memory_recall("quién soy"))
        assert len(result.facts) > 0
        assert result.summary != ""
        assert result.degraded is True
        assert "vector_search_failed" in result.warnings
        assert "recall_degraded_recent_facts" in result.warnings

    def test_recall_no_results(self, tools, mock_ollama):
        with patch.object(tools, "_vector_search", return_value=[]):
            result = asyncio.run(tools.memory_recall("nada"))

        assert result.facts == []
        assert result.related_sessions == []

    def test_recall_no_candidates_warning(self, tools, mock_ollama):
        db = tools.db
        f_low = Fact(key="dubious",
            value="not sure",
            category=FactCategory.PERSONAL,
            confidence=0.5,
        )
        fid_low = db.insert_fact(f_low)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid_low, "rowid": fid_low, "distance": 0.1}],
                [],
            ]
            result = asyncio.run(tools.memory_recall("test"))

        assert result.facts == []
        assert "no_candidates_after_filter" in result.warnings
        assert result.degraded is True

    def test_recall_online_composes_summary(self, tools_online, mock_ollama):
        db = tools_online.db
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid = db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_summary = AsyncMock(return_value="Víctor juega con Enrique.")
        mock_recall.compose_personality = AsyncMock(return_value=None)
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        with patch.object(tools_online, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid, "rowid": fid, "distance": 0.1}],
                [],
            ]
            result = asyncio.run(tools_online.memory_recall("juegos"))

        assert result.summary == "Víctor juega con Enrique."

    def test_recall_filters_low_confidence_facts(self, tools, mock_ollama):
        db = tools.db
        f_low = Fact(key="dubious",
            value="not sure",
            category=FactCategory.PERSONAL,
            confidence=0.5,
        )
        fid_low = db.insert_fact(f_low)
        f_high = Fact(key="sure",
            value="confirmed",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid_high = db.insert_fact(f_high)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [
                    {"id": fid_low, "rowid": fid_low, "distance": 0.1},
                    {"id": fid_high, "rowid": fid_high, "distance": 0.2},
                ],
                [],
            ]
            result = asyncio.run(tools.memory_recall("test"))

        keys = {f["key"] for f in result.facts}
        assert "sure" in keys
        assert "dubious" not in keys

    def test_recall_hybrid_re_ranking(self, tools, mock_ollama):
        db = tools.db
        asyncio.run(tools.session_start())
        
        # 1. Fact A: Very close vector-wise (distance=0.01 -> similarity=0.99), low access (0), confidence=1.0
        # Score = 0.99 * 0.6 + 1.0 * 0.2 + 0.0 * 0.2 = 0.594 + 0.2 = 0.794
        f_a = Fact(key="fact_a",
            value="A very close match semantically",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=0,
        )
        fid_a = db.insert_fact(f_a)
        
        # 2. Fact B: Slightly further vector-wise (distance=0.4 -> similarity=0.714), high access (10 -> reinforcement=0.5), confidence=1.0
        # Score = 0.714 * 0.6 + 1.0 * 0.2 + 0.5 * 0.2 = 0.428 + 0.2 + 0.1 = 0.728
        f_b = Fact(key="fact_b",
            value="A slightly further match",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=10,
        )
        fid_b = db.insert_fact(f_b)

        # 3. Fact C: Furthest vector-wise (distance=0.6 -> similarity=0.625), extremely high access (20 -> reinforcement=1.0), confidence=1.0
        # Score = 0.625 * 0.6 + 1.0 * 0.2 + 1.0 * 0.2 = 0.375 + 0.2 + 0.2 = 0.775
        f_c = Fact(key="fact_c",
            value="The furthest match but very popular",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=20,
        )
        fid_c = db.insert_fact(f_c)

        # Update access counts directly in database since insert_fact ignores access_count
        db.conn.execute("UPDATE facts SET access_count = 10 WHERE id = ?", (fid_b,))
        db.conn.execute("UPDATE facts SET access_count = 20 WHERE id = ?", (fid_c,))
        db.conn.commit()

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [
                    {"id": fid_a, "rowid": fid_a, "distance": 0.01},
                    {"id": fid_b, "rowid": fid_b, "distance": 0.4},
                    {"id": fid_c, "rowid": fid_c, "distance": 0.6},
                ],
                [],
            ]
            result = asyncio.run(tools.memory_recall("search query"))

        # The expected ranking order of keys based on scores:
        # 1. fact_a (0.794)
        # 2. fact_c (0.775)
        # 3. fact_b (0.728)
        assert len(result.facts) == 3
        assert result.facts[0]["key"] == "fact_a"
        assert result.facts[1]["key"] == "fact_c"
        assert result.facts[2]["key"] == "fact_b"


# ── memory_search ────────────────────────────────────────────────


class TestMemorySearch:
    def test_search_finds_content(self, tools):
        r = asyncio.run(tools.session_start())
        append_msg(tools, r.session_id, "user", "palabra_clave")
        append_msg(tools, r.session_id, "assistant", "respuesta")

        result = asyncio.run(tools.memory_search("palabra_clave"))
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["content"] == "palabra_clave"

    def test_search_no_results(self, tools):
        result = asyncio.run(tools.memory_search("nonexistent_xyz"))
        assert "results" in result
        assert len(result["results"]) == 0




# ── memory_forget ────────────────────────────────────────────────


class TestMemoryContext:
    def test_context_empty_session(self, tools):
        r = asyncio.run(tools.session_start())
        result = asyncio.run(tools.memory_context(r.session_id))
        assert isinstance(result, MemoryContextOutput)
        assert result.total == 0
        assert result.messages == []
        assert result.offset == 0

    def test_context_with_messages(self, tools):
        r = asyncio.run(tools.session_start())
        append_msg(tools, r.session_id, "user", "msg1")
        append_msg(tools, r.session_id, "assistant", "msg2")
        append_msg(tools, r.session_id, "user", "msg3")

        result = asyncio.run(tools.memory_context(r.session_id))
        assert result.total == 3
        assert len(result.messages) == 3
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "msg1"

    def test_context_offset(self, tools):
        r = asyncio.run(tools.session_start())
        for i in range(5):
            append_msg(tools, r.session_id, "user", f"msg{i}")

        result = asyncio.run(tools.memory_context(r.session_id, offset=2, limit=2))
        assert result.total == 5
        assert result.offset == 2
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "msg2"

    def test_context_limit_capped_at_30(self, tools):
        r = asyncio.run(tools.session_start())
        result = asyncio.run(tools.memory_context(r.session_id, limit=100))
        assert result.limit == 30

    def test_context_includes_session_summary(self, tools):
        r = asyncio.run(tools.session_start())
        append_msg(tools, r.session_id, "user", "hi")
        tools.db.close_session(r.session_id)
        tools.db.set_session_summary(r.session_id, "Charla de prueba")

        result = asyncio.run(tools.memory_context(r.session_id))
        assert result.session_summary == "Charla de prueba"

    def test_context_nonexistent_session_raises(self, tools):
        with pytest.raises(ValueError, match="no encontrada"):
            asyncio.run(tools.memory_context("bad_id"))


# ── memory_forget ────────────────────────────────────────────────


class TestMemoryForget:
    def test_forget_by_fact_id(self, tools):
        f = Fact(key="temp",
            value="to forget",
            category=FactCategory.PERSONALITY,
        )
        fid = tools.db.insert_fact(f)
        result = asyncio.run(tools.memory_forget(fact_id=fid))
        assert result["deleted"] is True

        forgotten = tools.db.get_fact(fid)
        assert forgotten.confidence == 0

    def test_forget_by_key(self, tools):
        f = Fact(key="old_pref",
            value="vim",
            category=FactCategory.PREFERENCE,
        )
        tools.db.insert_fact(f)
        result = asyncio.run(tools.memory_forget(key="old_pref"))
        assert result["deleted"] is True

    def test_forget_nonexistent_raises(self, tools):
        with pytest.raises(ValueError):
            asyncio.run(tools.memory_forget(key="nope"))

    def test_forget_no_params_raises(self, tools):
        with pytest.raises(ValueError, match="Especifica"):
            asyncio.run(tools.memory_forget())


# ── memory_export ────────────────────────────────────────────────


class TestMemoryExport:
    def test_export_empty(self, tools):
        result = asyncio.run(tools.memory_export())
        assert result["facts"] == []
        assert result["sessions"] == []

    def test_export_with_data(self, tools):
        from alambique.tools import _insert_embedding

        r = asyncio.run(tools.session_start())
        append_msg(tools, r.session_id, "user", "hi")
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
        )
        fid = tools.db.insert_fact(f)
        _insert_embedding(tools.db.conn, "vec0_facts", fid, [0.1] * 1024)

        tools.db.close_session(r.session_id)
        tools.db.set_session_summary(r.session_id, "Saludo inicial")

        result = asyncio.run(tools.memory_export())
        assert len(result["facts"]) == 1
        assert len(result["sessions"]) == 1
        assert result["facts"][0]["embedding_ok"] is True
        assert result["facts"][0]["created_at"] is not None
        assert result["sessions"][0]["created_at"] is not None


# ── memory_status ────────────────────────────────────────────────


class TestMemoryHealth:
    def test_health_all_ok(self, tools_online, mock_ollama):
        result = asyncio.run(tools_online.memory_health())
        assert result.healthy is True
        assert result.mode == "online"
        assert result.checks["ollama"].status == "ok"
        assert result.checks["api_key"].status == "ok"
        assert result.warnings == []

    def test_health_ollama_down(self, tools_online, mock_ollama):
        mock_ollama.health = AsyncMock(return_value=False)
        result = asyncio.run(tools_online.memory_health())
        assert result.healthy is False
        assert "ollama_unavailable" in result.warnings

    def test_health_offline_mode(self, tools, mock_ollama):
        result = asyncio.run(tools.memory_health())
        assert result.mode == "offline"
        assert "offline_mode" in result.warnings

    def test_health_orphan_embeddings(self, tools, mock_ollama):
        tools.db.insert_fact(
            Fact(key="orphan", value="sin vector", category=FactCategory.PERSONAL)
        )
        result = asyncio.run(tools.memory_health())
        assert result.checks["embeddings"].status == "warning"
        assert "embeddings_orphaned" in result.warnings


class TestMemoryStatus:
    def test_status_empty(self, tools):
        result = asyncio.run(tools.memory_status())
        assert result.sessions == 0
        assert result.facts == 0
        assert result.pending_consolidation == 0
        assert result.last_consolidation is None

    def test_status_with_data(self, tools):
        r1 = asyncio.run(tools.session_start())
        r2 = asyncio.run(tools.session_start())
        asyncio.run(tools.session_end(r1.session_id))  # close to count as open => count session

        f = Fact(
            key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
        )
        tools.db.insert_fact(f)

        result = asyncio.run(tools.memory_status())
        assert result.sessions == 2
        assert result.facts == 1
        assert result.pending_consolidation == 1


# ── Vector helpers ───────────────────────────────────────────────


class TestVectorHelpers:
    def test_session_id_rowid_roundtrip(self):
        from alambique.tools import _session_id_to_rowid, _rowid_to_session_id

        test_ids = [
            "sess_000000000000",
            "sess_ffffffffffff",
            "sess_a1b2c3d4e5f6",
            "sess_123456789abc",
        ]
        for sid in test_ids:
            assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_insert_embedding_fact(self, db, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = db.conn
        _insert_embedding(conn, "vec0_facts", 1, [0.1] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = 1"
        ).fetchall()
        assert len(rows) == 1

    def test_insert_embedding_session(self, db, mock_ollama):
        from alambique.tools import _insert_embedding, _session_id_to_rowid

        conn = db.conn
        sid = "sess_abc123def456"
        rowid = _session_id_to_rowid(sid)
        _insert_embedding(conn, "vec0_sessions", sid, [0.1] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_sessions WHERE rowid = ?", (rowid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == rowid

    def test_update_embedding(self, db, mock_ollama):
        from alambique.tools import _insert_embedding, _update_embedding

        conn = db.conn
        _insert_embedding(conn, "vec0_facts", 42, [1.0] * 1024)
        _update_embedding(conn, "vec0_facts", 42, [0.5] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = 42"
        ).fetchall()
        assert len(rows) == 1

    def test_upsert_embedding_inserts_then_updates(self, db, mock_ollama):
        from alambique.tools import _upsert_embedding

        conn = db.conn
        _upsert_embedding(conn, "vec0_facts", 7, [0.1] * 1024)
        _upsert_embedding(conn, "vec0_facts", 7, [0.9] * 1024)

        row = conn.execute(
            "SELECT embedding FROM vec0_facts WHERE rowid = 7"
        ).fetchone()
        assert parse_embedding_blob(row["embedding"])[0] == pytest.approx(0.9)

    def test_vector_search_facts(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = tools.db.conn
        f1 = Fact(key="a", value="v", category=FactCategory.PERSONAL)
        f2 = Fact(key="b", value="v", category=FactCategory.PERSONALITY)
        fid1 = tools.db.insert_fact(f1)
        fid2 = tools.db.insert_fact(f2)

        # Create same embedding for both
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_facts", fid1, emb)
        _insert_embedding(conn, "vec0_facts", fid2, emb)

        results = tools._vector_search("vec0_facts", emb, limit=10)
        assert len(results) == 2

    def test_vector_search_active_facts_only_excludes_forgotten(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = tools.db.conn
        f = Fact(key="gone", value="v", category=FactCategory.PERSONAL)
        fid = tools.db.insert_fact(f)
        tools.db.forget_fact(fid)
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_facts", fid, emb)

        results = tools._vector_search(
            "vec0_facts",
            emb,
            limit=10,
            active_facts_only=True,
        )
        assert results == []

    def test_vector_search_sessions(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding, _session_id_to_rowid, _rowid_to_session_id

        conn = tools.db.conn
        r = asyncio.run(tools.session_start())
        sid = r.session_id
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_sessions", sid, emb)

        results = tools._vector_search("vec0_sessions", emb, limit=5)
        assert len(results) == 1
        assert results[0]["session_id"] == sid



# ── consolidation fact retrieval ───────────────────────────────────


class TestConsolidationFactRetrieval:
    def test_consolidation_search_text(self):
        msgs = [
            Message(session_id="s", role="user", content="Uso CachyOS"),
            Message(session_id="s", role="assistant", content="Interesante distro"),
        ]
        text = consolidation_search_text(msgs)
        assert "user: Uso CachyOS" in text
        assert "assistant: Interesante distro" in text

    def test_consolidation_search_text_truncates(self):
        msgs = [Message(session_id="s", role="user", content="x" * 9000)]
        text = consolidation_search_text(msgs, max_chars=100)
        assert len(text) == 100
        assert text.endswith("x")

    def test_facts_for_consolidation_prefers_semantic_match(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        target = tools.db.insert_fact(
            Fact(
                key="keyboard",
                value="Teclado mecánico Keychron",
                category=FactCategory.POSSESSIONS,
                confidence=1.0,
            )
        )
        noise = tools.db.insert_fact(
            Fact(
                key="favorite_color",
                value="Azul",
                category=FactCategory.PREFERENCE,
                confidence=1.0,
            )
        )
        emb = [0.2] * 1024
        _insert_embedding(tools.db.conn, "vec0_facts", target, emb)
        _insert_embedding(tools.db.conn, "vec0_facts", noise, [0.9] * 1024)

        mock_ollama.health = AsyncMock(return_value=True)
        mock_ollama.embed = AsyncMock(return_value=emb)

        msgs = [
            Message(session_id="s", role="user", content="Mi teclado mecánico Keychron es genial"),
        ]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert warnings == []
        assert any(f.id == target for f in facts)
        assert facts[0].id == target

    def test_facts_for_consolidation_includes_personality(self, tools, mock_ollama):
        trait_id = tools.db.insert_fact(
            Fact(
                key="sarcastic",
                value="Es sarcástica",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        mock_ollama.health = AsyncMock(return_value=False)

        msgs = [Message(session_id="s", role="user", content="Hola")]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert "consolidation_facts_offline_fallback" in warnings
        assert any(f.id == trait_id for f in facts)

    def test_facts_for_consolidation_vector_fallback(self, tools, mock_ollama):
        mock_ollama.health = AsyncMock(return_value=True)
        mock_ollama.embed = AsyncMock(side_effect=RuntimeError("embed down"))

        tools.db.insert_fact(
            Fact(key="fallback", value="dato", category=FactCategory.PERSONAL, confidence=1.0)
        )
        msgs = [Message(session_id="s", role="user", content="Algo")]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert "consolidation_facts_vector_fallback" in warnings
        assert any(f.key == "fallback" for f in facts)


# ── consolidation embeddings ───────────────────────────────────────


class TestConsolidationEmbeddings:
    def test_update_refreshes_existing_fact_embedding(self, tools, mock_ollama):
        fid = tools.db.insert_fact(
            Fact(key="os", value="Linux", category=FactCategory.PREFERENCE, confidence=1.0)
        )
        from alambique.tools import _insert_embedding

        _insert_embedding(tools.db.conn, "vec0_facts", fid, [0.1] * 1024)

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        new_emb = [0.9] * 1024
        mock_ollama.embed_batch = AsyncMock(return_value=[new_emb])
        mock_ollama.embed = AsyncMock(return_value=[0.2] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.UPDATE,
                    key="os",
                    value="CachyOS",
                    category=FactCategory.PREFERENCE,
                    confidence=1.0,
                    related_fact_id=fid,
                    reason="SO actualizado",
                )
            ],
            session_summary="Cambio de sistema operativo",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        updated = tools.db.get_fact(fid)
        assert updated.value == "CachyOS"
        stored = tools.db.get_fact_embedding(fid)
        assert stored[0] == pytest.approx(0.9)

    def test_create_with_existing_key_updates_in_place(self, tools, mock_ollama):
        fid = tools.db.insert_fact(
            Fact(key="city", value="Madrid", category=FactCategory.PERSONAL, confidence=1.0)
        )

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.4] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.1] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.CREATE,
                    key="city",
                    value="Barcelona",
                    category=FactCategory.PERSONAL,
                    confidence=1.0,
                    reason="Corrección de ciudad",
                )
            ],
            session_summary="Ciudad actualizada",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        active = tools.db.get_fact_by_key("city")
        assert active.id == fid
        assert active.value == "Barcelona"

    def test_contradict_allocates_alternate_key(self, tools, mock_ollama):
        existing_id = tools.db.insert_fact(
            Fact(key="mood", value="cansado", category=FactCategory.STATE, confidence=1.0, ttl=86400)
        )

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.6] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.1] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.CONTRADICT,
                    key="mood",
                    value="recuperado",
                    category=FactCategory.STATE,
                    confidence=1.0,
                    ttl=86400,
                    related_fact_id=existing_id,
                    reason="Estado resuelto",
                )
            ],
            session_summary="Ánimo mejorado",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        assert tools.db.get_fact_by_key("mood").id == existing_id
        alt = tools.db.get_fact_by_key("mood__alt")
        assert alt is not None
        assert alt.value == "recuperado"

    def test_batch_create_and_update_embeddings(self, tools, mock_ollama):
        existing_id = tools.db.insert_fact(
            Fact(key="gpu", value="RTX 3080", category=FactCategory.POSSESSIONS, confidence=1.0)
        )
        from alambique.tools import _insert_embedding

        _insert_embedding(tools.db.conn, "vec0_facts", existing_id, [0.1] * 1024)

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.5] * 1024, [0.8] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.3] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.UPDATE,
                    key="gpu",
                    value="RTX 4080",
                    category=FactCategory.POSSESSIONS,
                    confidence=1.0,
                    related_fact_id=existing_id,
                    reason="GPU actualizada",
                ),
                ConsolidationFactItem(
                    action=ConsolidationAction.CREATE,
                    key="keyboard",
                    value="Teclado mecánico",
                    category=FactCategory.POSSESSIONS,
                    confidence=1.0,
                    reason="Nueva posesión",
                ),
            ],
            session_summary="Hardware actualizado",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        assert tools.db.get_fact(existing_id).value == "RTX 4080"
        assert tools.db.get_fact_embedding(existing_id)[0] == pytest.approx(0.5)

        new_fact = tools.db.get_fact_by_key("keyboard")
        assert new_fact is not None
        assert tools.db.get_fact_embedding(new_fact.id)[0] == pytest.approx(0.8)


# ── Background tasks / lifecycle ─────────────────────────────────


class TestDbGuard:
    def test_db_guard_serializes_concurrent_access(self, tools):
        order: list[str] = []

        async def job(label: str) -> None:
            async with tools._db_guard():
                order.append(f"start-{label}")
                await asyncio.sleep(0.02)
                order.append(f"end-{label}")

        async def run() -> None:
            await asyncio.gather(job("a"), job("b"))

        asyncio.run(run())

        assert order.index("end-a") < order.index("start-b") or order.index("end-b") < order.index("start-a")




class TestBackgroundTasks:
    def test_consolidation_loop_idle(self, tools):
        asyncio.run(tools.start_background_tasks())
        assert tools._consolidator_task is not None
        assert tools._watchdog_task is not None
        asyncio.run(tools.stop_background_tasks())

    def test_watchdog_detects_stale(self, tools, mock_ollama):
        """Mark a session as stale (>30 min old, still open)."""
        asyncio.run(tools.session_start())
        stale = tools.db.find_stale_sessions(timeout_minutes=-1)
        assert len(stale) >= 1

    def test_watchdog_syncs_bound_grok_transcript(self, tools, mock_ollama):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-watchdog-001"
        workspace = "/tmp/alambique-grok-watchdog"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        lines = [
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_query>\nSin end manual\n</user_query>"}],
            },
            {"type": "assistant", "content": "Consolidación automática"},
        ]
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                [{"session_id": test_conv_id, "pid": 1, "cwd": workspace, "opened_at": "now"}]
            ),
            encoding="utf-8",
        )

        try:
            started = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            tools.db.conn.execute(
                "UPDATE sessions SET created_at = datetime('now', '-31 minutes') WHERE id = ?",
                (started.session_id,),
            )
            tools.db.conn.commit()

            stale = tools.db.find_stale_sessions(timeout_minutes=30)
            assert any(s.id == started.session_id for s in stale)

            bound = tools.db.get_session(started.session_id)
            asyncio.run(
                tools._close_session(
                    bound.id,
                    SessionStatus.TRUNCATED,
                    conversation_id=bound.conversation_id,
                    client=bound.client,
                )
            )

            closed = tools.db.get_session(started.session_id)
            msgs = tools.db.get_session_messages(started.session_id)
            assert closed.status == SessionStatus.TRUNCATED
            assert len(msgs) == 2
            assert msgs[0].content == "Sin end manual"
            assert msgs[1].content == "Consolidación automática"
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_watchdog_without_binding_leaves_session_empty(self, tools, mock_ollama):
        started = asyncio.run(tools.session_start())
        tools.db.conn.execute(
            "UPDATE sessions SET created_at = datetime('now', '-31 minutes') WHERE id = ?",
            (started.session_id,),
        )
        tools.db.conn.commit()

        asyncio.run(
            tools._close_session(started.session_id, SessionStatus.TRUNCATED)
        )

        msgs = tools.db.get_session_messages(started.session_id)
        assert msgs == []

    def test_shutdown_open_sessions_syncs_and_closes(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-shutdown-001"
        workspace = "/tmp/alambique-grok-shutdown"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        lines = [
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_query>\nShutdown sync\n</user_query>"}],
            },
            {"type": "assistant", "content": "Cerrando limpio"},
        ]
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        active_path = Path.home() / ".grok" / "active_sessions.json"
        backup = active_path.read_text(encoding="utf-8") if active_path.exists() else None
        active_path.parent.mkdir(parents=True, exist_ok=True)
        active_path.write_text(
            json.dumps(
                [{"session_id": test_conv_id, "pid": 1, "cwd": workspace, "opened_at": "now"}]
            ),
            encoding="utf-8",
        )

        try:
            started = asyncio.run(
                tools.session_start(client="grok", workspace=workspace)
            )
            asyncio.run(tools.shutdown_open_sessions())

            closed = tools.db.get_session(started.session_id)
            msgs = tools.db.get_session_messages(started.session_id)
            assert closed.status == SessionStatus.TRUNCATED
            assert len(msgs) == 2
            assert msgs[0].content == "Shutdown sync"
            assert msgs[1].content == "Cerrando limpio"
        finally:
            if backup is None:
                active_path.unlink(missing_ok=True)
            else:
                active_path.write_text(backup, encoding="utf-8")
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_consolidation_resyncs_empty_bound_session(self, tools, monkeypatch):
        from pathlib import Path
        import shutil
        import json
        from urllib.parse import quote

        test_conv_id = "test-grok-consolidate-resync-001"
        workspace = "/tmp/alambique-grok-consolidate-resync"
        encoded_cwd = quote(workspace, safe="")
        session_dir = Path.home() / ".grok" / "sessions" / encoded_cwd / test_conv_id
        transcript_file = session_dir / "chat_history.jsonl"

        lines = [
            {
                "type": "user",
                "content": [{"type": "text", "text": "<user_query>\nRe-sync\n</user_query>"}],
            },
            {"type": "assistant", "content": "Importado tarde"},
        ]
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line) + "\n")

        session = tools.db.create_session(client="grok", conversation_id=test_conv_id)
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        try:
            bound = tools.db.get_session(session.id)
            asyncio.run(tools._consolidate_session(bound))

            msgs = tools.db.get_session_messages(session.id)
            assert len(msgs) == 2
            assert msgs[0].content == "Re-sync"
            assert msgs[1].content == "Importado tarde"
        finally:
            group_dir = session_dir.parent
            if group_dir.exists():
                shutil.rmtree(group_dir)

    def test_online_property_false_without_key(self, tools):
        assert tools.online is False

    def test_online_property_true_with_key(self, tools_online):
        assert tools_online.online is True

    def test_consolidator_raises_without_key(self, tools):
        with pytest.raises(RuntimeError, match="API key"):
            _ = tools.consolidator

    def test_recall_raises_without_key(self, tools):
        with pytest.raises(RuntimeError, match="API key"):
            _ = tools.recall


# ── messages_for_consolidation ────────────────────────────────────


class TestMessagesForConsolidation:
    def test_keeps_user_and_assistant(self):
        msgs = [
            Message(session_id="s1", role="user", content="Hola"),
            Message(session_id="s1", role="assistant", content="Qué tal"),
        ]
        assert len(messages_for_consolidation(msgs)) == 2

    def test_drops_auto_commentary(self):
        msgs = [
            Message(session_id="s1", role="assistant", content="[Auto] Un goblin corre"),
            Message(session_id="s1", role="user", content="[Tú] ¿qué ves?"),
            Message(session_id="s1", role="assistant", content="[Aria] Una cueva"),
        ]
        filtered = messages_for_consolidation(msgs)
        assert len(filtered) == 2
        assert filtered[0].content.startswith("[Tú]")
        assert filtered[1].content.startswith("[Aria]")

    def test_drops_system_and_tool_roles(self):
        msgs = [
            Message(session_id="s1", role="system", content="ignore"),
            Message(session_id="s1", role="user", content="ok"),
        ]
        assert len(messages_for_consolidation(msgs)) == 1


# ── Observer pattern: session closure enforces correct state ─────


class TestSessionLifecycle:
    def test_full_lifecycle(self, tools):
        r = asyncio.run(tools.session_start())
        assert tools.db.get_session(r.session_id).status == SessionStatus.OPEN

        append_msg(tools, r.session_id, "user", "Hola")
        append_msg(tools, r.session_id, "assistant", "Respuesta")
        append_msg(tools, r.session_id, "user", "Gracias")

        msgs = tools.db.get_session_messages(r.session_id)
        assert len(msgs) == 3

        asyncio.run(tools.session_end(r.session_id))
        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.CLOSED
        assert s.ended_at is not None
