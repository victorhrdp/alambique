"""Tests for ToolHandler — all 8 MCP tools with mocked external deps."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alambique.database import Database
from alambique.memory_maintenance import parse_embedding_blob
from alambique.models import (
    ConsolidationResponse,
    MemoryContextOutput,
    MemoryRecallOutput,
    MemoryStatusOutput,
    Message,
    SessionEndOutput,
    SessionStartOutput,
    SessionStatus,
)
from alambique.tools import (
    ToolHandler,
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

    def test_session_start_binds_antigravity_conversation(self, tools, monkeypatch, tmp_path):
        import json
        import shutil

        test_conv_id = "test-antigravity-bind-001"
        workspace = "/tmp/alambique-antigravity-bind"
        agy_home = tmp_path / "antigravity-cli"
        brain = agy_home / "brain" / test_conv_id
        logs = brain / ".system_generated" / "logs"
        logs.mkdir(parents=True)
        (logs / "transcript_full.jsonl").write_text(
            json.dumps(
                {
                    "type": "PLANNER_RESPONSE",
                    "source": "MODEL",
                    "content": "hola",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        history = agy_home / "history.jsonl"
        history.write_text(
            json.dumps(
                {
                    "display": "ping",
                    "timestamp": 99,
                    "workspace": workspace,
                    "conversationId": test_conv_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "alambique.transcripts.antigravity_cli.ANTIGRAVITY_HOME",
            agy_home,
        )

        try:
            result = asyncio.run(
                tools.session_start(client="antigravity_cli", workspace=workspace)
            )
            stored = tools.db.get_session(result.session_id)
            assert stored.client == "antigravity_cli"
            assert stored.conversation_id == test_conv_id
            assert result.conversation_id == test_conv_id
        finally:
            if brain.parent.exists():
                shutil.rmtree(brain.parent)

    def test_session_start_binds_opencode_conversation(self, tools, monkeypatch, tmp_path):
        import json
        import sqlite3

        test_conv_id = "ses_opencode_bind_001"
        workspace = "/tmp/alambique-opencode-bind"
        data_dir = tmp_path / "opencode"
        data_dir.mkdir(parents=True)
        db_path = data_dir / "opencode.db"

        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                directory TEXT NOT NULL,
                time_updated INTEGER NOT NULL
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO session (id, directory, time_updated) VALUES (?, ?, ?)",
            (test_conv_id, workspace, 1),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "alambique.transcripts.opencode_cli.OPENCODE_DATA",
            data_dir,
        )

        result = asyncio.run(
            tools.session_start(client="opencode", workspace=workspace)
        )
        stored = tools.db.get_session(result.session_id)
        assert stored.client == "opencode"
        assert stored.conversation_id == test_conv_id
        assert result.conversation_id == test_conv_id

    def test_session_end_with_opencode_transcript_sync(self, tools, monkeypatch, tmp_path):
        import json
        import sqlite3

        test_conv_id = "ses_opencode_sync_001"
        workspace = "/tmp/alambique-opencode-sync"
        data_dir = tmp_path / "opencode"
        data_dir.mkdir(parents=True)
        db_path = data_dir / "opencode.db"

        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE session (
                id TEXT PRIMARY KEY,
                directory TEXT NOT NULL,
                time_updated INTEGER NOT NULL
            );
            CREATE TABLE message (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            CREATE TABLE part (
                id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                time_created INTEGER NOT NULL,
                data TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO session (id, directory, time_updated) VALUES (?, ?, ?)",
            (test_conv_id, workspace, 1),
        )
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
            ("msg_u", test_conv_id, 1, json.dumps({"role": "user"})),
        )
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
            (
                "msg_a",
                test_conv_id,
                2,
                json.dumps({"role": "assistant", "finish": "stop"}),
            ),
        )
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, data) VALUES (?, ?, ?, ?, ?)",
            (
                "part_u",
                "msg_u",
                test_conv_id,
                1,
                json.dumps({"type": "text", "text": "Hola Lucy."}),
            ),
        )
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, data) VALUES (?, ?, ?, ?, ?)",
            (
                "part_a",
                "msg_a",
                test_conv_id,
                2,
                json.dumps({"type": "text", "text": "¡Hola, Víctor!"}),
            ),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "alambique.transcripts.opencode_cli.OPENCODE_DATA",
            data_dir,
        )

        result = asyncio.run(
            tools.session_start(client="opencode", workspace=workspace)
        )
        asyncio.run(tools.session_end(result.session_id))

        msgs = tools.db.get_session_messages(result.session_id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[0].content == "Hola Lucy."
        assert msgs[1].content == "¡Hola, Víctor!"

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

    def test_closed_session_triggers_auto_consolidation(self, tools):
        """session_end queues work and fire-and-forget consolidate (no persistent loop).

        Offline / empty transcript still marks consolidated=1 without leaving a
        sticky pending row — that is intentional after the queue-loop redesign.
        """

        async def flow():
            r = await tools.session_start()
            out = await tools.session_end(r.session_id)
            assert out.queued is True
            # Same event loop as create_task: let auto-consolidate finish.
            for _ in range(50):
                s = tools.db.get_session(r.session_id)
                if s and s.consolidated:
                    break
                await asyncio.sleep(0.01)
            return r.session_id

        sid = asyncio.run(flow())
        session = tools.db.get_session(sid)
        assert session is not None
        assert session.status == SessionStatus.CLOSED
        assert session.consolidated is True
        assert tools.db.get_pending_consolidations() == []

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

        history_path = Path.home() / ".gemini" / "antigravity-cli" / "history.jsonl"
        history_backup = history_path.read_text(encoding="utf-8") if history_path.exists() else None
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                {
                    "display": "sync test",
                    "timestamp": 1,
                    "workspace": "/tmp/alambique-antigravity-sync",
                    "conversationId": test_conv_id,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        try:
            r = asyncio.run(
                tools.session_start(
                    client="antigravity_cli",
                    workspace="/tmp/alambique-antigravity-sync",
                )
            )
            asyncio.run(tools.session_end(r.session_id))

            msgs = tools.db.get_session_messages(r.session_id)
            assert len(msgs) == 2
            assert msgs[0].role == "user"
            assert msgs[0].content == "Hola Lucy."
            assert msgs[1].content == "¡Hola, Víctor! ¿Cómo estás?"
        finally:
            if history_backup is None:
                history_path.unlink(missing_ok=True)
            else:
                history_path.write_text(history_backup, encoding="utf-8")
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


# ── session_update ───────────────────────────────────────────────


class TestSessionUpdate:
    def test_updates_expression_and_mood(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(
            tools.session_update(r.session_id, "happy", "contenta de verte")
        )
        assert out == {"status": "ok"}

        row = tools.db.get_latest_open_session_row()
        assert row["expression"] == "happy"
        assert row["mood_text"] == "contenta de verte"

    def test_rejects_unknown_session(self, tools):
        with pytest.raises(ValueError, match="Sesión no encontrada"):
            asyncio.run(
                tools.session_update("sess_deadbeef", "normal", "ok")
            )

    def test_rejects_invalid_expression(self, tools):
        r = asyncio.run(tools.session_start())
        with pytest.raises(ValueError, match="Expresión inválida"):
            asyncio.run(
                tools.session_update(r.session_id, "furious", "enfadada")
            )




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


# (legacy memory_forget and memory_export tests removed)


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

class TestApiKeyRuntime:
    def test_note_api_key_attempt_logs_failure_reason(self, tools, caplog):
        from alambique.consolidator import ApiKeyFetchResult

        with caplog.at_level("WARNING"):
            loaded = tools.note_api_key_attempt(
                ApiKeyFetchResult(error="pass no respondió en 120s — pinentry")
            )

        assert loaded is False
        assert tools._api_key_attempt_count == 1
        assert "pass no respondió" in tools._api_key_last_error
        assert any("intento 1" in record.message for record in caplog.records)

    def test_api_key_runtime_state_waiting_pass(self, tools):
        from alambique.consolidator import ApiKeyFetchResult

        tools.note_api_key_attempt(
            ApiKeyFetchResult(error="pass no respondió en 120s — pinentry")
        )
        state = tools._api_key_runtime_state()
        assert state.status == "waiting_pass"
        assert "120s" in (state.detail or "")

    def test_waiting_pass_sends_desktop_notification_once(self, tools, monkeypatch):
        from alambique.consolidator import ApiKeyFetchResult

        calls: list[tuple[str, str]] = []

        def fake_notify(title, body, **kwargs):
            calls.append((title, body))
            return True

        monkeypatch.setattr("alambique.tools.base.send_desktop_notification", fake_notify)

        err = ApiKeyFetchResult(error="pass no respondió en 120s — pinentry")
        tools.note_api_key_attempt(err)
        tools.note_api_key_attempt(err)
        assert len(calls) == 1
        assert "contraseña GPG" in calls[0][0]


class TestDaemonStatus:
    def test_daemon_status_uses_widget_checks(self, tools_online, mock_ollama):
        result = asyncio.run(tools_online.daemon_status(port=9042))
        assert result.online is True
        assert result.checks["daemon"].status == "ok"
        assert "9042" in (result.checks["daemon"].detail or "")
        assert "llm" in result.checks
        assert result.api_key.status == "loaded"

    def test_daemon_status_offline_reports_reason(self, tools, mock_ollama):
        from alambique.consolidator import ApiKeyFetchResult

        tools.note_api_key_attempt(
            ApiKeyFetchResult(error="pass no respondió en 120s — pinentry")
        )
        result = asyncio.run(tools.daemon_status())
        assert result.mode == "offline"
        assert result.overall == "degraded"
        assert result.api_key.status == "waiting_pass"
        assert "pinentry" in (result.checks["llm"].detail or "").lower()

    def test_daemon_status_system_message_ok(self, tools_online, mock_ollama):
        result = asyncio.run(tools_online.daemon_status(port=9042))
        assert result.system_message_level == "ok"
        assert "perfecto" in result.system_message.lower()

    def test_daemon_status_system_message_llm_recovery(self, tools_online, mock_ollama):
        tools_online._note_llm_outcome(False, "500 Internal Server Error")
        tools_online._note_llm_outcome(True, retried=True)
        result = asyncio.run(tools_online.daemon_status(port=9042))
        assert result.system_message_level == "warning"
        assert "inestabilidad" in result.system_message.lower()
        assert "ahora" in result.system_message.lower()

    def test_daemon_status_humanizes_consolidation_warning(self, tools_online, mock_ollama):
        tools_online._consolidation_warnings.append(
            "consolidation_filtered_prefix:architecture_"
        )
        result = asyncio.run(tools_online.daemon_status(port=9042))
        assert result.system_message_level == "info"
        assert result.overall == "ok"
        assert result.status_summary == "Operativo"
        assert "Todo funciona" in result.system_message
        assert "arquitectura del código" in result.system_message
        assert "consolidation_filtered" not in result.system_message


class TestMemoryStatus:
    def test_status_empty(self, tools):
        result = asyncio.run(tools.memory_status())
        assert result.sessions == 0
        assert result.threads == 0
        assert result.pending_consolidation == 0
        assert result.last_consolidation is None

    def test_status_with_data(self, tools):
        async def flow():
            r1 = await tools.session_start()
            await tools.session_start()
            await tools.session_end(r1.session_id)
            for _ in range(50):
                s = tools.db.get_session(r1.session_id)
                if s and s.consolidated:
                    break
                await asyncio.sleep(0.01)
            return await tools.memory_status()

        result = asyncio.run(flow())
        assert result.sessions == 2
        # Auto-consolidation after session_end clears the pending queue.
        assert result.pending_consolidation == 0


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
    def test_background_tasks_api_key_only(self, tools):
        """No consolidation loop: only the API-key retry task runs in background."""
        asyncio.run(tools.start_background_tasks())
        assert tools._api_key_retry_task is not None
        assert not hasattr(tools, "_consolidator_task") or tools.__dict__.get("_consolidator_task") is None
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

    def test_consolidation_applies_new_model(self, tools, mock_ollama):
        # Test that _consolidation_db_phase handles threads, capsules, echoes with new fields
        from alambique.models import ConsolidationResponse
        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        # Mock response with new fields
        response = ConsolidationResponse(
            threads=[{
                "action": "create",
                "key": "test_thread",
                "title": "Test Thread",
                "current_state": "This is a longer current state for the thread to pass validation checks.",
                "tone_guidance": "Tone here",
                "open_questions": ["Q1?", "¿Cómo se ve el UTF-8?"],
                "search_text": "search text",
                "salience": 0.9,
                "description": "Desc here",
                "reason": "test reason"
            }],
            relationship_capsules=[{
                "scope": "general",
                "content": "Capsule content",
                "reason": "test cap"
            }],
            echoes=[{
                "thread_key": "test_thread",
                "content": "Echo content",
                "context": "ctx",
                "salience": 0.8,
                "emotional_valence": 0.5,
                "reason": "test echo"
            }]
        )

        embed_requests = tools._consolidation_db_phase(bound, response)
        assert len(embed_requests) == 3  # thread, cap, echo

        # Check DB
        t = tools.db.get_thread_by_key("test_thread")
        assert t is not None
        assert t["description"] == "Desc here"
        assert "Q1?" in str(t["open_questions"])
        # Real UTF-8 in storage, not ensure_ascii \\uXXXX escapes
        assert "¿Cómo se ve el UTF-8?" in t["open_questions"]
        assert "\\u00" not in t["open_questions"]

        caps = tools.db.conn.execute("SELECT * FROM relationship_capsules WHERE scope = 'general'").fetchone()
        assert caps is not None
        assert "Capsule content" in caps["content"]

        ech = tools.db.conn.execute("SELECT * FROM echoes").fetchone()
        assert ech is not None
        assert ech["emotional_valence"] == 0.5

        # Check audit
        audits = tools.db.conn.execute("SELECT * FROM consolidations WHERE session_id = ?", (session.id,)).fetchall()
        assert len(audits) >= 3

    def test_rewrite_open_questions_utf8(self, tools, mock_ollama):
        """Legacy rows stored with ensure_ascii=True get rewritten to real UTF-8."""
        from alambique.tools.consolidation import rewrite_open_questions_utf8

        escaped = json.dumps(["¿Cómo estás?", "¡Víctor!"])  # ensure_ascii=True default
        assert "\\u00" in escaped
        tools.db.create_thread(
            key="unicode_legacy",
            title="Legacy",
            current_state="Estado largo suficiente para pasar validaciones mínimas del hilo.",
            tone_guidance="tono",
            open_questions=escaped,
        )
        n = rewrite_open_questions_utf8(tools.db.conn)
        tools.db.conn.commit()
        assert n >= 1
        row = tools.db.get_thread_by_key("unicode_legacy")
        assert "¿Cómo estás?" in row["open_questions"]
        assert "¡Víctor!" in row["open_questions"]
        assert "\\u00" not in row["open_questions"]

    def test_consolidation_skips_invalid_thread(self, tools, mock_ollama):
        from alambique.models import ConsolidationResponse
        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        response = ConsolidationResponse(
            threads=[{
                "action": "create",
                "key": "bad_thread",
                "title": "Bad",
                "current_state": "short",  # too short, should skip
                "tone_guidance": "tone",
                "search_text": "search",
                "salience": 0.5,
                "reason": "bad"
            }]
        )

        embed_requests = tools._consolidation_db_phase(bound, response)
        # should have only cap? no, response has no cap/echo, so 0
        assert len(embed_requests) == 0
        t = tools.db.get_thread_by_key('bad_thread')
        assert t is None  # skipped

    def test_activation_includes_new_fields(self, tools):
        # Test that activation surfaces description and open_questions
        from alambique.activation import ActivationEngine
        engine = ActivationEngine(tools.db, tools.ollama)

        # Seed a thread with new fields
        tools.db.conn.execute("""
            INSERT INTO threads (key, title, current_state, tone_guidance, description, open_questions, salience, status, last_active_at)
            VALUES ('philo_test', 'Philo Test', 'Current state long enough here for validation.', 'Tone guidance.', 'This is the description of the thread.', '["Q1 open?", "Q2?"]', 0.9, 'active', datetime('now'))
        """)
        tools.db.conn.commit()

        result = asyncio.run(engine.activate(None))
        context = result.get('initial_context', '')
        assert 'philo_test' in context
        assert 'This is the description of the thread.' in context
        assert 'Q1 open?' in context or 'open_questions' in context.lower()

    def test_merge_during_consolidation(self, tools, mock_ollama):
        from alambique.models import ConsolidationResponse
        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        # Seed an existing thread to merge into
        tools.db.conn.execute("""
            INSERT INTO threads (key, title, current_state, tone_guidance, salience, status)
            VALUES ('existing_key', 'Existing', 'Old state.', 'Old tone.', 0.5, 'active')
        """)
        tools.db.conn.commit()

        response = ConsolidationResponse(
            threads=[{
                "action": "merge",
                "key": "existing_key",
                "title": "Merged Title",
                "current_state": "This is a long enough merged current state now.",
                "tone_guidance": "Merged tone.",
                "search_text": "merged search",
                "salience": 0.95,
                "description": "Merged desc",
                "open_questions": ["Merged Q?"],
                "reason": "merged because same topic",
                "merged_from": ["old_other_key"]  # even if not exist, test handling
            }]
        )

        tools._consolidation_db_phase(bound, response)

        t = tools.db.get_thread_by_key('existing_key')
        assert t is not None
        assert t['current_state'] == 'This is a long enough merged current state now.'
        assert 'Merged desc' in (t.get('description') or '')

    def test_consolidation_create_on_existing_key_updates(self, tools, mock_ollama):
        """LLM action=create on an existing key must update, not UNIQUE-fail."""
        from alambique.models import ConsolidationResponse

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        tools.db.create_thread(
            key="already_there",
            title="Old title",
            current_state="Old state that is already long enough here.",
            tone_guidance="Old tone",
            search_text="old",
            salience=0.4,
        )

        response = ConsolidationResponse(
            threads=[{
                "action": "create",
                "key": "already_there",
                "title": "New title",
                "current_state": "Updated state after create-on-existing path, long enough.",
                "tone_guidance": "New tone",
                "search_text": "new search",
                "salience": 0.8,
                "reason": "should update not insert",
            }],
            relationship_capsules=[],
            echoes=[],
        )

        tools._consolidation_db_phase(bound, response)

        t = tools.db.get_thread_by_key("already_there")
        assert t is not None
        assert t["title"] == "New title"
        assert "Updated state" in t["current_state"]
        rows = tools.db.conn.execute(
            "SELECT COUNT(*) AS c FROM threads WHERE key = ?", ("already_there",)
        ).fetchone()
        assert rows["c"] == 1

    def test_consolidation_participation_idempotent(self, tools, mock_ollama):
        """Re-applying the same thread for a session must not UNIQUE-fail participations."""
        from alambique.models import ConsolidationResponse

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        item = {
            "action": "create",
            "key": "idem_thread",
            "title": "Idem",
            "current_state": "First contribution state, long enough for validation rules.",
            "tone_guidance": "tone",
            "search_text": "idem",
            "salience": 0.7,
            "reason": "first pass",
        }
        response = ConsolidationResponse(threads=[item], relationship_capsules=[], echoes=[])
        tools._consolidation_db_phase(bound, response)

        item2 = dict(item)
        item2["action"] = "update"
        item2["current_state"] = "Second contribution state, still long enough for validation."
        item2["reason"] = "second pass"
        response2 = ConsolidationResponse(threads=[item2], relationship_capsules=[], echoes=[])
        tools._consolidation_db_phase(bound, response2)

        t = tools.db.get_thread_by_key("idem_thread")
        parts = tools.db.conn.execute(
            "SELECT contribution_summary FROM thread_participations WHERE thread_id = ? AND session_id = ?",
            (t["id"], session.id),
        ).fetchall()
        assert len(parts) == 1
        assert "second pass" in (parts[0]["contribution_summary"] or "")

    def test_consolidation_creates_lucy_initiative(self, tools, mock_ollama):
        """Apply path persists lucy_initiative and supersedes previous pending."""
        from alambique.models import ConsolidationResponse

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        tools.db.create_initiative(
            "Iniciativa vieja que debe quedar superseded al crear una nueva."
        )

        response = ConsolidationResponse(
            threads=[],
            relationship_capsules=[],
            echoes=[],
            lucy_initiative={
                "prompt_payload": (
                    "Cuando el flujo lo permita, pregunta a Víctor si prefiere "
                    "validar el MVP de iniciativas con 10 sesiones reales."
                ),
                "thread_key": "alambique_autonomy_design",
                "reason": "Inquietud propia de Lucy, no un open_question",
            },
        )
        tools._consolidation_db_phase(bound, response)

        pending = tools.db.get_pending_initiative()
        assert pending is not None
        assert "MVP de iniciativas" in pending["prompt_payload"]
        assert pending["thread_key"] == "alambique_autonomy_design"
        assert pending["source_session_id"] == session.id
        statuses = [
            r["status"]
            for r in tools.db.conn.execute(
                "SELECT status FROM initiatives ORDER BY id"
            ).fetchall()
        ]
        assert statuses == ["superseded", "pending"]

    def test_consolidation_skips_short_initiative(self, tools, mock_ollama):
        from alambique.models import ConsolidationResponse

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)
        bound = tools.db.get_session(session.id)

        response = ConsolidationResponse(
            threads=[],
            relationship_capsules=[],
            echoes=[],
            lucy_initiative={"prompt_payload": "corto", "reason": "too short"},
        )
        tools._consolidation_db_phase(bound, response)
        assert tools.db.get_pending_initiative() is None


class TestInitiativeActivation:
    def test_session_start_injects_initiative(self, tools, mock_ollama, tmp_path, monkeypatch):
        # Keep widget state out of the real ~/.local/share/alambique
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", lambda: fake_home)

        tools.db.create_initiative(
            "Pregúntale a Víctor si quiere revisar el diff del MVP de iniciativas."
        )
        result = asyncio.run(tools.session_start())
        assert result.status == "ok"
        assert result.initial_context is not None
        assert "INICIATIVA DE LUCY PARA HOY" in result.initial_context
        assert "MVP de iniciativas" in result.initial_context
        # One injection consumed a TTL slot
        pending = tools.db.get_pending_initiative()
        assert pending is not None
        assert pending["sessions_seen"] == 1
        mem = fake_home / ".local" / "share" / "alambique" / "active_memory.json"
        assert mem.exists()
        data = json.loads(mem.read_text(encoding="utf-8"))
        assert data.get("pending_initiative")
        assert "MVP" in data["pending_initiative"]["prompt_payload"]