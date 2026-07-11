"""Unit tests for OpenCode transcript provider."""

import json
import sqlite3
from pathlib import Path

from alambique.transcripts.opencode_cli import (
    _parse_session_messages,
    resolve_opencode_session_id,
)


def _write_test_db(path: Path, sessions: list[dict], messages: list[dict], parts: list[dict]) -> None:
    conn = sqlite3.connect(path)
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
    for session in sessions:
        conn.execute(
            "INSERT INTO session (id, directory, time_updated) VALUES (?, ?, ?)",
            (session["id"], session["directory"], session["time_updated"]),
        )
    for message in messages:
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, data) VALUES (?, ?, ?, ?)",
            (
                message["id"],
                message["session_id"],
                message["time_created"],
                json.dumps(message["data"]),
            ),
        )
    for part in parts:
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, time_created, data) VALUES (?, ?, ?, ?, ?)",
            (
                part["id"],
                part["message_id"],
                part["session_id"],
                part["time_created"],
                json.dumps(part["data"]),
            ),
        )
    conn.commit()
    conn.close()


class TestParseSessionMessages:
    def test_extracts_user_and_final_assistant_text(self, tmp_path, monkeypatch):
        db_path = tmp_path / "opencode.db"
        session_id = "ses_test_001"
        _write_test_db(
            db_path,
            sessions=[
                {"id": session_id, "directory": "/w", "time_updated": 1},
            ],
            messages=[
                {
                    "id": "msg_user",
                    "session_id": session_id,
                    "time_created": 1,
                    "data": {"role": "user"},
                },
                {
                    "id": "msg_tooling",
                    "session_id": session_id,
                    "time_created": 2,
                    "data": {"role": "assistant", "finish": "tool-calls"},
                },
                {
                    "id": "msg_reply",
                    "session_id": session_id,
                    "time_created": 3,
                    "data": {"role": "assistant", "finish": "stop"},
                },
            ],
            parts=[
                {
                    "id": "part_u",
                    "message_id": "msg_user",
                    "session_id": session_id,
                    "time_created": 1,
                    "data": {"type": "text", "text": "Hola Lucy."},
                },
                {
                    "id": "part_plan",
                    "message_id": "msg_tooling",
                    "session_id": session_id,
                    "time_created": 2,
                    "data": {"type": "text", "text": "Voy a consultar memoria."},
                },
                {
                    "id": "part_tool",
                    "message_id": "msg_tooling",
                    "session_id": session_id,
                    "time_created": 3,
                    "data": {"type": "tool", "tool": "memory_recall"},
                },
                {
                    "id": "part_a",
                    "message_id": "msg_reply",
                    "session_id": session_id,
                    "time_created": 4,
                    "data": {"type": "text", "text": "¡Hola, Víctor!"},
                },
            ],
        )

        monkeypatch.setattr(
            "alambique.transcripts.opencode_cli.OPENCODE_DATA",
            tmp_path,
        )

        messages = _parse_session_messages(session_id)
        assert messages == [
            {"role": "user", "content": "Hola Lucy."},
            {"role": "assistant", "content": "¡Hola, Víctor!"},
        ]


class TestResolveOpenCodeSessionId:
    def test_resolves_from_workspace(self, tmp_path, monkeypatch):
        db_path = tmp_path / "opencode.db"
        _write_test_db(
            db_path,
            sessions=[
                {"id": "ses_old", "directory": "/w", "time_updated": 10},
                {"id": "ses_new", "directory": "/w", "time_updated": 20},
            ],
            messages=[],
            parts=[],
        )

        monkeypatch.setattr(
            "alambique.transcripts.opencode_cli.OPENCODE_DATA",
            tmp_path,
        )

        resolved, warnings = resolve_opencode_session_id(workspace="/w")
        assert resolved == "ses_new"
        assert "opencode_multiple_active_sessions" in warnings

    def test_resolves_explicit_conversation_id(self, tmp_path, monkeypatch):
        db_path = tmp_path / "opencode.db"
        _write_test_db(
            db_path,
            sessions=[{"id": "ses_explicit", "directory": "/w", "time_updated": 1}],
            messages=[],
            parts=[],
        )

        monkeypatch.setattr(
            "alambique.transcripts.opencode_cli.OPENCODE_DATA",
            tmp_path,
        )

        resolved, warnings = resolve_opencode_session_id(conversation_id="ses_explicit")
        assert resolved == "ses_explicit"
        assert warnings == []