"""Unit tests for Grok transcript provider."""

import json
from pathlib import Path

import pytest

from alambique.transcripts.grok_cli import (
    _extract_user_text,
    _parse_chat_history,
    resolve_grok_session_id,
)


class TestExtractUserText:
    def test_user_query(self):
        text = "<user_query>\nHola Lucy.\n</user_query>"
        assert _extract_user_text(text) == "Hola Lucy."

    def test_user_info_dropped(self):
        text = "<user_info>meta</user_info>"
        assert _extract_user_text(text) is None

    def test_plain_text(self):
        assert _extract_user_text("  ping  ") == "ping"


class TestParseChatHistory:
    def test_keeps_assistant_content_with_tool_calls(self, tmp_path):
        path = tmp_path / "chat_history.jsonl"
        records = [
            {
                "type": "assistant",
                "content": "Arranco Alambique.",
                "tool_calls": [{"id": "1", "name": "Read", "arguments": "{}"}],
            },
            {"type": "tool_result", "tool_call_id": "1", "content": "ok"},
            {"type": "assistant", "content": "Listo."},
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        messages = _parse_chat_history(path)
        assert len(messages) == 2
        assert messages[0] == {"role": "assistant", "content": "Arranco Alambique."}
        assert messages[1] == {"role": "assistant", "content": "Listo."}

    def test_skips_tool_only_assistant_turn(self, tmp_path):
        path = tmp_path / "chat_history.jsonl"
        records = [
            {
                "type": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "Read", "arguments": "{}"}],
            }
        ]
        path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

        assert _parse_chat_history(path) == []


class TestResolveGrokSessionId:
    def test_multiple_active_sessions_warning(self, monkeypatch, tmp_path):
        grok_home = tmp_path / ".grok"
        sessions_root = grok_home / "sessions"
        sessions_root.mkdir(parents=True)

        for idx, sid in enumerate(("sess-a", "sess-b")):
            group = sessions_root / f"encoded-{idx}"
            group.mkdir(parents=True)
            session_dir = group / sid
            session_dir.mkdir()
            (session_dir / "chat_history.jsonl").write_text(
                json.dumps({"type": "assistant", "content": "hola"}) + "\n",
                encoding="utf-8",
            )

        active = grok_home / "active_sessions.json"
        active.write_text(
            json.dumps(
                [
                    {"session_id": "sess-a", "cwd": "/w", "pid": 1},
                    {"session_id": "sess-b", "cwd": "/w", "pid": 2},
                ]
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "alambique.transcripts.grok_cli.GROK_HOME",
            grok_home,
        )

        resolved, warnings = resolve_grok_session_id(workspace="/w")
        assert resolved is None
        assert "grok_multiple_active_sessions" in warnings