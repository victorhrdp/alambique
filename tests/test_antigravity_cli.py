"""Unit tests for Antigravity transcript provider."""

import json
from pathlib import Path

from alambique.transcripts.antigravity_cli import (
    _parse_transcript,
    resolve_antigravity_conversation_id,
)


class TestParseTranscript:
    def test_keeps_assistant_content_with_tool_calls(self, tmp_path):
        path = tmp_path / "transcript_full.jsonl"
        records = [
            {
                "type": "PLANNER_RESPONSE",
                "source": "MODEL",
                "content": "Arranco Alambique.",
                "tool_calls": [{"name": "view_file"}],
            },
            {
                "type": "PLANNER_RESPONSE",
                "source": "MODEL",
                "content": "Listo.",
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        messages = _parse_transcript(path)
        assert len(messages) == 2
        assert messages[0] == {"role": "assistant", "content": "Arranco Alambique."}
        assert messages[1] == {"role": "assistant", "content": "Listo."}

    def test_extracts_user_request(self, tmp_path):
        path = tmp_path / "transcript_full.jsonl"
        records = [
            {
                "type": "USER_INPUT",
                "source": "USER_EXPLICIT",
                "content": "<USER_REQUEST>\nHola Lucy.\n</USER_REQUEST>",
            }
        ]
        path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")

        messages = _parse_transcript(path)
        assert messages == [{"role": "user", "content": "Hola Lucy."}]


class TestResolveAntigravityConversationId:
    def test_resolves_from_history_by_workspace(self, monkeypatch, tmp_path):
        agy_home = tmp_path / "antigravity-cli"
        brain = agy_home / "brain" / "conv-recent"
        brain.mkdir(parents=True)
        (brain / ".system_generated" / "logs").mkdir(parents=True)
        (brain / ".system_generated" / "logs" / "transcript_full.jsonl").write_text(
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
                    "display": "new",
                    "timestamp": 2,
                    "workspace": "/w",
                    "conversationId": "conv-recent",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "alambique.transcripts.antigravity_cli.ANTIGRAVITY_HOME",
            agy_home,
        )

        resolved, warnings = resolve_antigravity_conversation_id(workspace="/w")
        assert resolved == "conv-recent"
        assert warnings == []

    def test_multiple_sessions_picks_most_recent(self, monkeypatch, tmp_path):
        agy_home = tmp_path / "antigravity-cli"
        for cid in ("conv-a", "conv-b"):
            brain = agy_home / "brain" / cid
            brain.mkdir(parents=True)

        history = agy_home / "history.jsonl"
        history.write_text(
            "\n".join(
                json.dumps(rec)
                for rec in [
                    {
                        "display": "a",
                        "timestamp": 10,
                        "workspace": "/w",
                        "conversationId": "conv-a",
                    },
                    {
                        "display": "b",
                        "timestamp": 20,
                        "workspace": "/w",
                        "conversationId": "conv-b",
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "alambique.transcripts.antigravity_cli.ANTIGRAVITY_HOME",
            agy_home,
        )

        resolved, warnings = resolve_antigravity_conversation_id(workspace="/w")
        assert resolved == "conv-b"
        assert "antigravity_multiple_active_sessions" in warnings