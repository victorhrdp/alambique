import json
import logging
import os
from pathlib import Path

from .base import BaseTranscriptProvider

logger = logging.getLogger("alambique.transcripts.grok_cli")

GROK_HOME = Path(os.environ.get("GROK_HOME", Path.home() / ".grok"))


def _resolve_conversation_id(conversation_id: str | None) -> str | None:
    return conversation_id or os.environ.get("GROK_SESSION_ID")


def resolve_grok_session_id(
    conversation_id: str | None = None,
    workspace: str | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve the Grok session UUID for transcript binding."""
    warnings: list[str] = []

    candidates: list[str] = []
    if conversation_id:
        candidates.append(conversation_id)
    env_id = os.environ.get("GROK_SESSION_ID")
    if env_id and env_id not in candidates:
        candidates.append(env_id)

    for candidate in candidates:
        if _find_chat_history(candidate):
            return candidate, warnings
    if conversation_id:
        warnings.append("grok_conversation_not_found")
        return None, warnings
    if env_id:
        warnings.append("grok_session_id_env_stale")

    active_path = GROK_HOME / "active_sessions.json"
    if not active_path.is_file():
        warnings.append("grok_active_sessions_missing")
        return None, warnings

    try:
        entries = json.loads(active_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        warnings.append("grok_active_sessions_invalid")
        return None, warnings

    if not isinstance(entries, list):
        warnings.append("grok_active_sessions_invalid")
        return None, warnings

    if workspace:
        entries = [e for e in entries if isinstance(e, dict) and e.get("cwd") == workspace]

    session_ids = [
        e.get("session_id")
        for e in entries
        if isinstance(e, dict) and e.get("session_id")
    ]

    if len(session_ids) == 1:
        sid = session_ids[0]
        if sid and _find_chat_history(sid):
            return sid, warnings
        warnings.append("grok_active_session_transcript_missing")
        return None, warnings

    if len(session_ids) > 1:
        warnings.append("grok_multiple_active_sessions")
        return None, warnings

    warnings.append("grok_no_active_session")
    return None, warnings


def _find_chat_history(conversation_id: str) -> Path | None:
    sessions_root = GROK_HOME / "sessions"
    if not sessions_root.is_dir():
        return None

    for group_dir in sessions_root.iterdir():
        if not group_dir.is_dir():
            continue
        candidate = group_dir / conversation_id / "chat_history.jsonl"
        if candidate.is_file():
            return candidate

    return None


def _extract_user_text(text: str) -> str | None:
    if "<user_query>" in text:
        start = text.find("<user_query>") + len("<user_query>")
        end = text.find("</user_query>")
        if end > start:
            return text[start:end].strip()

    if "<user_info>" in text:
        return None

    stripped = text.strip()
    return stripped or None


def _parse_chat_history(path: Path) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            record_type = record.get("type")
            if record_type == "user":
                content = record.get("content")
                if isinstance(content, list):
                    texts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    text = "\n".join(texts)
                elif isinstance(content, str):
                    text = content
                else:
                    continue

                clean = _extract_user_text(text)
                if clean:
                    messages.append({"role": "user", "content": clean})

            elif record_type == "assistant":
                content = record.get("content", "")
                if isinstance(content, str) and content.strip():
                    messages.append({"role": "assistant", "content": content})

    return messages


class GrokCliProvider(BaseTranscriptProvider):
    def can_handle(self, conversation_id: str | None = None, client: str | None = None) -> bool:
        if client and client != "grok":
            return False

        conv_id = _resolve_conversation_id(conversation_id)
        if not conv_id:
            return False

        return _find_chat_history(conv_id) is not None

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        conv_id = _resolve_conversation_id(conversation_id)
        if not conv_id:
            logger.warning("GrokCliProvider: No conversation ID provided or found in environment.")
            return []

        path = _find_chat_history(conv_id)
        if not path:
            logger.warning("GrokCliProvider: chat_history.jsonl not found for session %s", conv_id)
            return []

        return _parse_chat_history(path)