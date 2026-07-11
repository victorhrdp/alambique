import json
import logging
import os
from pathlib import Path

from .base import BaseTranscriptProvider
from .grok_cli import normalize_workspace

logger = logging.getLogger("alambique.transcripts.antigravity_cli")

ANTIGRAVITY_HOME = Path(
    os.environ.get("ANTIGRAVITY_HOME", Path.home() / ".gemini" / "antigravity-cli")
)


def _brain_dir(conversation_id: str) -> Path:
    return ANTIGRAVITY_HOME / "brain" / conversation_id


def _transcript_path(conversation_id: str) -> Path | None:
    log_dir = _brain_dir(conversation_id) / ".system_generated" / "logs"
    full_path = log_dir / "transcript_full.jsonl"
    normal_path = log_dir / "transcript.jsonl"
    if full_path.is_file():
        return full_path
    if normal_path.is_file():
        return normal_path
    return None


def _entry_workspace_matches(entry: dict, workspace: str) -> bool:
    entry_ws = entry.get("workspace")
    if not entry_ws:
        return False
    normalized_entry = normalize_workspace(entry_ws)
    normalized_workspace = normalize_workspace(workspace)
    if not normalized_entry or not normalized_workspace:
        return False
    return normalized_entry == normalized_workspace


def _load_history_entries(workspace: str | None) -> list[dict]:
    history_path = ANTIGRAVITY_HOME / "history.jsonl"
    if not history_path.is_file():
        return []

    try:
        lines = history_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    entries: list[dict] = []
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") == "slash_command":
            continue
        conversation_id = record.get("conversationId")
        if not conversation_id:
            continue
        if workspace and not _entry_workspace_matches(record, workspace):
            continue
        entries.append(record)
    return entries


def _pick_history_entry(entries: list[dict], warnings: list[str]) -> dict | None:
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    by_id: dict[str, dict] = {}
    for entry in entries:
        cid = entry.get("conversationId")
        if not cid:
            continue
        current = by_id.get(cid)
        if current is None or entry.get("timestamp", 0) > current.get("timestamp", 0):
            by_id[cid] = entry

    if len(by_id) == 1:
        return next(iter(by_id.values()))

    warnings.append("antigravity_multiple_active_sessions")
    return max(by_id.values(), key=lambda e: e.get("timestamp", 0))


def resolve_antigravity_conversation_id(
    conversation_id: str | None = None,
    workspace: str | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve Antigravity conversation UUID for transcript binding."""
    warnings: list[str] = []
    workspace = normalize_workspace(workspace)

    if conversation_id:
        if not _brain_dir(conversation_id).is_dir():
            warnings.append("antigravity_conversation_not_found")
            return None, warnings
        if not _transcript_path(conversation_id):
            warnings.append("antigravity_transcript_pending")
        return conversation_id, warnings

    env_id = os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
    if env_id:
        if _brain_dir(env_id).is_dir():
            if not _transcript_path(env_id):
                warnings.append("antigravity_transcript_pending")
            return env_id, warnings
        warnings.append("antigravity_conversation_id_env_stale")

    history_path = ANTIGRAVITY_HOME / "history.jsonl"
    if not history_path.is_file():
        warnings.append("antigravity_history_missing")
        return None, warnings

    entries = _load_history_entries(workspace)
    entry = _pick_history_entry(entries, warnings)
    if not entry:
        warnings.append("antigravity_no_active_session")
        return None, warnings

    resolved = entry.get("conversationId")
    if not resolved:
        warnings.append("antigravity_no_active_session")
        return None, warnings

    if not _brain_dir(resolved).is_dir():
        warnings.append("antigravity_conversation_not_found")
        return None, warnings

    if not _transcript_path(resolved):
        warnings.append("antigravity_transcript_pending")

    return resolved, warnings


def _extract_user_text(content: str) -> str | None:
    if "<USER_REQUEST>" in content:
        start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
        end = content.find("</USER_REQUEST>")
        if end > start:
            return content[start:end].strip()

    stripped = content.strip()
    return stripped or None


def _parse_transcript(path: Path) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue

            step_type = step.get("type")
            source = step.get("source")
            content = step.get("content", "")

            if step_type == "USER_INPUT" or source == "USER_EXPLICIT":
                if not isinstance(content, str):
                    continue
                clean_content = _extract_user_text(content)
                if clean_content:
                    messages.append({"role": "user", "content": clean_content})

            elif step_type == "PLANNER_RESPONSE" or source == "MODEL":
                if isinstance(content, str) and content.strip():
                    messages.append({"role": "assistant", "content": content})

    return messages


class AntigravityCliProvider(BaseTranscriptProvider):
    def can_handle(self, conversation_id: str | None = None, client: str | None = None) -> bool:
        if client and client != "antigravity_cli":
            return False

        conv_id = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        if not conv_id:
            return False

        return _transcript_path(conv_id) is not None

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        conv_id = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        if not conv_id:
            logger.warning(
                "AntigravityCliProvider: No conversation ID provided or found in environment."
            )
            return []

        path = _transcript_path(conv_id)
        if not path:
            logger.warning(
                "AntigravityCliProvider: Transcript file not found for session %s",
                conv_id,
            )
            return []

        return _parse_transcript(path)