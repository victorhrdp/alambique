import json
import logging
import os
import sqlite3
from pathlib import Path

from .base import BaseTranscriptProvider
from .grok_cli import normalize_workspace

logger = logging.getLogger("alambique.transcripts.opencode_cli")

OPENCODE_DATA = Path(
    os.environ.get("OPENCODE_DATA", Path.home() / ".local" / "share" / "opencode")
)


def _db_path() -> Path:
    return OPENCODE_DATA / "opencode.db"


def _connect_readonly() -> sqlite3.Connection | None:
    path = _db_path()
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
    except sqlite3.Error:
        return None


def _session_exists(session_id: str) -> bool:
    conn = _connect_readonly()
    if not conn:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM session WHERE id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _directory_matches(entry_directory: str | None, workspace: str) -> bool:
    if not entry_directory:
        return False
    normalized_entry = normalize_workspace(entry_directory)
    normalized_workspace = normalize_workspace(workspace)
    if not normalized_entry or not normalized_workspace:
        return False
    return normalized_entry == normalized_workspace


def _sessions_for_workspace(workspace: str) -> list[sqlite3.Row]:
    conn = _connect_readonly()
    if not conn:
        return []
    try:
        rows = conn.execute(
            "SELECT id, directory, time_updated FROM session ORDER BY time_updated DESC"
        ).fetchall()
        return [row for row in rows if _directory_matches(row["directory"], workspace)]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _pick_session(rows: list[sqlite3.Row], warnings: list[str]) -> str | None:
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0]["id"]

    by_id: dict[str, sqlite3.Row] = {}
    for row in rows:
        sid = row["id"]
        current = by_id.get(sid)
        if current is None or row["time_updated"] > current["time_updated"]:
            by_id[sid] = row

    if len(by_id) == 1:
        return next(iter(by_id.keys()))

    warnings.append("opencode_multiple_active_sessions")
    return max(by_id.values(), key=lambda r: r["time_updated"])["id"]


def resolve_opencode_session_id(
    conversation_id: str | None = None,
    workspace: str | None = None,
) -> tuple[str | None, list[str]]:
    """Resolve OpenCode session id (ses_…) for transcript binding."""
    warnings: list[str] = []
    workspace = normalize_workspace(workspace)

    if conversation_id:
        if not _session_exists(conversation_id):
            warnings.append("opencode_conversation_not_found")
            return None, warnings
        return conversation_id, warnings

    env_id = os.environ.get("OPENCODE_SESSION_ID")
    if env_id:
        if _session_exists(env_id):
            return env_id, warnings
        warnings.append("opencode_session_id_env_stale")

    if not _db_path().is_file():
        warnings.append("opencode_db_missing")
        return None, warnings

    if not workspace:
        warnings.append("binding_missing_workspace")
        warnings.append("binding_failed")
        return None, warnings

    rows = _sessions_for_workspace(workspace)
    resolved = _pick_session(rows, warnings)
    if not resolved:
        warnings.append("opencode_no_active_session")
        return None, warnings

    return resolved, warnings


def _text_parts(conn: sqlite3.Connection, message_id: str) -> str | None:
    rows = conn.execute(
        "SELECT data FROM part WHERE message_id = ? ORDER BY time_created",
        (message_id,),
    ).fetchall()
    texts: list[str] = []
    for row in rows:
        try:
            part = json.loads(row["data"])
        except json.JSONDecodeError:
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    if not texts:
        return None
    return "\n".join(texts)


def _parse_session_messages(session_id: str) -> list[dict[str, str]]:
    conn = _connect_readonly()
    if not conn:
        return []

    messages: list[dict[str, str]] = []
    try:
        rows = conn.execute(
            "SELECT id, data FROM message WHERE session_id = ? ORDER BY time_created",
            (session_id,),
        ).fetchall()
        for row in rows:
            try:
                meta = json.loads(row["data"])
            except json.JSONDecodeError:
                continue

            role = meta.get("role")
            if role == "user":
                content = _text_parts(conn, row["id"])
                if content:
                    messages.append({"role": "user", "content": content})
            elif role == "assistant":
                if meta.get("finish") == "tool-calls":
                    continue
                content = _text_parts(conn, row["id"])
                if content:
                    messages.append({"role": "assistant", "content": content})
    except sqlite3.Error as exc:
        logger.error("OpenCodeCliProvider: failed reading session %s: %s", session_id, exc)
    finally:
        conn.close()

    return messages


class OpenCodeCliProvider(BaseTranscriptProvider):
    def can_handle(self, conversation_id: str | None = None, client: str | None = None) -> bool:
        if client and client != "opencode":
            return False

        session_id = conversation_id or os.environ.get("OPENCODE_SESSION_ID")
        if not session_id:
            return False

        return _session_exists(session_id)

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        session_id = conversation_id or os.environ.get("OPENCODE_SESSION_ID")
        if not session_id:
            logger.warning(
                "OpenCodeCliProvider: No session ID provided or found in environment."
            )
            return []

        if not _session_exists(session_id):
            logger.warning(
                "OpenCodeCliProvider: Session not found in opencode.db: %s",
                session_id,
            )
            return []

        return _parse_session_messages(session_id)