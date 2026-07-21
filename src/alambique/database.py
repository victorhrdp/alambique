"""SQLite database layer for Alambique."""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import sqlite_vec

from alambique.memory_config import STATE_DEFAULT_TTL
from alambique.vector_store import vector_knn
from alambique.models import (
    Consolidation,
    ConsolidationAction,
    Message,
    Session,
    SessionStatus,
)

logger = logging.getLogger("alambique.db")

SCHEMA_VERSION = 8

# facts index removed (legacy)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','truncated')),
    consolidated      INTEGER NOT NULL DEFAULT 0,
    summary           TEXT,
    client            TEXT,
    conversation_id   TEXT,
    expression        TEXT,
    mood_text         TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at          TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role          TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
    content       TEXT NOT NULL,
    tool_calls    TEXT,
    tool_results  TEXT,
    timestamp     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- facts table removed (legacy atomic facts replaced by threads/capsules/echoes)

CREATE TABLE IF NOT EXISTS consolidations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    action          TEXT NOT NULL,
    thread_id       INTEGER,
    capsule_scope   TEXT,
    echo_id         INTEGER,
    previous_value  TEXT,
    new_value       TEXT,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- New tables for redesigned memory (Threads as main hilo conductor)
CREATE TABLE IF NOT EXISTS threads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    current_state   TEXT,
    tone_guidance   TEXT,
    search_text     TEXT,
    salience        REAL NOT NULL DEFAULT 0.5,
    last_active_at  TEXT NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'paused', 'archived', 'merged')),
    migrated_from_legacy INTEGER DEFAULT 0,
    description     TEXT,
    open_questions  TEXT
);

CREATE TABLE IF NOT EXISTS thread_participations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id               INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    session_id              TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    contribution_summary    TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(thread_id, session_id)
);

CREATE TABLE IF NOT EXISTS relationship_capsules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scope           TEXT NOT NULL,
    content         TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,
    last_updated    TEXT NOT NULL DEFAULT (datetime('now')),
    access_count    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(scope)
);

CREATE TABLE IF NOT EXISTS echoes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       INTEGER REFERENCES threads(id) ON DELETE SET NULL,
    content         TEXT NOT NULL,
    context         TEXT,
    emotional_valence REAL,
    salience        REAL NOT NULL DEFAULT 0.6,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at    TEXT
);

-- For tracking expansions per session to enforce limits
CREATE TABLE IF NOT EXISTS session_expansions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    thread_key      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, thread_key)
);
"""

MessageFTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='id'
);
"""

MessageFTS_Triggers = """
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


class Database:
    """SQLite database manager for Alambique."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.conn: sqlite3.Connection | None = None

    # ── lifecycle ──────────────────────────────────────────────

    def connect(self) -> None:
        path_str = str(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path_str, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        self._run_migrations()
        # Always ensure auxiliary (vec, fts) and redesign tables (threads, capsules, echoes)
        # for DBs that were created before the redesign or migrations that didn't cover it.
        self._ensure_auxiliary_schema()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def _run_migrations(self) -> None:
        current = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if current == 0:
            self._bootstrap_schema()
            return
        if current < 2:
            self._migrate_v1_to_v2(current)
            current = 2
        if current < 3:
            self._migrate_v2_to_v3()
            current = 3
        if current < 4:
            self._migrate_v3_to_v4()
            current = 4
        if current < 5:
            self._migrate_v4_to_v5()
            current = 5
        if current < 6:
            self._migrate_v5_to_v6()
            current = 6
        if current < 7:
            self._migrate_v6_to_v7()
            current = 7
        if current < 8:
            self._migrate_v7_to_v8()

    def _backup_database(self) -> Path:
        """Copy the database (and WAL sidecars) before a schema migration."""
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.conn.commit()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = self.path.parent / f"{self.path.stem}.bak-{stamp}{self.path.suffix}"
        shutil.copy2(self.path, backup_path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self.path}{suffix}")
            if sidecar.exists():
                shutil.copy2(sidecar, Path(f"{backup_path}{suffix}"))
        logger.info("Backup de base de datos: %s", backup_path)
        return backup_path

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def _column_exists(self, table: str, column: str) -> bool:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(r["name"] == column for r in rows)

    def _migrate_v1_to_v2(self, from_version: int) -> None:
        """Incremental migration from multi-agent v1 to Lucy-only v2."""
        self._backup_database()
        logger.info("Migrando de v%d a v2 (incremental)...", from_version)

        has_agent = self._column_exists("sessions", "agent")
        legacy_tables = self._table_exists("agents")

        if not has_agent and not legacy_tables:
            self.conn.execute("PRAGMA user_version = 2")
            self.conn.commit()
            logger.info("Esquema ya compatible con v2; versión actualizada.")
            return

        if has_agent:
            orphan_ids = [
                row["id"]
                for row in self.conn.execute(
                    "SELECT id FROM sessions WHERE lower(agent) NOT IN ('lucy', '')"
                ).fetchall()
            ]
            if orphan_ids:
                self.delete_sessions(orphan_ids)
            self.conn.execute("ALTER TABLE sessions DROP COLUMN agent")
            self.conn.commit()

        if self._table_exists("agents"):
            self.conn.execute("DROP TABLE agents")

        if self._column_exists("consolidations", "agent"):
            self.conn.execute("ALTER TABLE consolidations DROP COLUMN agent")
            self.conn.commit()

        self._ensure_auxiliary_schema()
        self.conn.execute("PRAGMA user_version = 2")
        self.conn.commit()
        logger.info("Migración v1→v2 completada.")

    def _ensure_auxiliary_schema(self) -> None:
        """Create vec0 / FTS objects if an older DB predates them."""
        # Always ensure vec tables (IF NOT EXISTS inside covers new ones like vec0_threads)
        self._create_vec_tables()
        if not self._table_exists("messages_fts"):
            self.conn.executescript(MessageFTS_SQL)
            self.conn.executescript(MessageFTS_Triggers)
            self.conn.execute(
                "INSERT INTO messages_fts(rowid, content) SELECT id, content FROM messages"
            )
            self.conn.commit()

        # Ensure redesign tables (threads etc) exist for existing DBs
        self._ensure_redesign_tables()
        self._migrate_legacy_facts_to_capsules()

    def _ensure_redesign_tables(self) -> None:
        """Idempotent creation of new thread/capsule/echo tables (for migrations from pre-redesign)."""
        # We can run a trimmed script; IF NOT EXISTS makes it safe.
        redesign_sql = """
CREATE TABLE IF NOT EXISTS threads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    current_state   TEXT,
    tone_guidance   TEXT,
    search_text     TEXT,
    salience        REAL NOT NULL DEFAULT 0.5,
    last_active_at  TEXT NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    status          TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'paused', 'archived', 'merged')),
    migrated_from_legacy INTEGER DEFAULT 0,
    description     TEXT,
    open_questions  TEXT
);

CREATE TABLE IF NOT EXISTS thread_participations (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id               INTEGER NOT NULL REFERENCES threads(id) ON DELETE CASCADE,
    session_id              TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    contribution_summary    TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(thread_id, session_id)
);

CREATE TABLE IF NOT EXISTS relationship_capsules (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    scope           TEXT NOT NULL,
    content         TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 1.0,
    last_updated    TEXT NOT NULL DEFAULT (datetime('now')),
    access_count    INTEGER NOT NULL DEFAULT 0,
    UNIQUE(scope)
);

CREATE TABLE IF NOT EXISTS echoes (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id       INTEGER REFERENCES threads(id) ON DELETE SET NULL,
    content         TEXT NOT NULL,
    context         TEXT,
    emotional_valence REAL,
    salience        REAL NOT NULL DEFAULT 0.6,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at    TEXT
);

CREATE TABLE IF NOT EXISTS session_expansions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    thread_key      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(session_id, thread_key)
);
"""
        try:
            self.conn.executescript(redesign_sql)
            self.conn.commit()
        except Exception as e:
            logger.warning("Could not ensure redesign tables: %s", e)

    def _migrate_legacy_facts_to_capsules(self) -> None:
        """Selective migration: personality facts to relationship_capsule.
        Then drop facts table if present (no legacy).
        """
        if not self._table_exists("facts"):
            return
        try:
            personality = self.conn.execute(
                "SELECT key, value FROM facts WHERE category = 'personality' AND confidence > 0"
            ).fetchall()
            if personality:
                content = "\n".join(f"- {p['key']}: {p['value']}" for p in personality)
                self.upsert_relationship_capsule("personality", content)
            # drop facts table and related
            self.conn.execute("DROP TABLE IF EXISTS facts")
            self.conn.execute("DROP TABLE IF EXISTS vec0_facts")
            self.conn.commit()
            logger.info("Migrated personality facts to capsules and removed legacy facts table.")
        except Exception as e:
            logger.warning("Facts migration skipped or partial: %s", e)

    # _deduplicate_cross_namespace_facts removed (legacy facts)

    def _bootstrap_schema(self) -> None:
        logger.info("Creando esquema inicial (v%d)...", SCHEMA_VERSION)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.executescript(MessageFTS_SQL)
        self.conn.executescript(MessageFTS_Triggers)
        self._create_vec_tables()
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()
        logger.info("Esquema creado.")

    def _migrate_v2_to_v3(self) -> None:
        self._backup_database()
        logger.info("Migrando de v2 a v3 (no-op for facts removal).")
        self.conn.execute("PRAGMA user_version = 3")
        self.conn.commit()
        logger.info("Migración v2→v3 completada.")

    def _migrate_v3_to_v4(self) -> None:
        """Drop legacy consolidations.agent (Lucy-only; inserts no longer supply it)."""
        if self._column_exists("consolidations", "agent"):
            self._backup_database()
            logger.info("Migrando de v3 a v4 (eliminar consolidations.agent)...")
            self.conn.execute("ALTER TABLE consolidations DROP COLUMN agent")
        self.conn.execute("PRAGMA user_version = 4")
        self.conn.commit()
        logger.info("Migración v3→v4 completada.")

    def _migrate_v4_to_v5(self) -> None:
        """Add client/conversation_id binding columns to sessions."""
        self._backup_database()
        logger.info("Migrando de v4 a v5 (binding client/conversation_id en sessions)...")
        if not self._column_exists("sessions", "client"):
            self.conn.execute("ALTER TABLE sessions ADD COLUMN client TEXT")
        if not self._column_exists("sessions", "conversation_id"):
            self.conn.execute("ALTER TABLE sessions ADD COLUMN conversation_id TEXT")
        self.conn.execute("PRAGMA user_version = 5")
        self.conn.commit()
        logger.info("Migración v4→v5 completada.")

    def _migrate_v5_to_v6(self) -> None:
        """Add avatar expression columns to sessions."""
        self._backup_database()
        logger.info("Migrando de v5 a v6 (expression/mood_text en sessions)...")
        if not self._column_exists("sessions", "expression"):
            self.conn.execute("ALTER TABLE sessions ADD COLUMN expression TEXT")
        if not self._column_exists("sessions", "mood_text"):
            self.conn.execute("ALTER TABLE sessions ADD COLUMN mood_text TEXT")
        self.conn.execute("PRAGMA user_version = 6")
        self.conn.commit()
        logger.info("Migración v5→v6 completada.")

    def _migrate_v6_to_v7(self) -> None:
        """Add description and open_questions to threads table (for new model)."""
        self._backup_database()
        logger.info("Migrando de v6 a v7 (description y open_questions en threads)...")
        if not self._column_exists("threads", "description"):
            self.conn.execute("ALTER TABLE threads ADD COLUMN description TEXT")
        if not self._column_exists("threads", "open_questions"):
            self.conn.execute("ALTER TABLE threads ADD COLUMN open_questions TEXT")
        self.conn.execute("PRAGMA user_version = 7")
        self.conn.commit()
        logger.info("Migración v6→v7 completada.")

    def _migrate_v7_to_v8(self) -> None:
        """Enhance consolidations table for better audit of threads/capsules/echoes."""
        self._backup_database()
        logger.info("Migrando de v7 a v8 (audit columns for new memory entities)...")
        if not self._column_exists("consolidations", "thread_id"):
            self.conn.execute("ALTER TABLE consolidations ADD COLUMN thread_id INTEGER")
        if not self._column_exists("consolidations", "capsule_scope"):
            self.conn.execute("ALTER TABLE consolidations ADD COLUMN capsule_scope TEXT")
        if not self._column_exists("consolidations", "echo_id"):
            self.conn.execute("ALTER TABLE consolidations ADD COLUMN echo_id INTEGER")
        self.conn.execute("PRAGMA user_version = 8")
        self.conn.commit()
        logger.info("Migración v7→v8 completada.")

    def get_latest_open_session_row(self) -> sqlite3.Row | None:
        if self._column_exists("sessions", "expression"):
            return self.conn.execute(
                "SELECT id, client, conversation_id, expression, mood_text "
                "FROM sessions WHERE status = 'open' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return self.conn.execute(
            "SELECT id, client, conversation_id FROM sessions "
            "WHERE status = 'open' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    # facts key index and dedup removed (legacy)

    def _create_vec_tables(self) -> None:
        """Create sqlite-vec virtual tables for sessions, threads, capsules and echoes."""
        for name in ["vec0_sessions", "vec0_threads", "vec0_relationship_capsules", "vec0_echoes"]:
            self.conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING vec0(embedding FLOAT[1024])"
            )

    # ── sessions ───────────────────────────────────────────────

    def create_session(
        self,
        client: str | None = None,
        conversation_id: str | None = None,
    ) -> Session:
        sid = f"sess_{uuid.uuid4().hex[:12]}"
        self.conn.execute(
            "INSERT INTO sessions (id, status, client, conversation_id) VALUES (?, 'open', ?, ?)",
            (sid, client, conversation_id),
        )
        self.conn.commit()
        return self.get_session(sid)

    def get_session(self, session_id: str) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        return Session(**dict(row))

    def get_active_session(self) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'open' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return Session(**dict(row))

    def get_open_sessions_by_binding(
        self, client: str, conversation_id: str
    ) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'open' "
            "AND client = ? AND conversation_id = ? "
            "ORDER BY created_at DESC",
            (client, conversation_id),
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def get_open_session_by_binding(
        self, client: str, conversation_id: str
    ) -> Session | None:
        sessions = self.get_open_sessions_by_binding(client, conversation_id)
        return sessions[0] if sessions else None

    def get_sessions_by_binding(
        self, client: str, conversation_id: str
    ) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions "
            "WHERE client = ? AND conversation_id = ? "
            "ORDER BY created_at DESC",
            (client, conversation_id),
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def get_session_by_binding(
        self, client: str, conversation_id: str
    ) -> Session | None:
        sessions = self.get_sessions_by_binding(client, conversation_id)
        return sessions[0] if sessions else None

    def reopen_session(self, session_id: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET status = \"open\", ended_at = NULL WHERE id = ?",
            (session_id,),
        )
        self.conn.commit()

    def get_open_sessions(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'open' ORDER BY created_at"
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def close_session(
        self, session_id: str, status: SessionStatus = SessionStatus.CLOSED
    ) -> Session | None:
        self.conn.execute(
            "UPDATE sessions SET status = ?, ended_at = datetime('now') WHERE id = ?",
            (status.value, session_id),
        )
        self.conn.commit()
        return self.get_session(session_id)

    def set_session_summary(self, session_id: str, summary: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET summary = ?, consolidated = 1 WHERE id = ?",
            (summary, session_id),
        )
        self.conn.commit()

    def update_session_expression(
        self, session_id: str, expression: str, mood_text: str
    ) -> None:
        self.conn.execute(
            "UPDATE sessions SET expression = ?, mood_text = ? WHERE id = ?",
            (expression, mood_text, session_id),
        )
        self.conn.commit()

    def get_pending_consolidations(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status IN ('closed','truncated') AND consolidated = 0 ORDER BY ended_at"
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def find_stale_sessions(self, timeout_minutes: int = 30) -> list[Session]:
        """Compute staleness in Python to avoid SQLite 'now' timezone mismatches with stored naive timestamps."""
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status = 'open'"
        ).fetchall()
        stale = []
        # Use UTC to match sqlite datetime('now') behavior (which is UTC)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        cutoff = now - timedelta(minutes=timeout_minutes)
        for row in rows:
            sdict = dict(row)
            last_str = None
            # get last message ts if any
            msg = self.conn.execute(
                "SELECT MAX(timestamp) as ts FROM messages WHERE session_id = ?",
                (sdict["id"],)
            ).fetchone()
            if msg and msg["ts"]:
                last_str = msg["ts"]
            if not last_str:
                last_str = sdict.get("created_at")
            if last_str:
                try:
                    # handle possible 'Z' or offset
                    last_str = last_str.replace('Z', '')
                    if '+' in last_str or last_str.count('-') > 2:  # has offset?
                        last = datetime.fromisoformat(last_str)
                    else:
                        last = datetime.fromisoformat(last_str)
                    if last.tzinfo is not None:
                        last = last.replace(tzinfo=None)  # make naive for compare
                    if last < cutoff:
                        stale.append(Session(**sdict))
                except Exception:
                    # if parse fails, don't consider stale to be safe
                    pass
        return stale

    def session_message_count(self, session_id: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row["cnt"]

    # ── messages ───────────────────────────────────────────────

    def append_message(self, msg: Message) -> int:
        cur = self.conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_calls, tool_results) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg.session_id, msg.role, msg.content, msg.tool_calls, msg.tool_results),
        )
        self.conn.commit()
        return cur.lastrowid

    def clear_and_set_session_messages(self, session_id: str, messages: list[Message]) -> None:
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        for msg in messages:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content, tool_calls, tool_results) "
                "VALUES (?, ?, ?, ?, ?)",
                (msg.session_id, msg.role, msg.content, msg.tool_calls, msg.tool_results),
            )
        self.conn.commit()

    def get_session_messages(self, session_id: str) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [Message(**dict(r)) for r in rows]

    def get_session_messages_range(self, session_id: str, offset: int = 0, limit: int = 15) -> tuple[list[Message], int]:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND role IN ('user','assistant') "
            "ORDER BY id LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()
        return [Message(**dict(r)) for r in rows], total

    def search_messages_fts(self, query: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT m.id, m.session_id, m.role, m.content, m.timestamp "
            "FROM messages_fts f JOIN messages m ON f.rowid = m.id "
            "WHERE messages_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # insert_fact / update_fact removed (legacy facts eliminated)

    def delete_session_embedding(self, session_id: str, *, commit: bool = True) -> bool:
        from alambique.vector_store import delete_embedding

        if not self._table_exists("vec0_sessions"):
            return False
        return delete_embedding(
            self.conn, "vec0_sessions", session_id, commit=commit
        )

    def delete_sessions(self, session_ids: list[str]) -> dict[str, int]:
        """Delete sessions and dependent rows (messages, consolidations, vec0)."""
        if not session_ids:
            return {
                "sessions": 0,
                "messages": 0,
                "consolidations": 0,
                "session_embeddings": 0,
            }

        ph = ",".join("?" for _ in session_ids)
        messages = self.conn.execute(
            f"SELECT COUNT(*) AS cnt FROM messages WHERE session_id IN ({ph})",
            session_ids,
        ).fetchone()["cnt"]
        consolidations = self.conn.execute(
            f"SELECT COUNT(*) AS cnt FROM consolidations WHERE session_id IN ({ph})",
            session_ids,
        ).fetchone()["cnt"]

        embeddings_removed = 0
        for sid in session_ids:
            if self.delete_session_embedding(sid, commit=False):
                embeddings_removed += 1

        self.conn.execute(
            f"DELETE FROM messages WHERE session_id IN ({ph})", session_ids
        )
        self.conn.execute(
            f"DELETE FROM consolidations WHERE session_id IN ({ph})", session_ids
        )
        cur = self.conn.execute(
            f"DELETE FROM sessions WHERE id IN ({ph})", session_ids
        )
        self.conn.commit()
        return {
            "sessions": cur.rowcount,
            "messages": messages,
            "consolidations": consolidations,
            "session_embeddings": embeddings_removed,
        }

    # forget_fact, count_stale_embeddings, cleanup_stale_embeddings removed (legacy facts)

    def count_orphan_session_embeddings(self) -> int:
        """vec0_sessions rows whose session_id no longer exists."""
        if not self._table_exists("vec0_sessions"):
            return 0
        from alambique.vector_store import rowid_to_session_id

        count = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_sessions").fetchall():
            sid = rowid_to_session_id(row["rowid"])
            exists = self.conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            if not exists:
                count += 1
        return count

    def count_sessions_missing_embeddings(self) -> int:
        """Sessions with summary but no vec0_sessions row."""
        if not self._table_exists("vec0_sessions"):
            return 0
        from alambique.vector_store import has_embedding

        rows = self.conn.execute(
            "SELECT id FROM sessions WHERE summary IS NOT NULL AND summary != ''"
        ).fetchall()
        return sum(
            1
            for row in rows
            if not has_embedding(self.conn, "vec0_sessions", row["id"])
        )

    def count_threads_missing_embeddings(self) -> int:
        """Threads but no vec0_threads row."""
        if not self._table_exists("vec0_threads"):
            return 0
        from alambique.vector_store import has_embedding

        rows = self.conn.execute("SELECT id FROM threads").fetchall()
        return sum(
            1
            for row in rows
            if not has_embedding(self.conn, "vec0_threads", row["id"])
        )

    def count_capsules_missing_embeddings(self) -> int:
        if not self._table_exists("vec0_relationship_capsules"):
            return 0
        from alambique.vector_store import has_embedding

        rows = self.conn.execute("SELECT id FROM relationship_capsules").fetchall()
        return sum(
            1
            for row in rows
            if not has_embedding(self.conn, "vec0_relationship_capsules", row["id"])
        )

    def count_echoes_missing_embeddings(self) -> int:
        if not self._table_exists("vec0_echoes"):
            return 0
        from alambique.vector_store import has_embedding

        rows = self.conn.execute("SELECT id FROM echoes").fetchall()
        return sum(
            1
            for row in rows
            if not has_embedding(self.conn, "vec0_echoes", row["id"])
        )

    def cleanup_orphan_session_embeddings(self) -> int:
        """Remove vec0_sessions rows for deleted sessions."""
        if not self._table_exists("vec0_sessions"):
            return 0
        from alambique.vector_store import rowid_to_session_id

        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_sessions").fetchall():
            sid = rowid_to_session_id(row["rowid"])
            exists = self.conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            if not exists:
                self.conn.execute(
                    "DELETE FROM vec0_sessions WHERE rowid = ?", (row["rowid"],)
                )
                removed += 1
        self.conn.commit()
        return removed

    def count_orphan_thread_embeddings(self) -> int:
        if not self._table_exists("vec0_threads"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_threads").fetchall():
            tid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM threads WHERE id = ?", (tid,)
            ).fetchone()
            if not exists:
                removed += 1
        return removed

    def count_orphan_capsule_embeddings(self) -> int:
        if not self._table_exists("vec0_relationship_capsules"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_relationship_capsules").fetchall():
            cid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM relationship_capsules WHERE id = ?", (cid,)
            ).fetchone()
            if not exists:
                removed += 1
        return removed

    def count_orphan_echo_embeddings(self) -> int:
        if not self._table_exists("vec0_echoes"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_echoes").fetchall():
            eid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM echoes WHERE id = ?", (eid,)
            ).fetchone()
            if not exists:
                removed += 1
        return removed

    def cleanup_orphan_thread_embeddings(self) -> int:
        if not self._table_exists("vec0_threads"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_threads").fetchall():
            tid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM threads WHERE id = ?", (tid,)
            ).fetchone()
            if not exists:
                self.conn.execute(
                    "DELETE FROM vec0_threads WHERE rowid = ?", (row["rowid"],)
                )
                removed += 1
        self.conn.commit()
        return removed

    def cleanup_orphan_capsule_embeddings(self) -> int:
        if not self._table_exists("vec0_relationship_capsules"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_relationship_capsules").fetchall():
            cid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM relationship_capsules WHERE id = ?", (cid,)
            ).fetchone()
            if not exists:
                self.conn.execute(
                    "DELETE FROM vec0_relationship_capsules WHERE rowid = ?", (row["rowid"],)
                )
                removed += 1
        self.conn.commit()
        return removed

    def cleanup_orphan_echo_embeddings(self) -> int:
        if not self._table_exists("vec0_echoes"):
            return 0
        removed = 0
        for row in self.conn.execute("SELECT rowid FROM vec0_echoes").fetchall():
            eid = row["rowid"]
            exists = self.conn.execute(
                "SELECT 1 FROM echoes WHERE id = ?", (eid,)
            ).fetchone()
            if not exists:
                self.conn.execute(
                    "DELETE FROM vec0_echoes WHERE rowid = ?", (row["rowid"],)
                )
                removed += 1
        self.conn.commit()
        return removed

    # get_fact_embedding, get_active_embedded_facts, vector_search for facts, get_fact*, get_recent_facts, get_facts, record_fact_access removed (legacy)

    def count_pending_consolidations_db(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE status IN ('closed','truncated') AND consolidated = 0"
        ).fetchone()
        return row["cnt"]

    def last_consolidation_time(self) -> str | None:
        row = self.conn.execute(
            "SELECT created_at FROM consolidations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["created_at"] if row else None

    # get_all_facts, get_embedded_fact_ids, count_facts_missing_embeddings, get_facts_missing_embeddings removed (legacy)

    def get_all_sessions(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def get_sessions(self, limit: int = 15, status: str | None = None) -> list[Session]:
        query = "SELECT * FROM sessions"
        conditions = []
        params = []
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self.conn.execute(query, params).fetchall()
        return [Session(**dict(r)) for r in rows]


    # ── consolidations ─────────────────────────────────────────

    def insert_consolidation(self, c: Consolidation) -> int:
        cur = self.conn.execute(
            "INSERT INTO consolidations (session_id, action, thread_id, capsule_scope, echo_id, previous_value, new_value, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (c.session_id, c.action.value, c.thread_id, c.capsule_scope, c.echo_id, c.previous_value, c.new_value, c.reason),
        )
        self.conn.commit()
        return cur.lastrowid

    # ── New methods for redesigned memory (Threads etc.) ─────────────────

    def get_relevant_relationship_capsule(self, scope: str = None) -> str:
        """Return the most relevant relationship capsule content."""
        if scope:
            row = self.conn.execute(
                "SELECT content FROM relationship_capsules WHERE scope = ? ORDER BY confidence DESC LIMIT 1",
                (scope,)
            ).fetchone()
            if row:
                return row["content"]
        row = self.conn.execute(
            "SELECT content FROM relationship_capsules WHERE scope = 'general' OR scope IS NULL ORDER BY confidence DESC LIMIT 1"
        ).fetchone()
        return row["content"] if row else ""

    def upsert_relationship_capsule(self, scope: str, content: str) -> int:
        """Insert or update a relationship capsule by scope. Returns the id."""
        existing = self.conn.execute(
            "SELECT id FROM relationship_capsules WHERE scope = ?", (scope,)
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE relationship_capsules SET content = ?, last_updated = datetime('now') WHERE scope = ?",
                (content, scope)
            )
            cap_id = existing[0]
        else:
            self.conn.execute(
                "INSERT INTO relationship_capsules (scope, content) VALUES (?, ?)",
                (scope, content)
            )
            cap_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.conn.commit()
        return cap_id

    def get_high_salience_recent_threads(self, limit: int = 5, max_days: int = 30) -> list[dict]:
        """Return active threads with high salience and recent activity."""
        sql = """
            SELECT * FROM threads 
            WHERE status = 'active' 
              AND (julianday('now') - julianday(last_active_at)) < ?
            ORDER BY salience DESC, last_active_at DESC 
            LIMIT ?
        """
        rows = self.conn.execute(sql, (max_days, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_thread_by_key(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM threads WHERE key = ?", (key,)).fetchone()
        return dict(row) if row else None

    def create_thread(self, **kwargs) -> int:
        """Create a new thread. kwargs: key, title, current_state, tone_guidance, search_text, salience, ..."""
        cols = ", ".join(kwargs.keys())
        placeholders = ", ".join(["?"] * len(kwargs))
        cur = self.conn.execute(
            f"INSERT INTO threads ({cols}) VALUES ({placeholders})", 
            list(kwargs.values())
        )
        self.conn.commit()
        return cur.lastrowid

    def update_thread(self, key: str, **kwargs) -> None:
        """Update a thread by key."""
        if not kwargs:
            return
        sets = ", ".join([f"{k}=?" for k in kwargs])
        values = list(kwargs.values()) + [key]
        self.conn.execute(
            f"UPDATE threads SET {sets}, last_active_at=datetime('now') WHERE key=?",
            values
        )
        self.conn.commit()

    def add_thread_participation(self, thread_id: int, session_id: str, contribution: str):
        """Record or refresh this session's contribution to a thread.

        Idempotent: re-consolidation and duplicate thread keys in one LLM
        response would otherwise hit UNIQUE(thread_id, session_id).
        """
        self.conn.execute(
            """
            INSERT INTO thread_participations (thread_id, session_id, contribution_summary)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id, session_id) DO UPDATE SET
                contribution_summary = excluded.contribution_summary,
                created_at = datetime('now')
            """,
            (thread_id, session_id, contribution),
        )
        self.conn.commit()

    def get_top_echoes_for_thread(self, thread_id: int, limit: int = 5, exclude_ids: list[int] = None) -> list[dict]:
        sql = "SELECT * FROM echoes WHERE thread_id = ?"
        params = [thread_id]
        if exclude_ids:
            ph = ",".join("?" * len(exclude_ids))
            sql += f" AND id NOT IN ({ph})"
            params.extend(exclude_ids)
        sql += " ORDER BY salience DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent_participations(self, thread_id: int, limit: int = 3) -> list[dict]:
        sql = """
            SELECT tp.*, s.created_at 
            FROM thread_participations tp
            JOIN sessions s ON tp.session_id = s.id
            WHERE tp.thread_id = ?
            ORDER BY s.created_at DESC LIMIT ?
        """
        rows = self.conn.execute(sql, (thread_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def vector_search_threads(self, embedding: list[float], limit: int = 10) -> list[dict]:
        """KNN search on vec0_threads."""
        # Assume vec0_threads exists, similar to vector_search
        from alambique.vector_store import vector_knn
        results = vector_knn(self.conn, "vec0_threads", embedding, limit=limit)
        # Join with threads table
        thread_ids = [r["rowid"] for r in results]
        if not thread_ids:
            return []
        ph = ",".join("?" for _ in thread_ids)
        rows = self.conn.execute(
            f"SELECT * FROM threads WHERE id IN ({ph})", thread_ids
        ).fetchall()
        # map back
        id_to_thread = {r["id"]: dict(r) for r in rows}
        return [{"thread": id_to_thread.get(r["rowid"]), "distance": r["distance"]} for r in results if r["rowid"] in id_to_thread]

    def vector_search_capsules(self, embedding: list[float], limit: int = 5) -> list[dict]:
        """KNN search on vec0_relationship_capsules."""
        from alambique.vector_store import vector_knn
        results = vector_knn(self.conn, "vec0_relationship_capsules", embedding, limit=limit)
        cap_ids = [r["rowid"] for r in results]
        if not cap_ids:
            return []
        ph = ",".join("?" for _ in cap_ids)
        rows = self.conn.execute(
            f"SELECT * FROM relationship_capsules WHERE id IN ({ph})", cap_ids
        ).fetchall()
        id_to_cap = {r["id"]: dict(r) for r in rows}
        return [{"capsule": id_to_cap.get(r["rowid"]), "distance": r["distance"]} for r in results if r["rowid"] in id_to_cap]

    def vector_search_echoes(self, embedding: list[float], limit: int = 5) -> list[dict]:
        """KNN search on vec0_echoes."""
        from alambique.vector_store import vector_knn
        results = vector_knn(self.conn, "vec0_echoes", embedding, limit=limit)
        echo_ids = [r["rowid"] for r in results]
        if not echo_ids:
            return []
        ph = ",".join("?" for _ in echo_ids)
        rows = self.conn.execute(
            f"SELECT * FROM echoes WHERE id IN ({ph})", echo_ids
        ).fetchall()
        id_to_echo = {r["id"]: dict(r) for r in rows}
        return [{"echo": id_to_echo.get(r["rowid"]), "distance": r["distance"]} for r in results if r["rowid"] in id_to_echo]

    def get_threads_for_session(self, session_id: str) -> list[dict]:
        """Return threads that this session participated in, ordered by salience."""
        rows = self.conn.execute(
            """SELECT t.*, tp.contribution_summary
               FROM threads t
               JOIN thread_participations tp ON t.id = tp.thread_id
               WHERE tp.session_id = ?
               ORDER BY t.salience DESC""",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_sessions_for_thread_keys(self, keys: list[str], limit: int = 10) -> list[dict]:
        """Return recent sessions that participated in given thread keys."""
        if not keys:
            return []
        ph = ",".join("?" for _ in keys)
        rows = self.conn.execute(
            f"""SELECT DISTINCT s.id, s.status, s.client, s.conversation_id,
                       s.summary, s.created_at, s.consolidated
                FROM sessions s
                JOIN thread_participations tp ON s.id = tp.session_id
                JOIN threads t ON tp.thread_id = t.id
                WHERE t.key IN ({ph}) AND s.status != 'open'
                ORDER BY s.created_at DESC LIMIT ?""",
            (*keys, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# _LEGACY_CATEGORIES and _row_to_fact removed (legacy facts)
