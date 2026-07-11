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
from alambique.models import (
    Consolidation,
    ConsolidationAction,
    Fact,
    FactCategory,
    Message,
    Session,
    SessionStatus,
)

logger = logging.getLogger("alambique.db")

SCHEMA_VERSION = 5

FACTS_KEY_ACTIVE_INDEX_SQL = (
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_key_active "
    "ON facts(key) WHERE confidence > 0"
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','closed','truncated')),
    consolidated      INTEGER NOT NULL DEFAULT 0,
    summary           TEXT,
    client            TEXT,
    conversation_id   TEXT,
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

CREATE TABLE IF NOT EXISTS facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    category        TEXT NOT NULL,
    ttl             INTEGER,
    confidence      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS consolidations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    action          TEXT NOT NULL,
    fact_id         INTEGER REFERENCES facts(id) ON DELETE SET NULL,
    previous_value  TEXT,
    new_value       TEXT,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
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

        has_namespace = self._column_exists("facts", "namespace")
        has_agent = self._column_exists("sessions", "agent")
        legacy_tables = self._table_exists("agents") or self._table_exists("fact_tags")

        if not has_namespace and not has_agent and not legacy_tables:
            self.migrate_legacy_categories()
            self.conn.execute("PRAGMA user_version = 2")
            self.conn.commit()
            logger.info("Esquema ya compatible con v2; versión actualizada.")
            return

        if has_namespace:
            self.conn.execute(
                "DELETE FROM facts WHERE lower(namespace) NOT IN ('lucy', 'shared')"
            )
            self._deduplicate_cross_namespace_facts()
            self.conn.execute("ALTER TABLE facts DROP COLUMN namespace")
            self.conn.commit()

        if has_agent:
            orphan_ids = [
                row["id"]
                for row in self.conn.execute(
                    "SELECT id FROM sessions WHERE lower(agent) NOT IN ('lucy', '')"
                ).fetchall()
            ]
            for sid in orphan_ids:
                self.conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                self.conn.execute(
                    "DELETE FROM consolidations WHERE session_id = ?", (sid,)
                )
                self.conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
            self.conn.execute("ALTER TABLE sessions DROP COLUMN agent")
            self.conn.commit()

        if self._table_exists("fact_tags"):
            self.conn.execute("DROP TABLE fact_tags")
        if self._table_exists("agents"):
            self.conn.execute("DROP TABLE agents")

        if self._column_exists("consolidations", "agent"):
            self.conn.execute("ALTER TABLE consolidations DROP COLUMN agent")
            self.conn.commit()

        self._ensure_auxiliary_schema()
        self.migrate_legacy_categories()
        self.conn.execute("PRAGMA user_version = 2")
        self.conn.commit()
        logger.info("Migración v1→v2 completada.")

    def _ensure_auxiliary_schema(self) -> None:
        """Create vec0 / FTS objects if an older DB predates them."""
        if not self._table_exists("vec0_facts"):
            self._create_vec_tables()
        if not self._table_exists("messages_fts"):
            self.conn.executescript(MessageFTS_SQL)
            self.conn.executescript(MessageFTS_Triggers)
            self.conn.execute(
                "INSERT INTO messages_fts(rowid, content) SELECT id, content FROM messages"
            )
            self.conn.commit()

    def _deduplicate_cross_namespace_facts(self) -> int:
        """When lucy and shared share a key, keep lucy and soft-delete shared."""
        if not self._column_exists("facts", "namespace"):
            return 0
        rows = self.conn.execute(
            "SELECT key FROM facts WHERE confidence > 0 "
            "GROUP BY key HAVING COUNT(*) > 1"
        ).fetchall()
        removed = 0
        for row in rows:
            duplicates = self.conn.execute(
                "SELECT id FROM facts WHERE key = ? AND confidence > 0 "
                "ORDER BY CASE lower(namespace) WHEN 'lucy' THEN 0 "
                "WHEN 'shared' THEN 1 ELSE 2 END, confidence DESC, id ASC",
                (row["key"],),
            ).fetchall()
            for dup in duplicates[1:]:
                self.forget_fact(dup["id"])
                removed += 1
        return removed

    def _bootstrap_schema(self) -> None:
        logger.info("Creando esquema inicial (v%d)...", SCHEMA_VERSION)
        self.conn.executescript(SCHEMA_SQL)
        self.conn.executescript(MessageFTS_SQL)
        self.conn.executescript(MessageFTS_Triggers)
        self._create_vec_tables()
        self._ensure_facts_key_index()
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()
        logger.info("Esquema creado.")

    def _migrate_v2_to_v3(self) -> None:
        self._backup_database()
        logger.info("Migrando de v2 a v3 (índice único en facts.key)...")
        removed = self._deduplicate_active_fact_keys()
        self._ensure_facts_key_index()
        self.conn.execute("PRAGMA user_version = 3")
        self.conn.commit()
        logger.info(
            "Migración v2→v3 completada (%d hechos duplicados archivados).",
            removed,
        )

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

    def _ensure_facts_key_index(self) -> None:
        self.conn.execute(FACTS_KEY_ACTIVE_INDEX_SQL)
        self.conn.commit()

    def _deduplicate_active_fact_keys(self) -> int:
        """Keep the strongest active fact per key; soft-delete the rest."""
        rows = self.conn.execute(
            "SELECT key FROM facts WHERE confidence > 0 "
            "GROUP BY key HAVING COUNT(*) > 1"
        ).fetchall()
        removed = 0
        for row in rows:
            key = row["key"]
            duplicates = self.conn.execute(
                "SELECT id FROM facts WHERE key = ? AND confidence > 0 "
                "ORDER BY confidence DESC, access_count DESC, id ASC",
                (key,),
            ).fetchall()
            for dup in duplicates[1:]:
                self.forget_fact(dup["id"])
                removed += 1
        return removed

    def _create_vec_tables(self) -> None:
        """Create sqlite-vec virtual tables for facts and sessions."""
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec0_facts USING vec0("
            "  embedding FLOAT[1024]"
            ")"
        )
        self.conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS vec0_sessions USING vec0("
            "  embedding FLOAT[1024]"
            ")"
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

    def get_pending_consolidations(self) -> list[Session]:
        rows = self.conn.execute(
            "SELECT * FROM sessions WHERE status IN ('closed','truncated') AND consolidated = 0 ORDER BY ended_at"
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

    def find_stale_sessions(self, timeout_minutes: int = 30) -> list[Session]:
        if timeout_minutes >= 0:
            modifier = f"-{timeout_minutes} minutes"
        else:
            modifier = f"{-timeout_minutes} minutes"
        rows = self.conn.execute(
            "SELECT s.* FROM sessions s WHERE s.status = 'open' "
            "AND datetime(COALESCE("
            "    (SELECT MAX(timestamp) FROM messages WHERE session_id = s.id),"
            "    s.created_at"
            ")) < datetime('now', ?)",
            (modifier,),
        ).fetchall()
        return [Session(**dict(r)) for r in rows]

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

    def insert_fact(self, fact: Fact) -> int:
        cur = self.conn.execute(
            "INSERT INTO facts (key, value, category, ttl, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (fact.key, fact.value, fact.category.value, fact.ttl, fact.confidence),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_fact(
        self,
        fact_id: int,
        value: str,
        confidence: float,
        *,
        category: FactCategory | None = None,
        ttl: int | None = None,
        clear_ttl: bool = False,
    ) -> None:
        if category is not None and (ttl is not None or clear_ttl):
            ttl_val = None if clear_ttl else ttl
            self.conn.execute(
                "UPDATE facts SET value = ?, confidence = ?, category = ?, ttl = ?, "
                "last_accessed = datetime('now') WHERE id = ?",
                (value, confidence, category.value, ttl_val, fact_id),
            )
        elif category is not None:
            self.conn.execute(
                "UPDATE facts SET value = ?, confidence = ?, category = ?, "
                "last_accessed = datetime('now') WHERE id = ?",
                (value, confidence, category.value, fact_id),
            )
        elif ttl is not None or clear_ttl:
            ttl_val = None if clear_ttl else ttl
            self.conn.execute(
                "UPDATE facts SET value = ?, confidence = ?, ttl = ?, "
                "last_accessed = datetime('now') WHERE id = ?",
                (value, confidence, ttl_val, fact_id),
            )
        else:
            self.conn.execute(
                "UPDATE facts SET value = ?, confidence = ?, last_accessed = datetime('now') "
                "WHERE id = ?",
                (value, confidence, fact_id),
            )
        self.conn.commit()

    def delete_fact_embedding(self, fact_id: int) -> bool:
        if not self._table_exists("vec0_facts"):
            return False
        cur = self.conn.execute(
            "DELETE FROM vec0_facts WHERE rowid = ?", (fact_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def forget_fact(self, fact_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET confidence = 0 WHERE id = ?", (fact_id,)
        )
        self.delete_fact_embedding(fact_id)

    def count_stale_embeddings(self) -> int:
        """Embeddings in vec0_facts for forgotten facts (confidence=0)."""
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM vec0_facts v "
            "JOIN facts f ON f.id = v.rowid WHERE f.confidence = 0"
        ).fetchone()
        return row["cnt"]

    def cleanup_stale_embeddings(self) -> int:
        """Remove vec0 rows for forgotten facts. Returns rows deleted."""
        rows = self.conn.execute(
            "SELECT v.rowid FROM vec0_facts v "
            "JOIN facts f ON f.id = v.rowid WHERE f.confidence = 0"
        ).fetchall()
        for row in rows:
            self.conn.execute("DELETE FROM vec0_facts WHERE rowid = ?", (row["rowid"],))
        self.conn.commit()
        return len(rows)

    def get_fact_embedding(self, fact_id: int) -> list[float] | None:
        from alambique.memory_maintenance import parse_embedding_blob

        row = self.conn.execute(
            "SELECT embedding FROM vec0_facts WHERE rowid = ?", (fact_id,)
        ).fetchone()
        if row is None:
            return None
        return parse_embedding_blob(row["embedding"])

    def get_active_embedded_facts(self) -> list[Fact]:
        query = (
            "SELECT f.* FROM facts f "
            "INNER JOIN vec0_facts v ON v.rowid = f.id "
            "WHERE f.confidence > 0 "
            "AND (f.ttl IS NULL OR (CAST(strftime('%s','now') AS INTEGER) "
            "- CAST(strftime('%s', f.created_at) AS INTEGER)) < f.ttl) "
            "ORDER BY f.id"
        )
        rows = self.conn.execute(query).fetchall()
        return [_row_to_fact(r) for r in rows]

    def search_similar_facts(
        self,
        embedding: list[float],
        *,
        limit: int = 20,
        max_distance: float | None = None,
    ) -> list[dict]:
        """KNN search on vec0_facts joined with active facts."""
        emb_str = f"[{','.join(str(f) for f in embedding)}]"
        conditions = [
            "f.confidence > 0",
            "(f.ttl IS NULL OR (CAST(strftime('%s','now') AS INTEGER) "
            "- CAST(strftime('%s', f.created_at) AS INTEGER)) < f.ttl)",
        ]
        where_extra = " AND " + " AND ".join(conditions)
        query = f"""
            SELECT f.id AS id, v.distance
            FROM vec0_facts v
            JOIN facts f ON f.id = v.rowid
            WHERE v.embedding MATCH '{emb_str}' AND k = {int(limit)}
            {where_extra}
            ORDER BY v.distance
        """
        rows = self.conn.execute(query).fetchall()
        results = [{"id": r["id"], "distance": r["distance"]} for r in rows]
        if max_distance is not None:
            results = [r for r in results if r["distance"] <= max_distance]
        return results

    def get_fact(self, fact_id: int) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    def get_fact_by_key(self, key: str) -> Fact | None:
        row = self.conn.execute(
            "SELECT * FROM facts WHERE key = ? AND confidence > 0",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_fact(row)

    def get_facts_by_category(self, category: FactCategory, limit: int = 10) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE category = ? AND confidence > 0 "
            "AND (ttl IS NULL OR (CAST(strftime('%s','now') AS INTEGER) - CAST(strftime('%s', created_at) AS INTEGER)) < ttl) "
            "ORDER BY created_at DESC LIMIT ?",
            (category.value, limit),
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_recent_facts(self, limit: int = 10) -> list[Fact]:
        """Recent facts visible to the agent."""
        facts: list[Fact] = []
        seen: set[int] = set()
        for cat in FactCategory:
            for fact in self.get_facts_by_category(cat, limit=limit):
                if fact.id not in seen:
                    seen.add(fact.id)
                    facts.append(fact)
        return facts

    def get_facts(
        self, categories: tuple[FactCategory, ...] | None = None
    ) -> list[Fact]:
        if categories is None:
            categories = (FactCategory.PERSONALITY, FactCategory.STATE)
        cat_vals = tuple(c.value for c in categories)
        placeholders = ",".join("?" * len(cat_vals))
        rows = self.conn.execute(
            f"SELECT * FROM facts WHERE category IN ({placeholders}) "
            "AND confidence > 0 ORDER BY confidence DESC",
            cat_vals,
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def record_fact_access(self, fact_id: int) -> None:
        self.conn.execute(
            "UPDATE facts SET access_count = access_count + 1, last_accessed = datetime('now') "
            "WHERE id = ?",
            (fact_id,),
        )
        self.conn.commit()

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

    def get_all_facts(self) -> list[Fact]:
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE confidence > 0 ORDER BY category, created_at DESC"
        ).fetchall()
        return [_row_to_fact(r) for r in rows]

    def get_embedded_fact_ids(self) -> set[int]:
        rows = self.conn.execute("SELECT rowid FROM vec0_facts").fetchall()
        return {r["rowid"] for r in rows}

    def count_facts_missing_embeddings(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM facts f "
            "WHERE f.confidence > 0 "
            "AND NOT EXISTS (SELECT 1 FROM vec0_facts v WHERE v.rowid = f.id)"
        ).fetchone()
        return row["cnt"]

    def get_facts_missing_embeddings(self) -> list[Fact]:
        """Active facts with no row in vec0_facts."""
        query = (
            "SELECT * FROM facts f "
            "WHERE f.confidence > 0 "
            "AND NOT EXISTS (SELECT 1 FROM vec0_facts v WHERE v.rowid = f.id) "
            "ORDER BY f.id"
        )
        rows = self.conn.execute(query).fetchall()
        return [_row_to_fact(r) for r in rows]

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

    def migrate_legacy_categories(self) -> dict[str, int]:
        """Rename hardware→possessions, mood→state; set TTL on states missing it."""
        cur = self.conn.cursor()
        r1 = cur.execute(
            "UPDATE facts SET category = 'possessions' WHERE category = 'hardware'"
        ).rowcount
        r2 = cur.execute(
            "UPDATE facts SET category = 'state' WHERE category = 'mood'"
        ).rowcount
        r3 = cur.execute(
            "UPDATE facts SET ttl = ? "
            "WHERE category = 'state' AND ttl IS NULL AND confidence > 0",
            (STATE_DEFAULT_TTL,),
        ).rowcount
        self.conn.commit()
        return {"possessions": r1, "state": r2, "ttl_set": r3}

    def insert_consolidation(self, c: Consolidation) -> int:
        cur = self.conn.execute(
            "INSERT INTO consolidations (session_id, action, fact_id, previous_value, new_value, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (c.session_id, c.action.value, c.fact_id, c.previous_value, c.new_value, c.reason),
        )
        self.conn.commit()
        return cur.lastrowid


_LEGACY_CATEGORIES = {
    "hardware": FactCategory.POSSESSIONS,
    "mood": FactCategory.STATE,
}


def _row_to_fact(row: sqlite3.Row) -> Fact:
    rd = dict(row)
    raw_cat = rd["category"]
    rd["category"] = _LEGACY_CATEGORIES.get(raw_cat, FactCategory(raw_cat))
    fact = Fact(**rd)
    fact.confidence = fact.get_decayed_confidence()
    return fact
