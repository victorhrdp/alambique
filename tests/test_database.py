"""Tests for Database layer (SQLite, no external dependencies)."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from alambique.database import Database, SCHEMA_VERSION
from alambique.models import (
    Consolidation,
    ConsolidationAction,
    Message,
    Session,
    SessionStatus,
)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_alambique.db"


@pytest.fixture
def db(db_path):
    d = Database(db_path)
    d.connect()
    yield d
    d.close()


class TestDatabaseLifecycle:
    def test_connect_creates_db(self, db_path):
        assert not db_path.exists()
        d = Database(db_path)
        d.connect()
        assert db_path.exists()
        d.close()

    def test_schema_version(self, db):
        v = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert v == SCHEMA_VERSION

    def test_tables_exist(self, db):
        tables = {
            r[0]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for t in ("sessions", "messages", "consolidations", "initiatives"):
            assert t in tables, f"Table {t} not found"

    def test_wal_mode(self, db):
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

    def test_foreign_keys_on(self, db):
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestSessions:
    def test_create_session(self, db):
        s = db.create_session()
        assert s.id.startswith("sess_")
        assert s.status == SessionStatus.OPEN
        assert s.consolidated is False

    def test_get_session(self, db):
        s = db.create_session()
        retrieved = db.get_session(s.id)
        assert retrieved.id == s.id

    def test_get_session_nonexistent(self, db):
        assert db.get_session("bad_id") is None

    def test_get_active_session(self, db):
        s1 = db.create_session()
        s2 = db.get_active_session()
        assert s2.id == s1.id

    def test_close_session(self, db):
        s = db.create_session()
        closed = db.close_session(s.id)
        assert closed.status == SessionStatus.CLOSED
        assert closed.ended_at is not None

    def test_close_session_truncated(self, db):
        s = db.create_session()
        closed = db.close_session(s.id, SessionStatus.TRUNCATED)
        assert closed.status == SessionStatus.TRUNCATED

    def test_set_session_summary(self, db):
        s = db.create_session()
        db.close_session(s.id)
        db.set_session_summary(s.id, "Discusión sobre Python")
        updated = db.get_session(s.id)
        assert updated.summary == "Discusión sobre Python"
        assert updated.consolidated is True

    def test_get_pending_consolidations(self, db):
        s = db.create_session()
        db.close_session(s.id)
        pending = db.get_pending_consolidations()
        assert len(pending) == 1
        assert pending[0].id == s.id

    def test_no_pending_after_consolidation(self, db):
        s = db.create_session()
        db.close_session(s.id)
        db.set_session_summary(s.id, "done")
        pending = db.get_pending_consolidations()
        assert len(pending) == 0

    def test_find_stale_sessions_uses_sqlite_clock(self, db):
        fresh = db.create_session()
        stale = db.create_session()
        db.conn.execute(
            "UPDATE sessions SET created_at = datetime('now', '-45 minutes') WHERE id = ?",
            (stale.id,),
        )
        db.conn.commit()

        found = db.find_stale_sessions(timeout_minutes=30)
        ids = {s.id for s in found}
        assert stale.id in ids
        assert fresh.id not in ids


class TestMessages:
    def test_append_message(self, db):
        s = db.create_session()
        msg = Message(session_id=s.id, role="user", content="Hola")
        msg_id = db.append_message(msg)
        assert msg_id > 0

    def test_get_session_messages(self, db):
        s = db.create_session()
        db.append_message(Message(session_id=s.id, role="user", content="msg1"))
        db.append_message(Message(session_id=s.id, role="assistant", content="msg2"))
        msgs = db.get_session_messages(s.id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_message_with_tool_calls(self, db):
        s = db.create_session()
        msg = Message(
            session_id=s.id,
            role="assistant",
            content="result",
            tool_calls='[{"name": "echo"}]',
            tool_results='{"output": "ok"}',
        )
        msg_id = db.append_message(msg)
        msgs = db.get_session_messages(s.id)
        assert msgs[0].tool_calls == '[{"name": "echo"}]'
        assert msgs[0].tool_results == '{"output": "ok"}'

    def test_session_message_count(self, db):
        s = db.create_session()
        for i in range(5):
            db.append_message(Message(session_id=s.id, role="user", content=f"msg{i}"))
        assert db.session_message_count(s.id) == 5

    def test_search_messages_fts(self, db):
        s = db.create_session()
        db.append_message(Message(session_id=s.id, role="user", content="palabra_secreta"))
        db.append_message(Message(session_id=s.id, role="user", content="otro mensaje"))
        results = db.search_messages_fts("palabra_secreta")
        assert len(results) >= 1
        assert "palabra_secreta" in results[0]["content"]


# TestFacts class removed - legacy facts system eliminated

class TestConsolidations:
    def test_insert_consolidation(self, db):
        s = db.create_session()
        db.close_session(s.id)
        c = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.CREATE,
            thread_id=42,
            reason="test thread create",
        )
        cid = db.insert_consolidation(c)
        assert cid > 0

    def test_insert_consolidation_new_model(self, db):
        s = db.create_session()
        db.close_session(s.id)
        c = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.UPDATE,
            capsule_scope="general",
            reason="capsule update",
        )
        cid = db.insert_consolidation(c)
        assert cid > 0

        c2 = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.CREATE,
            echo_id=99,
            reason="echo create",
        )
        cid2 = db.insert_consolidation(c2)
        assert cid2 > 0

    def test_count_pending(self, db):
        s = db.create_session()
        db.close_session(s.id)
        assert db.count_pending_consolidations_db() == 1

    def test_last_consolidation(self, db):
        s = db.create_session()
        db.close_session(s.id)
        c = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.CREATE,
            thread_id=1,
            reason="test",
        )
        db.insert_consolidation(c)
        last = db.last_consolidation_time()
        assert last is not None

    def test_new_entity_embedding_counts(self, db):
        # Add some threads etc to test counts (no vec yet)
        db.conn.execute("INSERT INTO threads (key, title, current_state) VALUES ('t1', 't1', 'state')")
        db.conn.execute("INSERT INTO relationship_capsules (scope, content) VALUES ('c1', 'cont')")
        db.conn.execute("INSERT INTO echoes (content) VALUES ('e1')")
        db.conn.commit()
        assert db.count_threads_missing_embeddings() >= 1
        assert db.count_capsules_missing_embeddings() >= 1
        assert db.count_echoes_missing_embeddings() >= 1


def test_legacy_facts_migration(tmp_path):
        # Simulate old DB with facts table for personality
        d = Database(tmp_path / "old.db")
        d.connect()
        d.conn.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT, value TEXT, category TEXT, confidence REAL)")
        d.conn.execute("INSERT INTO facts (key, value, category, confidence) VALUES ('trait1', 'val1', 'personality', 1.0)")
        d.conn.commit()
        d._migrate_legacy_facts_to_capsules()
        cap = d.get_relevant_relationship_capsule("personality")
        assert cap is not None and ("trait1" in cap or "personality" in cap.lower())
        assert not d._table_exists("facts")
        d.close()

class TestThreadRetrieval:
    def test_recent_active_orders_by_last_active_not_salience(self, db):
        db.create_thread(
            key="old_famous",
            title="Old",
            current_state="Estado largo suficiente para un hilo famoso pero viejo.",
            tone_guidance="t",
            salience=0.99,
        )
        db.create_thread(
            key="new_quiet",
            title="New",
            current_state="Estado largo suficiente para un hilo nuevo y menos saliente.",
            tone_guidance="t",
            salience=0.4,
        )
        # Bump only new_quiet as more recent
        db.conn.execute(
            "UPDATE threads SET last_active_at = datetime('now', '-2 days') WHERE key = 'old_famous'"
        )
        db.conn.execute(
            "UPDATE threads SET last_active_at = datetime('now') WHERE key = 'new_quiet'"
        )
        db.conn.commit()
        recent = db.get_recent_active_threads(limit=2)
        assert recent[0]["key"] == "new_quiet"
        high = db.get_high_salience_recent_threads(limit=2)
        assert high[0]["key"] == "old_famous"


class TestMemoryExport:
    def test_get_all_sessions(self, db):
        s1 = db.create_session()
        s2 = db.create_session()
        sessions = db.get_all_sessions()
        assert len(sessions) == 2


class TestInitiatives:
    def test_create_and_get_pending(self, db):
        session = db.create_session()
        iid = db.create_initiative(
            "Pregúntale a Víctor si quiere bajar el MVP de iniciativas a un diff concreto.",
            thread_key="alambique_autonomy_design",
            source_session_id=session.id,
            ttl_sessions=3,
        )
        pending = db.get_pending_initiative()
        assert pending is not None
        assert pending["id"] == iid
        assert pending["status"] == "pending"
        assert "MVP" in pending["prompt_payload"]
        assert pending["thread_key"] == "alambique_autonomy_design"

    def test_single_slot_supersedes_previous(self, db):
        db.create_initiative("Primera iniciativa pendiente lo bastante larga para pasar.")
        second = db.create_initiative(
            "Segunda iniciativa que debe dejar solo un slot pendiente activo."
        )
        pending = db.get_pending_initiative()
        assert pending is not None
        assert pending["id"] == second
        rows = db.conn.execute(
            "SELECT status FROM initiatives ORDER BY id"
        ).fetchall()
        assert [r["status"] for r in rows] == ["superseded", "pending"]

    def test_injection_ttl_sessions(self, db):
        iid = db.create_initiative(
            "Iniciativa con TTL de dos arranques de sesión para probar caducidad.",
            ttl_sessions=2,
        )
        db.record_initiative_injection(iid)
        p = db.get_pending_initiative()
        assert p is not None
        assert p["sessions_seen"] == 1
        assert p["status"] == "pending"

        db.record_initiative_injection(iid)
        # After 2 injections, expired; no longer pending
        assert db.get_pending_initiative() is None
        row = db.conn.execute(
            "SELECT status, sessions_seen FROM initiatives WHERE id = ?", (iid,)
        ).fetchone()
        assert row["status"] == "expired"
        assert row["sessions_seen"] == 2

    def test_expire_by_age(self, db):
        iid = db.create_initiative(
            "Iniciativa antigua que debería caducar por edad en días."
        )
        db.conn.execute(
            "UPDATE initiatives SET created_at = datetime('now', '-20 days') WHERE id = ?",
            (iid,),
        )
        db.conn.commit()
        assert db.get_pending_initiative() is None
        row = db.conn.execute(
            "SELECT status FROM initiatives WHERE id = ?", (iid,)
        ).fetchone()
        assert row["status"] == "expired"


def test_legacy_facts_migration(tmp_path):
    # Simulate old DB with facts table
    d = Database(tmp_path / "old.db")
    d.connect()
    # create minimal facts
    d.conn.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER, key TEXT, value TEXT, category TEXT, confidence REAL)")
    d.conn.execute("INSERT INTO facts (key, value, category, confidence) VALUES ('trait1', 'val1', 'personality', 1.0)")
    d.conn.commit()
    # call migration
    d._migrate_legacy_facts_to_capsules()
    # should have capsule
    cap = d.get_relevant_relationship_capsule("personality")
    assert "trait1" in cap or "personality" in (cap or "")
    # facts gone
    assert not d._table_exists("facts")
    d.close()

V1_BOOTSTRAP_SQL = """
CREATE TABLE sessions (
    id            TEXT PRIMARY KEY,
    agent         TEXT NOT NULL DEFAULT 'lucy',
    status        TEXT NOT NULL DEFAULT 'open',
    consolidated  INTEGER NOT NULL DEFAULT 0,
    summary       TEXT,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at      TEXT
);
CREATE TABLE messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,
    content       TEXT NOT NULL,
    tool_calls    TEXT,
    tool_results  TEXT,
    timestamp     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE consolidations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    action          TEXT NOT NULL,
    previous_value  TEXT,
    new_value       TEXT,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE agents (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL
);
"""


def _seed_v1_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(V1_BOOTSTRAP_SQL)
    conn.execute("INSERT INTO agents (id, name) VALUES ('lucy', 'Lucy')")
    conn.execute("INSERT INTO agents (id, name) VALUES ('aria', 'Aria')")
    conn.execute(
        "INSERT INTO sessions (id, agent, status) VALUES ('sess_lucy1', 'lucy', 'closed')"
    )
    conn.execute(
        "INSERT INTO sessions (id, agent, status) VALUES ('sess_aria1', 'aria', 'closed')"
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content) "
        "VALUES ('sess_lucy1', 'user', 'hola lucy')"
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content) "
        "VALUES ('sess_aria1', 'user', 'hola aria')"
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


class TestV1Migration:
    def test_backup_database_creates_copy(self, db, db_path):
        backup = db._backup_database()
        assert backup.exists()
        assert backup.name.startswith("test_alambique.bak-")
        assert backup.suffix == ".db"

    def test_migrate_v1_to_v2_preserves_lucy_data(self, tmp_path):
        path = tmp_path / "legacy.db"
        _seed_v1_database(path)

        d = Database(path)
        d.connect()

        version = d.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        assert not d._column_exists("sessions", "agent")
        assert not d._table_exists("agents")

        assert d.get_session("sess_lucy1") is not None
        assert d.get_session("sess_aria1") is None
        lucy_msgs = d.get_session_messages("sess_lucy1")
        assert len(lucy_msgs) == 1
        assert lucy_msgs[0].content == "hola lucy"

        backups = list(tmp_path.glob("legacy.bak-*.db"))
        assert len(backups) >= 1

        d.close()


class TestV3ToV4Migration:
    def test_drops_consolidations_agent(self, tmp_path):
        path = tmp_path / "v3_legacy.db"
        d = Database(path)
        d.connect()
        d.conn.execute(
            "ALTER TABLE consolidations ADD COLUMN agent TEXT NOT NULL DEFAULT 'lucy'"
        )
        d.conn.execute("PRAGMA user_version = 3")
        d.conn.commit()
        d.close()

        d2 = Database(path)
        d2.connect()
        assert d2.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert not d2._column_exists("consolidations", "agent")

        s = d2.create_session()
        d2.close_session(s.id)
        cid = d2.insert_consolidation(
            Consolidation(
                session_id=s.id,
                action=ConsolidationAction.CREATE,
                new_value="v",
                reason="post-v4",
            )
        )
        assert cid > 0
        d2.close()


class TestV4ToV5Migration:
    def test_adds_session_binding_columns(self, tmp_path):
        path = tmp_path / "v4_legacy.db"
        d = Database(path)
        d.connect()
        d.conn.execute("PRAGMA user_version = 4")
        d.conn.commit()
        d.close()

        d2 = Database(path)
        d2.connect()
        assert d2.conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert d2._column_exists("sessions", "client")
        assert d2._column_exists("sessions", "conversation_id")

        s = d2.create_session(client="grok", conversation_id="abc-123")
        assert s.client == "grok"
        assert s.conversation_id == "abc-123"
        d2.close()