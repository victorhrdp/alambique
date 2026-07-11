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
    Fact,
    FactCategory,
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
        for t in ("sessions", "messages", "facts", "consolidations"):
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


class TestFacts:
    def test_insert_fact(self, db):
        f = Fact(
            key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid = db.insert_fact(f)
        assert fid > 0

    def test_get_fact(self, db):
        f = Fact(
            key="sarcastic",
            value="Es sarcástica",
            category=FactCategory.PERSONALITY,
            confidence=0.9,
        )
        fid = db.insert_fact(f)
        retrieved = db.get_fact(fid)
        assert retrieved.key == "sarcastic"
        assert retrieved.value == "Es sarcástica"
        assert retrieved.category == FactCategory.PERSONALITY
        assert retrieved.confidence == 0.9

    def test_get_fact_nonexistent(self, db):
        assert db.get_fact(9999) is None

    def test_update_fact(self, db):
        f = Fact(
            key="editor",
            value="VS Code",
            category=FactCategory.PREFERENCE,
        )
        fid = db.insert_fact(f)
        db.update_fact(fid, "Neovim", 1.0)
        updated = db.get_fact(fid)
        assert updated.value == "Neovim"

    def test_update_fact_category_and_ttl(self, db):
        f = Fact(
            key="neck",
            value="duele",
            category=FactCategory.PERSONAL,
        )
        fid = db.insert_fact(f)
        db.update_fact(
            fid,
            "ya no duele",
            1.0,
            category=FactCategory.STATE,
            ttl=86400,
        )
        updated = db.get_fact(fid)
        assert updated.value == "ya no duele"
        assert updated.category == FactCategory.STATE
        assert updated.ttl == 86400

    def test_migrate_legacy_categories(self, db):
        db.conn.execute(
            "INSERT INTO facts (key, value, category, confidence) "
            "VALUES ('hw1', 'GPU', 'hardware', 1.0)"
        )
        db.conn.execute(
            "INSERT INTO facts (key, value, category, confidence) "
            "VALUES ('m1', 'tired', 'mood', 1.0)"
        )
        db.conn.commit()
        counts = db.migrate_legacy_categories()
        assert counts["possessions"] >= 1
        assert counts["state"] >= 1
        hw = db.get_fact_by_key("hw1")
        assert hw.category == FactCategory.POSSESSIONS
        st = db.get_fact_by_key("m1")
        assert st.category == FactCategory.STATE
        assert st.ttl == 86400

    def test_forget_fact(self, db):
        from alambique.tools import _insert_embedding

        f = Fact(
            key="temp",
            value="to forget",
            category=FactCategory.PERSONAL,
        )
        fid = db.insert_fact(f)
        _insert_embedding(db.conn, "vec0_facts", fid, [0.1] * 1024)
        db.forget_fact(fid)
        forgotten = db.get_fact(fid)
        assert forgotten.confidence == 0
        rows = db.conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = ?", (fid,)
        ).fetchall()
        assert rows == []

    def test_get_fact_by_key(self, db):
        f = Fact(
            key="trait_1",
            value="graciosa",
            category=FactCategory.PERSONALITY,
        )
        db.insert_fact(f)
        found = db.get_fact_by_key("trait_1")
        assert found.key == "trait_1"

    def test_active_key_unique_index_exists(self, db):
        indexes = {
            r[0]
            for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_facts_key_active" in indexes

    def test_duplicate_active_key_rejected(self, db):
        db.insert_fact(
            Fact(key="city", value="Madrid", category=FactCategory.PERSONAL)
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_fact(
                Fact(key="city", value="Barcelona", category=FactCategory.PERSONAL)
            )

    def test_forgotten_key_can_be_reused(self, db):
        fid = db.insert_fact(
            Fact(key="old_pref", value="vim", category=FactCategory.PREFERENCE)
        )
        db.forget_fact(fid)
        new_id = db.insert_fact(
            Fact(key="old_pref", value="neovim", category=FactCategory.PREFERENCE)
        )
        assert new_id != fid
        active = db.get_fact_by_key("old_pref")
        assert active.value == "neovim"

    def test_deduplicate_active_fact_keys(self, db):
        db.conn.execute("DROP INDEX IF EXISTS idx_facts_key_active")
        db.insert_fact(
            Fact(key="dup", value="keeper", category=FactCategory.PERSONAL, confidence=1.0)
        )
        db.insert_fact(
            Fact(key="dup", value="loser", category=FactCategory.PERSONAL, confidence=0.6)
        )
        removed = db._deduplicate_active_fact_keys()
        assert removed == 1
        keeper = db.get_fact_by_key("dup")
        assert keeper.value == "keeper"
        db._ensure_facts_key_index()

    def test_migrate_v2_to_v3(self, db):
        db.conn.execute("DROP INDEX IF EXISTS idx_facts_key_active")
        db.conn.execute("PRAGMA user_version = 2")
        db.conn.commit()
        db.insert_fact(
            Fact(key="dup", value="best", category=FactCategory.PREFERENCE, confidence=0.9)
        )
        db.insert_fact(
            Fact(key="dup", value="weak", category=FactCategory.PREFERENCE, confidence=0.4)
        )
        db._migrate_v2_to_v3()
        version = db.conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == 3
        assert db.get_fact_by_key("dup").value == "best"
        with pytest.raises(sqlite3.IntegrityError):
            db.insert_fact(
                Fact(key="dup", value="again", category=FactCategory.PREFERENCE)
            )

    def test_get_facts_by_category(self, db):
        db.insert_fact(Fact(key="p1", value="a", category=FactCategory.PERSONALITY))
        db.insert_fact(Fact(key="p2", value="b", category=FactCategory.PERSONALITY))
        facts = db.get_facts_by_category(FactCategory.PERSONALITY)
        assert len(facts) == 2

    def test_get_facts(self, db):
        db.insert_fact(Fact(key="p1", value="a", category=FactCategory.PERSONALITY))
        db.insert_fact(Fact(key="m1", value="b", category=FactCategory.STATE, ttl=86400))
        db.insert_fact(Fact(key="s1", value="c", category=FactCategory.PERSONAL))
        facts = db.get_facts()
        assert len(facts) == 2  # Only personality and state by default

    def test_get_recent_facts(self, db):
        db.insert_fact(
            Fact(key="secret", value="dato privado", category=FactCategory.PERSONAL)
        )
        db.insert_fact(
            Fact(
                key="gaming",
                value="mató tabernero",
                category=FactCategory.PERSONAL,
            )
        )

        recent_facts = db.get_recent_facts(limit=10)
        assert len(recent_facts) == 2

    def test_get_recent_facts_samples_per_category(self, db):
        for cat in FactCategory:
            db.insert_fact(
                Fact(key=f"f_{cat.value}", value="v", category=cat)
            )
        facts = db.get_recent_facts(limit=5)
        assert len(facts) == 5

    def test_ttl_filtering_active_fact(self, db):
        """A fact with TTL=86400 created now should be found."""
        conn = db.conn
        conn.execute(
            "INSERT INTO facts (key, value, category, ttl, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            ("state_today", "happy", "state", 86400, 1.0),
        )
        conn.commit()
        facts = db.get_facts_by_category(FactCategory.STATE)
        assert len(facts) == 1

    def test_ttl_filtering_expired(self, db):
        """A fact with TTL=1 created 2 hours ago should be filtered out."""
        conn = db.conn
        conn.execute(
            "INSERT INTO facts (key, value, category, ttl, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now', '-2 hours'))",
            ("state_old", "sad", "state", 1, 1.0),
        )
        conn.commit()
        facts = db.get_facts_by_category(FactCategory.STATE)
        assert len(facts) == 0

    def test_ttl_null_always_valid(self, db):
        """A fact with TTL=NULL should never expire."""
        conn = db.conn
        conn.execute(
            "INSERT INTO facts (key, value, category, ttl, confidence, created_at) "
            "VALUES (?, ?, ?, NULL, ?, datetime('now', '-100 days'))",
            ("old_fact", "still here", "personal", 1.0),
        )
        conn.commit()
        facts = db.get_facts_by_category(FactCategory.PERSONAL)
        assert len(facts) == 1


class TestConsolidations:
    def test_insert_consolidation(self, db):
        s = db.create_session()
        db.close_session(s.id)
        f = Fact(key="k", value="v", category=FactCategory.PERSONAL)
        fid = db.insert_fact(f)
        c = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.CREATE,
            fact_id=fid,
            new_value="v",
            reason="test",
        )
        cid = db.insert_consolidation(c)
        assert cid > 0

    def test_count_pending(self, db):
        s = db.create_session()
        db.close_session(s.id)
        assert db.count_pending_consolidations_db() == 1

    def test_last_consolidation(self, db):
        s = db.create_session()
        db.close_session(s.id)
        f = Fact(key="k", value="v", category=FactCategory.PERSONAL)
        fid = db.insert_fact(f)
        c = Consolidation(
            session_id=s.id,
            action=ConsolidationAction.CREATE,
            fact_id=fid,
            new_value="v",
            reason="test",
        )
        db.insert_consolidation(c)
        last = db.last_consolidation_time()
        assert last is not None


class TestMemoryExport:
    def test_get_all_facts(self, db):
        db.insert_fact(Fact(key="k1", value="v1", category=FactCategory.PERSONAL))
        db.insert_fact(Fact(key="k2", value="v2", category=FactCategory.PERSONALITY))
        facts = db.get_all_facts()
        assert len(facts) == 2

    def test_get_all_sessions(self, db):
        s1 = db.create_session()
        s2 = db.create_session()
        sessions = db.get_all_sessions()
        assert len(sessions) == 2


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
CREATE TABLE facts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT NOT NULL,
    key             TEXT NOT NULL,
    value           TEXT NOT NULL,
    category        TEXT NOT NULL,
    ttl             INTEGER,
    confidence      REAL NOT NULL DEFAULT 1.0,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT NOT NULL DEFAULT (datetime('now')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE consolidations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    action          TEXT NOT NULL,
    fact_id         INTEGER,
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
    conn.execute(
        "INSERT INTO facts (namespace, key, value, category) "
        "VALUES ('lucy', 'user_name', 'Víctor', 'personal')"
    )
    conn.execute(
        "INSERT INTO facts (namespace, key, value, category) "
        "VALUES ('shared', 'user_name', 'Victor G', 'personal')"
    )
    conn.execute(
        "INSERT INTO facts (namespace, key, value, category) "
        "VALUES ('aria', 'game_map', 'Phandalin', 'preference')"
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
        assert not d._column_exists("facts", "namespace")
        assert not d._column_exists("sessions", "agent")
        assert not d._table_exists("agents")

        name_fact = d.get_fact_by_key("user_name")
        assert name_fact is not None
        assert name_fact.value == "Víctor"

        keys = {f.key for f in d.get_all_facts()}
        assert "game_map" not in keys

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
        fid = d2.insert_fact(Fact(key="k", value="v", category=FactCategory.PERSONAL))
        cid = d2.insert_consolidation(
            Consolidation(
                session_id=s.id,
                action=ConsolidationAction.CREATE,
                fact_id=fid,
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
