"""Tests for helper functions in tools.py and utilities."""

import pytest
from alambique.tools import _session_id_to_rowid, _rowid_to_session_id


class TestSessionIdMapping:
    """vec0_sessions requires integer rowids. Session IDs are strings (sess_<hex>).
    We encode the hex portion as an integer and can round-trip it."""

    def test_roundtrip_normal(self):
        sid = "sess_abc123def456"
        assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_roundtrip_all_zeros(self):
        sid = "sess_000000000000"
        assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_roundtrip_all_f(self):
        sid = "sess_ffffffffffff"
        assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_roundtrip_mixed(self):
        sid = "sess_a1b2c3d4e5f6"
        assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_rowid_is_positive(self):
        sid = "sess_000000000001"
        assert _session_id_to_rowid(sid) > 0

    def test_rowid_fits_in_64bit(self):
        sid = "sess_ffffffffffff"
        assert _session_id_to_rowid(sid) < 2**63


class TestTTLExpression:
    """Verify the TTL SQL expression we use in database.py is correct."""

    def test_ttl_expression(self):
        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.execute("""
            CREATE TABLE test_facts (
                id INTEGER PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                ttl INTEGER
            )
        """)

        TTL_SQL = """(ttl IS NULL OR
            (CAST(strftime('%s','now') AS INTEGER) -
             CAST(strftime('%s', created_at) AS INTEGER)) < ttl)"""

        conn.execute(
            "INSERT INTO test_facts (created_at, ttl) VALUES (datetime('now', '-30 seconds'), 60)"
        )
        conn.execute(
            "INSERT INTO test_facts (created_at, ttl) VALUES (datetime('now', '-120 seconds'), 60)"
        )
        conn.execute(
            "INSERT INTO test_facts (created_at, ttl) VALUES (datetime('now', '-5 seconds'), NULL)"
        )
        conn.execute(
            "INSERT INTO test_facts (created_at, ttl) VALUES (datetime('now', '-5 seconds'), 0)"
        )

        rows = conn.execute(
            f"SELECT id FROM test_facts WHERE {TTL_SQL}"
        ).fetchall()
        ids = {r[0] for r in rows}

        assert 1 in ids, "30s old, TTL 60 → should be valid"
        assert 2 not in ids, "120s old, TTL 60 → should be expired"
        assert 3 in ids, "TTL NULL → should be valid (no expiration)"
        assert 4 not in ids, "TTL 0 → should be expired immediately"

        conn.close()
