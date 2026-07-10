"""Tests for memory maintenance — validation, dedup, embedding cleanup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from alambique.database import Database
from alambique.memory_maintenance import (
    find_duplicate_pairs,
    merge_duplicate_pair,
    similarity_from_distance,
    validate_fact_classification,
)
from alambique.models import Fact, FactCategory
from alambique.tools import ToolHandler, _insert_embedding


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test_maintenance.db")
    d.connect()
    yield d
    d.close()


@pytest.fixture
def mock_ollama():
    o = MagicMock()
    o.embed = AsyncMock(return_value=[0.1] * 1024)
    o.embed_batch = AsyncMock(side_effect=lambda texts: [[0.1] * 1024 for _ in texts])
    o.health = AsyncMock(return_value=True)
    o.close = AsyncMock()
    return o


@pytest.fixture
def tools(db, mock_ollama):
    return ToolHandler(db, mock_ollama, api_key=None)


class TestSimilarityFromDistance:
    def test_identical_vectors(self):
        assert similarity_from_distance(0.0) == 1.0

    def test_threshold_boundary(self):
        sim = similarity_from_distance(0.176)
        assert sim >= 0.84


class TestValidateFactClassification:
    def test_health_in_personal_warns(self):
        warnings = validate_fact_classification("neck_pain", FactCategory.PERSONAL)
        assert len(warnings) == 1
        assert "personal" in warnings[0]

    def test_normal_personal_ok(self):
        assert validate_fact_classification("birth_date", FactCategory.PERSONAL) == []

    def test_state_health_ok(self):
        assert validate_fact_classification("neck_pain", FactCategory.STATE) == []


class TestEmbeddingCleanup:
    def test_forget_removes_embedding(self, db):
        from alambique.tools import _insert_embedding

        fid = db.insert_fact(
            Fact(key="k", value="v", category=FactCategory.PERSONAL)
        )
        _insert_embedding(db.conn, "vec0_facts", fid, [0.2] * 1024)
        db.forget_fact(fid)

        rows = db.conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = ?", (fid,)
        ).fetchall()
        assert rows == []
        assert db.get_fact(fid).confidence == 0

    def test_cleanup_stale_embeddings(self, db):
        from alambique.tools import _insert_embedding

        fid = db.insert_fact(
            Fact(key="k", value="v", category=FactCategory.PERSONAL)
        )
        _insert_embedding(db.conn, "vec0_facts", fid, [0.2] * 1024)
        db.conn.execute("UPDATE facts SET confidence = 0 WHERE id = ?", (fid,))
        db.conn.commit()

        assert db.count_stale_embeddings() == 1
        removed = db.cleanup_stale_embeddings()
        assert removed == 1
        assert db.count_stale_embeddings() == 0


class TestDeduplication:
    def test_find_duplicate_pairs_same_embedding(self, db):
        f1 = Fact(
            key="os_a",
            value="Usa CachyOS",
            category=FactCategory.PREFERENCE,
        )
        f2 = Fact(
            key="os_b",
            value="CachyOS es su sistema",
            category=FactCategory.PREFERENCE,
        )
        fid1 = db.insert_fact(f1)
        fid2 = db.insert_fact(f2)
        emb = [0.3] * 1024
        _insert_embedding(db.conn, "vec0_facts", fid1, emb)
        _insert_embedding(db.conn, "vec0_facts", fid2, emb)

        pairs = find_duplicate_pairs(db)
        assert len(pairs) == 1
        assert {pairs[0][0].id, pairs[0][1].id} == {fid1, fid2}
        assert pairs[0][2] >= 0.85

    def test_merge_keeps_higher_confidence(self, db):
        f1 = Fact(
            key="a",
            value="short",
            category=FactCategory.PREFERENCE,
            confidence=0.9,
        )
        f2 = Fact(
            key="b",
            value="longer value",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
        )
        fid1 = db.insert_fact(f1)
        fid2 = db.insert_fact(f2)

        keeper_id = merge_duplicate_pair(db, db.get_fact(fid1), db.get_fact(fid2))
        assert keeper_id == fid2
        assert db.get_fact(fid2).confidence >= 0.99
        forgotten = db.conn.execute(
            "SELECT confidence FROM facts WHERE id = ?", (fid1,)
        ).fetchone()
        assert forgotten["confidence"] == 0


class TestMemoryMaintenanceTools:

    def test_memory_deduplicate_dry_run(self, tools):
        from alambique.tools import _insert_embedding

        fid1 = tools.db.insert_fact(
            Fact(key="a", value="v1", category=FactCategory.PERSONAL)
        )
        fid2 = tools.db.insert_fact(
            Fact(key="b", value="v2", category=FactCategory.PERSONAL)
        )
        emb = [0.5] * 1024
        _insert_embedding(tools.db.conn, "vec0_facts", fid1, emb)
        _insert_embedding(tools.db.conn, "vec0_facts", fid2, emb)

        result = asyncio.run(tools.memory_deduplicate(dry_run=True))
        assert result.pairs_found == 1
        assert result.merged == 0
        assert tools.db.get_fact(fid1).confidence > 0
        assert tools.db.get_fact(fid2).confidence > 0

    def test_memory_deduplicate_merge(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        fid1 = tools.db.insert_fact(
            Fact(key="a", value="alpha", category=FactCategory.PERSONAL)
        )
        fid2 = tools.db.insert_fact(
            Fact(key="b", value="beta", category=FactCategory.PERSONAL)
        )
        emb = [0.5] * 1024
        _insert_embedding(tools.db.conn, "vec0_facts", fid1, emb)
        _insert_embedding(tools.db.conn, "vec0_facts", fid2, emb)

        result = asyncio.run(tools.memory_deduplicate(dry_run=False))
        assert result.merged == 1
        active = [f for f in (tools.db.get_fact(fid1), tools.db.get_fact(fid2)) if f.confidence > 0]
        assert len(active) == 1

    def test_memory_reembed_dry_run(self, tools):
        tools.db.insert_fact(
            Fact(key="no_emb", value="sin vector", category=FactCategory.PERSONAL)
        )
        result = asyncio.run(tools.memory_reembed(dry_run=True))
        assert result.dry_run is True
        assert result.missing_before == 1
        assert result.embedded == 0
        assert len(result.fact_ids) == 1

    def test_memory_reembed_embeds_missing(self, tools, mock_ollama):
        fid = tools.db.insert_fact(
            Fact(key="fix_me", value="necesita vector", category=FactCategory.PERSONAL)
        )
        result = asyncio.run(tools.memory_reembed())
        assert result.embedded == 1
        assert result.failed == 0
        assert tools.db.get_fact_embedding(fid) is not None

    def test_health_reports_stale_embeddings(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        fid = tools.db.insert_fact(
            Fact(key="k", value="v", category=FactCategory.PERSONAL)
        )
        _insert_embedding(tools.db.conn, "vec0_facts", fid, [0.1] * 1024)
        tools.db.conn.execute("UPDATE facts SET confidence = 0 WHERE id = ?", (fid,))
        tools.db.conn.commit()

        result = asyncio.run(tools.memory_health())
        assert "stale_embeddings" in result.warnings