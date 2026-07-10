"""Tests for ToolHandler — all 8 MCP tools with mocked external deps."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alambique.database import Database
from alambique.memory_maintenance import parse_embedding_blob
from alambique.models import (
    ConsolidationAction,
    ConsolidationFactItem,
    ConsolidationResponse,
    Fact,
    FactCategory,
    Message,
    MessageAppendOutput,
    SessionStartOutput,
    SessionEndOutput,
    SessionStatus,
    MemoryContextOutput,
)
from alambique.tools import (
    ToolHandler,
    SESSION_LIMIT,
    SESSION_WARNING_AT,
    consolidation_search_text,
    messages_for_consolidation,
)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test_tools.db")
    d.connect()
    yield d
    d.close()


@pytest.fixture
def mock_ollama():
    o = MagicMock()
    o.embed = AsyncMock(return_value=[0.1] * 1024)
    o.health = AsyncMock(return_value=True)
    o.ensure_model = AsyncMock(return_value=True)
    o.close = AsyncMock()
    return o


@pytest.fixture
def tools(db, mock_ollama):
    return ToolHandler(db, mock_ollama, api_key=None)


@pytest.fixture
def tools_online(db, mock_ollama):
    """ToolHandler with API key (online mode)."""
    # We patch recall and consolidator at the ToolHandler property level
    return ToolHandler(db, mock_ollama, api_key="test-key")


# ── session_start ─────────────────────────────────────────────────


class TestSessionStart:
    def test_starts_session_without_args(self, tools):
        result = asyncio.run(tools.session_start())
        assert isinstance(result, SessionStartOutput)
        assert result.status == "ok"
        assert result.session_id is not None
        assert result.session_id.startswith("sess_")

    def test_new_database(self, tools):
        result = asyncio.run(tools.session_start())
        assert result.status == "ok"
        assert result.is_new is True
        assert result.persona is None
        assert result.session_id.startswith("sess_")

    def test_existing_agent_offline_no_persona(self, tools):
        asyncio.run(tools.session_start())
        result = asyncio.run(tools.session_start())
        assert result.status == "ok"
        assert result.is_new is False
        assert result.persona is None  # offline = no persona composition

    def test_existing_agent_new_session_id_each_time(self, tools):
        r1 = asyncio.run(tools.session_start())
        r2 = asyncio.run(tools.session_start())
        assert r1.session_id != r2.session_id

    def test_session_stored_in_db(self, tools):
        r = asyncio.run(tools.session_start())
        s = tools.db.get_session(r.session_id)
        assert s is not None
        assert s.status == SessionStatus.OPEN

    def test_online_with_facts_composes_persona(self, tools_online, mock_ollama):
        db = tools_online.db
        f = Fact(
            key="sarcastic",
            value="Es sarcástica",
            category=FactCategory.PERSONALITY,
            confidence=0.9,
        )
        db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value="Eres Lucy. Sarcástica y genial.")
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.persona == "Eres Lucy. Sarcástica y genial."
        assert result.is_new is True

    def test_persona_seed_on_new_agent(self, tools_online):
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(
            return_value="Eres Lucy, la AI más inteligente."
        )
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(
            tools_online.session_start(persona_seed="Eres Lucy, la AI más inteligente.",
            )
        )
        assert result.is_new is True
        assert result.persona == "Eres Lucy, la AI más inteligente."
        traits = tools_online.db.get_facts(
            categories=(FactCategory.PERSONALITY,)
        )
        assert len(traits) == 1
        assert traits[0].key == "persona_seed"

    def test_persona_seed_ignored_when_traits_exist(self, tools_online):
        tools_online.db.insert_fact(
            Fact(
                key="existing",
                value="Personalidad previa",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value="Personalidad previa compuesta.")
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        asyncio.run(
            tools_online.session_start(persona_seed="Semilla que no debe guardarse")
        )
        traits = tools_online.db.get_facts(
            categories=(FactCategory.PERSONALITY,)
        )
        assert len(traits) == 1
        assert traits[0].key == "existing"

    def test_offline_uses_trait_without_llm(self, tools):
        tools.db.insert_fact(
            Fact(key="persona_seed",
                value="Eres Lucy offline.",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        result = asyncio.run(tools.session_start())
        assert result.persona == "Eres Lucy offline."
        assert result.degraded is True
        assert "offline_mode" in result.warnings
        assert "persona_offline_fallback" in result.warnings

    def test_online_no_facts_returns_null_persona(self, tools_online):
        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(return_value=None)
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.is_new is True
        assert result.persona is None

    def test_persona_composition_handles_error(self, tools_online):
        db = tools_online.db
        asyncio.run(tools_online.session_start())
        f = Fact(key="trait",
            value="algo",
            category=FactCategory.PERSONALITY,
            confidence=0.9,
        )
        db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_personality = AsyncMock(side_effect=Exception("API down"))
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        result = asyncio.run(tools_online.session_start())
        assert result.persona == "algo"  # fallback al rasgo si falla el LLM


# ── message_append ───────────────────────────────────────────────


class TestMessageAppend:
    def test_append_user_message(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.message_append(r.session_id, "user", "Hola"))
        assert out.ok is True
        assert out.messages_remaining == SESSION_LIMIT - 1

    def test_append_assistant_message(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.message_append(r.session_id, "user", "Hola"))
        out = asyncio.run(tools.message_append(r.session_id, "assistant", "Respuesta"))
        assert out.ok is True

    def test_append_with_tool_calls_json_dict(self, tools):
        r = asyncio.run(tools.session_start())
        tc = {"name": "echo", "args": {"text": "hi"}}
        tr = [{"output": "hi"}]
        out = asyncio.run(tools.message_append(
            r.session_id, "assistant", "done",
            tool_calls=tc, tool_results=tr,
        ))
        assert out.ok is True
        msgs = tools.db.get_session_messages(r.session_id)
        assert len(msgs) == 1
        assert '"name"' in msgs[0].tool_calls

    def test_append_with_tool_calls_already_string(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.message_append(
            r.session_id, "assistant", "done",
            tool_calls='[{"name":"x"}]',
            tool_results='{"ok":true}',
        ))
        assert out.ok

    def test_append_to_closed_session_returns_action(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.session_end(r.session_id))
        out = asyncio.run(tools.message_append(r.session_id, "user", "late"))
        assert out.ok is False
        assert out.action == "new_session_required"
        assert "no activa" in (out.warning or "")

    def test_append_to_nonexistent_session_returns_action(self, tools):
        out = asyncio.run(tools.message_append("bad_id", "user", "hi"))
        assert out.ok is False
        assert out.action == "new_session_required"
        assert "no activa" in (out.warning or "")

    @pytest.mark.slow
    def test_warning_at_180_messages(self, tools):
        r = asyncio.run(tools.session_start())
        for i in range(SESSION_WARNING_AT - 1):
            asyncio.run(tools.message_append(r.session_id, "user", f"msg{i}"))
        out = asyncio.run(tools.message_append(r.session_id, "user", "final"))
        assert out.warning is not None
        assert "Límite" in out.warning

    @pytest.mark.slow
    def test_force_close_at_200_messages(self, tools):
        r = asyncio.run(tools.session_start())
        for i in range(SESSION_LIMIT - 1):
            asyncio.run(tools.message_append(r.session_id, "user", f"msg{i}"))
        out = asyncio.run(tools.message_append(r.session_id, "user", "last"))
        assert "Sesión cerrada" in out.warning
        assert out.messages_remaining == 0

        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.CLOSED


# ── session_end ──────────────────────────────────────────────────


class TestSessionEnd:
    def test_close_session(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.session_end(r.session_id))
        assert isinstance(out, SessionEndOutput)
        assert out.queued is True

        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.CLOSED
        assert s.ended_at is not None

    def test_truncated_session(self, tools):
        r = asyncio.run(tools.session_start())
        out = asyncio.run(tools.session_end(r.session_id, truncated=True))
        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.TRUNCATED

    def test_closed_session_pending_consolidation(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.session_end(r.session_id))
        pending = tools.db.get_pending_consolidations()
        assert len(pending) == 1


# ── memory_recall ────────────────────────────────────────────────


class TestMemoryRecall:
    def test_recall_with_vector_results(self, tools, mock_ollama):
        db = tools.db
        asyncio.run(tools.session_start())
        f = Fact(key="juegos",
            value="Víctor juega It Takes Two con Enrique",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid = db.insert_fact(f)
        db.record_fact_access(fid)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid, "rowid": fid, "distance": 0.1}],  # vec0_facts
                [],  # vec0_sessions
            ]
            result = asyncio.run(tools.memory_recall("juegos cooperativos"))

        assert len(result.facts) == 1
        assert result.facts[0]["key"] == "juegos"

    def test_recall_fallback_on_vector_error(self, tools, mock_ollama):
        mock_ollama.embed.side_effect = Exception("Ollama down")

        asyncio.run(tools.session_start())
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        tools.db.insert_fact(f)

        result = asyncio.run(tools.memory_recall("quién soy"))
        assert len(result.facts) > 0
        assert result.summary != ""
        assert result.degraded is True
        assert "vector_search_failed" in result.warnings
        assert "recall_degraded_recent_facts" in result.warnings

    def test_recall_no_results(self, tools, mock_ollama):
        with patch.object(tools, "_vector_search", return_value=[]):
            result = asyncio.run(tools.memory_recall("nada"))

        assert result.facts == []
        assert result.related_sessions == []

    def test_recall_no_candidates_warning(self, tools, mock_ollama):
        db = tools.db
        f_low = Fact(key="dubious",
            value="not sure",
            category=FactCategory.PERSONAL,
            confidence=0.5,
        )
        fid_low = db.insert_fact(f_low)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid_low, "rowid": fid_low, "distance": 0.1}],
                [],
            ]
            result = asyncio.run(tools.memory_recall("test"))

        assert result.facts == []
        assert "no_candidates_after_filter" in result.warnings
        assert result.degraded is True

    def test_recall_online_composes_summary(self, tools_online, mock_ollama):
        db = tools_online.db
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid = db.insert_fact(f)

        mock_recall = MagicMock()
        mock_recall.compose_summary = AsyncMock(return_value="Víctor juega con Enrique.")
        mock_recall.compose_personality = AsyncMock(return_value=None)
        mock_recall.close = AsyncMock()
        tools_online._recall = mock_recall

        with patch.object(tools_online, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [{"id": fid, "rowid": fid, "distance": 0.1}],
                [],
            ]
            result = asyncio.run(tools_online.memory_recall("juegos"))

        assert result.summary == "Víctor juega con Enrique."

    def test_recall_filters_low_confidence_facts(self, tools, mock_ollama):
        db = tools.db
        f_low = Fact(key="dubious",
            value="not sure",
            category=FactCategory.PERSONAL,
            confidence=0.5,
        )
        fid_low = db.insert_fact(f_low)
        f_high = Fact(key="sure",
            value="confirmed",
            category=FactCategory.PERSONAL,
            confidence=1.0,
        )
        fid_high = db.insert_fact(f_high)

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [
                    {"id": fid_low, "rowid": fid_low, "distance": 0.1},
                    {"id": fid_high, "rowid": fid_high, "distance": 0.2},
                ],
                [],
            ]
            result = asyncio.run(tools.memory_recall("test"))

        keys = {f["key"] for f in result.facts}
        assert "sure" in keys
        assert "dubious" not in keys

    def test_recall_hybrid_re_ranking(self, tools, mock_ollama):
        db = tools.db
        asyncio.run(tools.session_start())
        
        # 1. Fact A: Very close vector-wise (distance=0.01 -> similarity=0.99), low access (0), confidence=1.0
        # Score = 0.99 * 0.6 + 1.0 * 0.2 + 0.0 * 0.2 = 0.594 + 0.2 = 0.794
        f_a = Fact(key="fact_a",
            value="A very close match semantically",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=0,
        )
        fid_a = db.insert_fact(f_a)
        
        # 2. Fact B: Slightly further vector-wise (distance=0.4 -> similarity=0.714), high access (10 -> reinforcement=0.5), confidence=1.0
        # Score = 0.714 * 0.6 + 1.0 * 0.2 + 0.5 * 0.2 = 0.428 + 0.2 + 0.1 = 0.728
        f_b = Fact(key="fact_b",
            value="A slightly further match",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=10,
        )
        fid_b = db.insert_fact(f_b)

        # 3. Fact C: Furthest vector-wise (distance=0.6 -> similarity=0.625), extremely high access (20 -> reinforcement=1.0), confidence=1.0
        # Score = 0.625 * 0.6 + 1.0 * 0.2 + 1.0 * 0.2 = 0.375 + 0.2 + 0.2 = 0.775
        f_c = Fact(key="fact_c",
            value="The furthest match but very popular",
            category=FactCategory.PREFERENCE,
            confidence=1.0,
            access_count=20,
        )
        fid_c = db.insert_fact(f_c)

        # Update access counts directly in database since insert_fact ignores access_count
        db.conn.execute("UPDATE facts SET access_count = 10 WHERE id = ?", (fid_b,))
        db.conn.execute("UPDATE facts SET access_count = 20 WHERE id = ?", (fid_c,))
        db.conn.commit()

        with patch.object(tools, "_vector_search") as mock_vs:
            mock_vs.side_effect = [
                [
                    {"id": fid_a, "rowid": fid_a, "distance": 0.01},
                    {"id": fid_b, "rowid": fid_b, "distance": 0.4},
                    {"id": fid_c, "rowid": fid_c, "distance": 0.6},
                ],
                [],
            ]
            result = asyncio.run(tools.memory_recall("search query"))

        # The expected ranking order of keys based on scores:
        # 1. fact_a (0.794)
        # 2. fact_c (0.775)
        # 3. fact_b (0.728)
        assert len(result.facts) == 3
        assert result.facts[0]["key"] == "fact_a"
        assert result.facts[1]["key"] == "fact_c"
        assert result.facts[2]["key"] == "fact_b"


# ── memory_search ────────────────────────────────────────────────


class TestMemorySearch:
    def test_search_finds_content(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.message_append(r.session_id, "user", "palabra_clave"))
        asyncio.run(tools.message_append(r.session_id, "assistant", "respuesta"))

        result = asyncio.run(tools.memory_search("palabra_clave"))
        assert "results" in result
        assert len(result["results"]) >= 1
        assert result["results"][0]["content"] == "palabra_clave"

    def test_search_no_results(self, tools):
        result = asyncio.run(tools.memory_search("nonexistent_xyz"))
        assert "results" in result
        assert len(result["results"]) == 0




# ── memory_forget ────────────────────────────────────────────────


class TestMemoryContext:
    def test_context_empty_session(self, tools):
        r = asyncio.run(tools.session_start())
        result = asyncio.run(tools.memory_context(r.session_id))
        assert isinstance(result, MemoryContextOutput)
        assert result.total == 0
        assert result.messages == []
        assert result.offset == 0

    def test_context_with_messages(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.message_append(r.session_id, "user", "msg1"))
        asyncio.run(tools.message_append(r.session_id, "assistant", "msg2"))
        asyncio.run(tools.message_append(r.session_id, "user", "msg3"))

        result = asyncio.run(tools.memory_context(r.session_id))
        assert result.total == 3
        assert len(result.messages) == 3
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "msg1"

    def test_context_offset(self, tools):
        r = asyncio.run(tools.session_start())
        for i in range(5):
            asyncio.run(tools.message_append(r.session_id, "user", f"msg{i}"))

        result = asyncio.run(tools.memory_context(r.session_id, offset=2, limit=2))
        assert result.total == 5
        assert result.offset == 2
        assert len(result.messages) == 2
        assert result.messages[0]["content"] == "msg2"

    def test_context_limit_capped_at_30(self, tools):
        r = asyncio.run(tools.session_start())
        result = asyncio.run(tools.memory_context(r.session_id, limit=100))
        assert result.limit == 30

    def test_context_includes_session_summary(self, tools):
        r = asyncio.run(tools.session_start())
        asyncio.run(tools.message_append(r.session_id, "user", "hi"))
        tools.db.close_session(r.session_id)
        tools.db.set_session_summary(r.session_id, "Charla de prueba")

        result = asyncio.run(tools.memory_context(r.session_id))
        assert result.session_summary == "Charla de prueba"

    def test_context_nonexistent_session_raises(self, tools):
        with pytest.raises(ValueError, match="no encontrada"):
            asyncio.run(tools.memory_context("bad_id"))


# ── memory_forget ────────────────────────────────────────────────


class TestMemoryForget:
    def test_forget_by_fact_id(self, tools):
        f = Fact(key="temp",
            value="to forget",
            category=FactCategory.PERSONALITY,
        )
        fid = tools.db.insert_fact(f)
        result = asyncio.run(tools.memory_forget(fact_id=fid))
        assert result["deleted"] is True

        forgotten = tools.db.get_fact(fid)
        assert forgotten.confidence == 0

    def test_forget_by_key(self, tools):
        f = Fact(key="old_pref",
            value="vim",
            category=FactCategory.PREFERENCE,
        )
        tools.db.insert_fact(f)
        result = asyncio.run(tools.memory_forget(key="old_pref"))
        assert result["deleted"] is True

    def test_forget_nonexistent_raises(self, tools):
        with pytest.raises(ValueError):
            asyncio.run(tools.memory_forget(key="nope"))

    def test_forget_no_params_raises(self, tools):
        with pytest.raises(ValueError, match="Especifica"):
            asyncio.run(tools.memory_forget())


# ── memory_export ────────────────────────────────────────────────


class TestMemoryExport:
    def test_export_empty(self, tools):
        result = asyncio.run(tools.memory_export())
        assert result["facts"] == []
        assert result["sessions"] == []

    def test_export_with_data(self, tools):
        from alambique.tools import _insert_embedding

        r = asyncio.run(tools.session_start())
        asyncio.run(tools.message_append(r.session_id, "user", "hi"))
        f = Fact(key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
        )
        fid = tools.db.insert_fact(f)
        _insert_embedding(tools.db.conn, "vec0_facts", fid, [0.1] * 1024)

        tools.db.close_session(r.session_id)
        tools.db.set_session_summary(r.session_id, "Saludo inicial")

        result = asyncio.run(tools.memory_export())
        assert len(result["facts"]) == 1
        assert len(result["sessions"]) == 1
        assert result["facts"][0]["embedding_ok"] is True
        assert result["facts"][0]["created_at"] is not None
        assert result["sessions"][0]["created_at"] is not None


# ── memory_status ────────────────────────────────────────────────


class TestMemoryHealth:
    def test_health_all_ok(self, tools_online, mock_ollama):
        result = asyncio.run(tools_online.memory_health())
        assert result.healthy is True
        assert result.mode == "online"
        assert result.checks["ollama"].status == "ok"
        assert result.checks["api_key"].status == "ok"
        assert result.warnings == []

    def test_health_ollama_down(self, tools_online, mock_ollama):
        mock_ollama.health = AsyncMock(return_value=False)
        result = asyncio.run(tools_online.memory_health())
        assert result.healthy is False
        assert "ollama_unavailable" in result.warnings

    def test_health_offline_mode(self, tools, mock_ollama):
        result = asyncio.run(tools.memory_health())
        assert result.mode == "offline"
        assert "offline_mode" in result.warnings

    def test_health_orphan_embeddings(self, tools, mock_ollama):
        tools.db.insert_fact(
            Fact(key="orphan", value="sin vector", category=FactCategory.PERSONAL)
        )
        result = asyncio.run(tools.memory_health())
        assert result.checks["embeddings"].status == "warning"
        assert "embeddings_orphaned" in result.warnings


class TestMemoryStatus:
    def test_status_empty(self, tools):
        result = asyncio.run(tools.memory_status())
        assert result.sessions == 0
        assert result.facts == 0
        assert result.pending_consolidation == 0
        assert result.last_consolidation is None

    def test_status_with_data(self, tools):
        r1 = asyncio.run(tools.session_start())
        r2 = asyncio.run(tools.session_start())
        asyncio.run(tools.session_end(r1.session_id))  # close to count as open => count session

        f = Fact(
            key="nombre",
            value="Víctor",
            category=FactCategory.PERSONAL,
        )
        tools.db.insert_fact(f)

        result = asyncio.run(tools.memory_status())
        assert result.sessions == 2
        assert result.facts == 1
        assert result.pending_consolidation == 1


# ── Vector helpers ───────────────────────────────────────────────


class TestVectorHelpers:
    def test_session_id_rowid_roundtrip(self):
        from alambique.tools import _session_id_to_rowid, _rowid_to_session_id

        test_ids = [
            "sess_000000000000",
            "sess_ffffffffffff",
            "sess_a1b2c3d4e5f6",
            "sess_123456789abc",
        ]
        for sid in test_ids:
            assert _rowid_to_session_id(_session_id_to_rowid(sid)) == sid

    def test_insert_embedding_fact(self, db, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = db.conn
        _insert_embedding(conn, "vec0_facts", 1, [0.1] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = 1"
        ).fetchall()
        assert len(rows) == 1

    def test_insert_embedding_session(self, db, mock_ollama):
        from alambique.tools import _insert_embedding, _session_id_to_rowid

        conn = db.conn
        sid = "sess_abc123def456"
        rowid = _session_id_to_rowid(sid)
        _insert_embedding(conn, "vec0_sessions", sid, [0.1] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_sessions WHERE rowid = ?", (rowid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == rowid

    def test_update_embedding(self, db, mock_ollama):
        from alambique.tools import _insert_embedding, _update_embedding

        conn = db.conn
        _insert_embedding(conn, "vec0_facts", 42, [1.0] * 1024)
        _update_embedding(conn, "vec0_facts", 42, [0.5] * 1024)

        rows = conn.execute(
            "SELECT rowid FROM vec0_facts WHERE rowid = 42"
        ).fetchall()
        assert len(rows) == 1

    def test_upsert_embedding_inserts_then_updates(self, db, mock_ollama):
        from alambique.tools import _upsert_embedding

        conn = db.conn
        _upsert_embedding(conn, "vec0_facts", 7, [0.1] * 1024)
        _upsert_embedding(conn, "vec0_facts", 7, [0.9] * 1024)

        row = conn.execute(
            "SELECT embedding FROM vec0_facts WHERE rowid = 7"
        ).fetchone()
        assert parse_embedding_blob(row["embedding"])[0] == pytest.approx(0.9)

    def test_vector_search_facts(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = tools.db.conn
        f1 = Fact(key="a", value="v", category=FactCategory.PERSONAL)
        f2 = Fact(key="b", value="v", category=FactCategory.PERSONALITY)
        fid1 = tools.db.insert_fact(f1)
        fid2 = tools.db.insert_fact(f2)

        # Create same embedding for both
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_facts", fid1, emb)
        _insert_embedding(conn, "vec0_facts", fid2, emb)

        results = tools._vector_search("vec0_facts", emb, limit=10)
        assert len(results) == 2

    def test_vector_search_active_facts_only_excludes_forgotten(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        conn = tools.db.conn
        f = Fact(key="gone", value="v", category=FactCategory.PERSONAL)
        fid = tools.db.insert_fact(f)
        tools.db.forget_fact(fid)
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_facts", fid, emb)

        results = tools._vector_search(
            "vec0_facts",
            emb,
            limit=10,
            active_facts_only=True,
        )
        assert results == []

    def test_vector_search_sessions(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding, _session_id_to_rowid, _rowid_to_session_id

        conn = tools.db.conn
        r = asyncio.run(tools.session_start())
        sid = r.session_id
        emb = [0.1] * 1024
        _insert_embedding(conn, "vec0_sessions", sid, emb)

        results = tools._vector_search("vec0_sessions", emb, limit=5)
        assert len(results) == 1
        assert results[0]["session_id"] == sid



# ── consolidation fact retrieval ───────────────────────────────────


class TestConsolidationFactRetrieval:
    def test_consolidation_search_text(self):
        msgs = [
            Message(session_id="s", role="user", content="Uso CachyOS"),
            Message(session_id="s", role="assistant", content="Interesante distro"),
        ]
        text = consolidation_search_text(msgs)
        assert "user: Uso CachyOS" in text
        assert "assistant: Interesante distro" in text

    def test_consolidation_search_text_truncates(self):
        msgs = [Message(session_id="s", role="user", content="x" * 9000)]
        text = consolidation_search_text(msgs, max_chars=100)
        assert len(text) == 100
        assert text.endswith("x")

    def test_facts_for_consolidation_prefers_semantic_match(self, tools, mock_ollama):
        from alambique.tools import _insert_embedding

        target = tools.db.insert_fact(
            Fact(
                key="keyboard",
                value="Teclado mecánico Keychron",
                category=FactCategory.POSSESSIONS,
                confidence=1.0,
            )
        )
        noise = tools.db.insert_fact(
            Fact(
                key="favorite_color",
                value="Azul",
                category=FactCategory.PREFERENCE,
                confidence=1.0,
            )
        )
        emb = [0.2] * 1024
        _insert_embedding(tools.db.conn, "vec0_facts", target, emb)
        _insert_embedding(tools.db.conn, "vec0_facts", noise, [0.9] * 1024)

        mock_ollama.health = AsyncMock(return_value=True)
        mock_ollama.embed = AsyncMock(return_value=emb)

        msgs = [
            Message(session_id="s", role="user", content="Mi teclado mecánico Keychron es genial"),
        ]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert warnings == []
        assert any(f.id == target for f in facts)
        assert facts[0].id == target

    def test_facts_for_consolidation_includes_personality(self, tools, mock_ollama):
        trait_id = tools.db.insert_fact(
            Fact(
                key="sarcastic",
                value="Es sarcástica",
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        mock_ollama.health = AsyncMock(return_value=False)

        msgs = [Message(session_id="s", role="user", content="Hola")]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert "consolidation_facts_offline_fallback" in warnings
        assert any(f.id == trait_id for f in facts)

    def test_facts_for_consolidation_vector_fallback(self, tools, mock_ollama):
        mock_ollama.health = AsyncMock(return_value=True)
        mock_ollama.embed = AsyncMock(side_effect=RuntimeError("embed down"))

        tools.db.insert_fact(
            Fact(key="fallback", value="dato", category=FactCategory.PERSONAL, confidence=1.0)
        )
        msgs = [Message(session_id="s", role="user", content="Algo")]
        facts, warnings = asyncio.run(tools._facts_for_consolidation(msgs))

        assert "consolidation_facts_vector_fallback" in warnings
        assert any(f.key == "fallback" for f in facts)


# ── consolidation embeddings ───────────────────────────────────────


class TestConsolidationEmbeddings:
    def test_update_refreshes_existing_fact_embedding(self, tools, mock_ollama):
        fid = tools.db.insert_fact(
            Fact(key="os", value="Linux", category=FactCategory.PREFERENCE, confidence=1.0)
        )
        from alambique.tools import _insert_embedding

        _insert_embedding(tools.db.conn, "vec0_facts", fid, [0.1] * 1024)

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        new_emb = [0.9] * 1024
        mock_ollama.embed_batch = AsyncMock(return_value=[new_emb])
        mock_ollama.embed = AsyncMock(return_value=[0.2] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.UPDATE,
                    key="os",
                    value="CachyOS",
                    category=FactCategory.PREFERENCE,
                    confidence=1.0,
                    related_fact_id=fid,
                    reason="SO actualizado",
                )
            ],
            session_summary="Cambio de sistema operativo",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        updated = tools.db.get_fact(fid)
        assert updated.value == "CachyOS"
        stored = tools.db.get_fact_embedding(fid)
        assert stored[0] == pytest.approx(0.9)

    def test_create_with_existing_key_updates_in_place(self, tools, mock_ollama):
        fid = tools.db.insert_fact(
            Fact(key="city", value="Madrid", category=FactCategory.PERSONAL, confidence=1.0)
        )

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.4] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.1] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.CREATE,
                    key="city",
                    value="Barcelona",
                    category=FactCategory.PERSONAL,
                    confidence=1.0,
                    reason="Corrección de ciudad",
                )
            ],
            session_summary="Ciudad actualizada",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        active = tools.db.get_fact_by_key("city")
        assert active.id == fid
        assert active.value == "Barcelona"

    def test_contradict_allocates_alternate_key(self, tools, mock_ollama):
        existing_id = tools.db.insert_fact(
            Fact(key="mood", value="cansado", category=FactCategory.STATE, confidence=1.0, ttl=86400)
        )

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.6] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.1] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.CONTRADICT,
                    key="mood",
                    value="recuperado",
                    category=FactCategory.STATE,
                    confidence=1.0,
                    ttl=86400,
                    related_fact_id=existing_id,
                    reason="Estado resuelto",
                )
            ],
            session_summary="Ánimo mejorado",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        assert tools.db.get_fact_by_key("mood").id == existing_id
        alt = tools.db.get_fact_by_key("mood__alt")
        assert alt is not None
        assert alt.value == "recuperado"

    def test_batch_create_and_update_embeddings(self, tools, mock_ollama):
        existing_id = tools.db.insert_fact(
            Fact(key="gpu", value="RTX 3080", category=FactCategory.POSSESSIONS, confidence=1.0)
        )
        from alambique.tools import _insert_embedding

        _insert_embedding(tools.db.conn, "vec0_facts", existing_id, [0.1] * 1024)

        session = tools.db.create_session()
        tools.db.close_session(session.id, SessionStatus.CLOSED)

        mock_ollama.embed_batch = AsyncMock(return_value=[[0.5] * 1024, [0.8] * 1024])
        mock_ollama.embed = AsyncMock(return_value=[0.3] * 1024)

        response = ConsolidationResponse(
            facts=[
                ConsolidationFactItem(
                    action=ConsolidationAction.UPDATE,
                    key="gpu",
                    value="RTX 4080",
                    category=FactCategory.POSSESSIONS,
                    confidence=1.0,
                    related_fact_id=existing_id,
                    reason="GPU actualizada",
                ),
                ConsolidationFactItem(
                    action=ConsolidationAction.CREATE,
                    key="keyboard",
                    value="Teclado mecánico",
                    category=FactCategory.POSSESSIONS,
                    confidence=1.0,
                    reason="Nueva posesión",
                ),
            ],
            session_summary="Hardware actualizado",
        )

        asyncio.run(tools._apply_consolidation(session, response))

        assert tools.db.get_fact(existing_id).value == "RTX 4080"
        assert tools.db.get_fact_embedding(existing_id)[0] == pytest.approx(0.5)

        new_fact = tools.db.get_fact_by_key("keyboard")
        assert new_fact is not None
        assert tools.db.get_fact_embedding(new_fact.id)[0] == pytest.approx(0.8)


# ── Background tasks / lifecycle ─────────────────────────────────


class TestDbGuard:
    def test_db_guard_serializes_concurrent_access(self, tools):
        order: list[str] = []

        async def job(label: str) -> None:
            async with tools._db_guard():
                order.append(f"start-{label}")
                await asyncio.sleep(0.02)
                order.append(f"end-{label}")

        async def run() -> None:
            await asyncio.gather(job("a"), job("b"))

        asyncio.run(run())

        assert order.index("end-a") < order.index("start-b") or order.index("end-b") < order.index("start-a")

    def test_concurrent_message_appends(self, tools):
        async def run() -> None:
            r = await tools.session_start()
            await asyncio.gather(
                *[
                    tools.message_append(r.session_id, "user", f"msg-{i}")
                    for i in range(25)
                ]
            )
            assert tools.db.session_message_count(r.session_id) == 25

        asyncio.run(run())


class TestBackgroundTasks:
    def test_consolidation_loop_idle(self, tools):
        asyncio.run(tools.start_background_tasks())
        assert tools._consolidator_task is not None
        assert tools._watchdog_task is not None
        asyncio.run(tools.stop_background_tasks())

    def test_watchdog_detects_stale(self, tools, mock_ollama):
        """Mark a session as stale (>30 min old, still open)."""
        asyncio.run(tools.session_start())
        stale = tools.db.find_stale_sessions(timeout_minutes=-1)
        assert len(stale) >= 1

    def test_online_property_false_without_key(self, tools):
        assert tools.online is False

    def test_online_property_true_with_key(self, tools_online):
        assert tools_online.online is True

    def test_consolidator_raises_without_key(self, tools):
        with pytest.raises(RuntimeError, match="API key"):
            _ = tools.consolidator

    def test_recall_raises_without_key(self, tools):
        with pytest.raises(RuntimeError, match="API key"):
            _ = tools.recall


# ── messages_for_consolidation ────────────────────────────────────


class TestMessagesForConsolidation:
    def test_keeps_user_and_assistant(self):
        msgs = [
            Message(session_id="s1", role="user", content="Hola"),
            Message(session_id="s1", role="assistant", content="Qué tal"),
        ]
        assert len(messages_for_consolidation(msgs)) == 2

    def test_drops_auto_commentary(self):
        msgs = [
            Message(session_id="s1", role="assistant", content="[Auto] Un goblin corre"),
            Message(session_id="s1", role="user", content="[Tú] ¿qué ves?"),
            Message(session_id="s1", role="assistant", content="[Aria] Una cueva"),
        ]
        filtered = messages_for_consolidation(msgs)
        assert len(filtered) == 2
        assert filtered[0].content.startswith("[Tú]")
        assert filtered[1].content.startswith("[Aria]")

    def test_drops_system_and_tool_roles(self):
        msgs = [
            Message(session_id="s1", role="system", content="ignore"),
            Message(session_id="s1", role="user", content="ok"),
        ]
        assert len(messages_for_consolidation(msgs)) == 1


# ── Observer pattern: session closure enforces correct state ─────


class TestSessionLifecycle:
    def test_full_lifecycle(self, tools):
        r = asyncio.run(tools.session_start())
        assert tools.db.get_session(r.session_id).status == SessionStatus.OPEN

        asyncio.run(tools.message_append(r.session_id, "user", "Hola"))
        asyncio.run(tools.message_append(r.session_id, "assistant", "Respuesta"))
        asyncio.run(tools.message_append(r.session_id, "user", "Gracias"))

        msgs = tools.db.get_session_messages(r.session_id)
        assert len(msgs) == 3

        asyncio.run(tools.session_end(r.session_id))
        s = tools.db.get_session(r.session_id)
        assert s.status == SessionStatus.CLOSED
        assert s.ended_at is not None
