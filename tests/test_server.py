"""Tests for server.py — tool dispatch and MCP definitions."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.server import Server

from alambique.database import Database
from alambique.models import SessionStartOutput, MemoryRecallOutput
from alambique.server import TOOL_DEFINITIONS, _register_handlers, build_sse_app, run
from alambique.tools import ToolHandler


def _make_fake_server():
    """Create a fake MCP Server that captures the handler registrations."""
    handlers = {}

    class FakeServer:
        def list_tools(self):
            def decorator(fn):
                handlers["list_tools"] = fn
                return fn
            return decorator

        def call_tool(self):
            def decorator(fn):
                handlers["call_tool"] = fn
                return fn
            return decorator

    return FakeServer(), handlers


class TestToolDefinitions:
    def test_tool_count(self):
        assert len(TOOL_DEFINITIONS) == 12

    def test_all_tool_names(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        expected = {
            "session_start", "session_end", "session_update",
            "memory_recall", "memory_search",
            "memory_status", "memory_health",
            "memory_rebuild_vectors",
            "memory_context", "session_list",
            "memory_expand_thread", "close_session",
        }
        assert names == expected

    def test_each_tool_has_input_schema(self):
        for t in TOOL_DEFINITIONS:
            assert t.inputSchema is not None, f"{t.name} missing inputSchema"

    def test_session_start_persona_seed_optional(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "session_start")
        assert "persona_seed" in t.inputSchema["properties"]
        assert "client" in t.inputSchema["properties"]
        assert "workspace" in t.inputSchema["properties"]
        assert "persona_seed" not in t.inputSchema.get("required", [])

    def test_memory_recall_query_required(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "memory_recall")
        assert t.inputSchema["required"] == ["query"]

    def test_memory_search_query_only(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "memory_search")
        assert t.inputSchema["required"] == ["query"]
        assert "agent" not in t.inputSchema["properties"]

    def test_session_update_required_fields(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "session_update")
        assert t.inputSchema["required"] == ["session_id", "expression", "mood_text"]


class TestRegisterHandlers:
    def test_handlers_registered(self):
        server, handlers = _make_fake_server()
        tools = MagicMock(spec=ToolHandler)
        _register_handlers(server, tools)

        assert "list_tools" in handlers
        assert "call_tool" in handlers

    def test_list_tools_returns_definitions(self):
        server, handlers = _make_fake_server()
        tools = MagicMock(spec=ToolHandler)
        _register_handlers(server, tools)

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(handlers["list_tools"](None))
        loop.close()

        assert len(result) == 12


class TestToolDispatch:
    @pytest.fixture
    def dispatch(self):
        server, handlers = _make_fake_server()
        tools = MagicMock(spec=ToolHandler)
        _register_handlers(server, tools)
        return handlers["call_tool"], tools

    def test_session_start_dispatched(self, dispatch):
        call_tool, tools = dispatch
        tools.session_start = AsyncMock(return_value=SessionStartOutput(status="ok"))

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("session_start", {})
        )
        loop.close()

        tools.session_start.assert_called_once_with(
            persona_seed=None,
            client=None,
            conversation_id=None,
            workspace=None,
        )
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "ok"

    def test_session_start_with_persona_seed(self, dispatch):
        call_tool, tools = dispatch
        tools.session_start = AsyncMock(
            return_value=SessionStartOutput(status="ok", session_id="sess_abc")
        )

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("session_start", {"persona_seed": "Eres Lucy."})
        )
        loop.close()

        tools.session_start.assert_called_once_with(
            persona_seed="Eres Lucy.",
            client=None,
            conversation_id=None,
            workspace=None,
        )
        data = json.loads(result[0].text)
        assert data["status"] == "ok"

    def test_unknown_tool(self, dispatch):
        call_tool, tools = dispatch

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(call_tool("bogus_tool", {}))
        loop.close()

        assert "Tool desconocida" in result[0].text

    def test_value_error_handled(self, dispatch):
        call_tool, tools = dispatch
        tools.memory_recall = AsyncMock(
            side_effect=ValueError("Invalid query")
        )

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("memory_recall", {
                "query": "hi",
            })
        )
        loop.close()

        data = json.loads(result[0].text)
        assert "error" in data
        assert "Invalid query" in data["error"]

    def test_generic_exception_handled(self, dispatch):
        call_tool, tools = dispatch
        tools.memory_status = AsyncMock(
            side_effect=Exception("Boom!")
        )

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(call_tool("memory_status", {}))
        loop.close()

        data = json.loads(result[0].text)
        assert "error" in data
        assert "Boom!" in data["error"]

    def test_result_serialized_with_model_dump(self, dispatch):
        call_tool, tools = dispatch
        tools.memory_recall = AsyncMock(return_value=MemoryRecallOutput(
            summary="resumen",
            related_sessions=[],
            related_threads=[],
            related_capsules=[],
        ))

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("memory_recall", {"query": "test"})
        )
        loop.close()

        data = json.loads(result[0].text)
        assert data["summary"] == "resumen"
        assert "facts" not in data  # legacy facts field removed from recall output

    def test_memory_search_dispatched(self, dispatch):
        call_tool, tools = dispatch
        tools.memory_search = AsyncMock(return_value={"results": []})

        loop = __import__("asyncio").new_event_loop()
        loop.run_until_complete(
            call_tool("memory_search", {"query": "test"})
        )
        loop.close()

        tools.memory_search.assert_called_once_with("test")

    def test_session_update_dispatched(self, dispatch):
        call_tool, tools = dispatch
        tools.session_update = AsyncMock(return_value={"status": "ok"})

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool(
                "session_update",
                {
                    "session_id": "sess_abc",
                    "expression": "thinking",
                    "mood_text": "revisando código",
                },
            )
        )
        loop.close()

        tools.session_update.assert_called_once_with(
            session_id="sess_abc",
            expression="thinking",
            mood_text="revisando código",
        )
        data = json.loads(result[0].text)
        assert data["status"] == "ok"


class TestImportStructure:
    def test_version_importable(self):
        from alambique import __version__
        assert isinstance(__version__, str)

    def test_main_function(self):
        from alambique.server import main
        assert callable(main)


class TestSseApp:
    def test_build_sse_app_status_endpoint(self, tmp_path):
        db = Database(tmp_path / "sse_smoke.db")
        db.connect()
        try:
            mock_ollama = MagicMock()
            mock_ollama.health = AsyncMock(return_value=False)
            mock_ollama.ensure_model = AsyncMock()
            mock_ollama.close = AsyncMock()
            tools = ToolHandler(db, mock_ollama, api_key=None)

            mcp_server = Server("alambique")
            _register_handlers(mcp_server, tools)
            app, _ = build_sse_app(mcp_server, tools, port=19042)

            async def _request():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://test",
                ) as client:
                    status_resp = await client.get("/status")
                    oauth_resp = await client.get(
                        "/.well-known/oauth-protected-resource"
                    )

                assert status_resp.status_code == 200
                payload = status_resp.json()
                assert "version" in payload
                assert "healthy" in payload
                assert "active_session" in payload

                assert oauth_resp.status_code == 200
                assert "resource" in oauth_resp.json()

            asyncio.run(_request())
        finally:
            db.close()

    def test_run_always_shuts_down(self, tmp_path):
        mock_db = MagicMock()
        mock_db.connect = MagicMock()
        mock_db.close = MagicMock()

        mock_ollama = MagicMock()
        mock_ollama.health = AsyncMock(return_value=False)
        mock_ollama.ensure_model = AsyncMock()
        mock_ollama.close = AsyncMock()

        mock_tools = MagicMock()
        mock_tools.note_api_key_attempt = MagicMock(return_value=False)
        mock_tools.online = False
        mock_tools._api_key_last_error = None
        mock_tools.start_background_tasks = AsyncMock()
        mock_tools.stop_background_tasks = AsyncMock()
        mock_tools.shutdown_open_sessions = AsyncMock()

        with (
            patch("alambique.server.Database", return_value=mock_db),
            patch("alambique.server.OllamaClient", return_value=mock_ollama),
            patch("alambique.server.ToolHandler", return_value=mock_tools),
            patch("alambique.server.fetch_api_key", return_value=MagicMock(key=None)),
            patch("alambique.server._serve_sse", new_callable=AsyncMock) as mock_serve,
            patch("alambique.server._shutdown", new_callable=AsyncMock) as mock_shutdown,
        ):
            asyncio.run(run(transport="sse", port=19042))

        mock_serve.assert_awaited_once()
        mock_shutdown.assert_awaited_once()
        mock_tools.start_background_tasks.assert_awaited_once()
