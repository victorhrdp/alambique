"""Tests for server.py — tool dispatch and MCP definitions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alambique.server import TOOL_DEFINITIONS, _register_handlers
from alambique.models import SessionStartOutput, MessageAppendOutput, MemoryRecallOutput
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
    def test_thirteen_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 13

    def test_all_tool_names(self):
        names = {t.name for t in TOOL_DEFINITIONS}
        expected = {
            "session_start", "message_append", "session_end",
            "memory_recall", "memory_search", "memory_forget",
            "memory_export", "memory_status", "memory_health",
            "memory_reembed", "memory_deduplicate",
            "memory_context", "session_list",
        }
        assert names == expected

    def test_each_tool_has_input_schema(self):
        for t in TOOL_DEFINITIONS:
            assert t.inputSchema is not None, f"{t.name} missing inputSchema"

    def test_session_start_persona_seed_optional(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "session_start")
        assert "persona_seed" in t.inputSchema["properties"]
        assert "persona_seed" not in t.inputSchema.get("required", [])

    def test_message_append_required_fields(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "message_append")
        assert set(t.inputSchema["required"]) == {"session_id", "role", "content"}

    def test_memory_recall_query_required(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "memory_recall")
        assert t.inputSchema["required"] == ["query"]

    def test_memory_search_query_only(self):
        t = next(t for t in TOOL_DEFINITIONS if t.name == "memory_search")
        assert t.inputSchema["required"] == ["query"]
        assert "agent" not in t.inputSchema["properties"]


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

        assert len(result) == 13


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

        tools.session_start.assert_called_once_with(None)
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

        tools.session_start.assert_called_once_with("Eres Lucy.")
        data = json.loads(result[0].text)
        assert data["status"] == "ok"

    def test_message_append_dispatched(self, dispatch):
        call_tool, tools = dispatch
        tools.message_append = AsyncMock(
            return_value=MessageAppendOutput(messages_remaining=199)
        )

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("message_append", {
                "session_id": "sess_abc",
                "role": "user",
                "content": "Hola",
            })
        )
        loop.close()

        tools.message_append.assert_called_once_with(
            session_id="sess_abc",
            role="user",
            content="Hola",
            tool_calls=None,
            tool_results=None,
        )

    def test_unknown_tool(self, dispatch):
        call_tool, tools = dispatch

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(call_tool("bogus_tool", {}))
        loop.close()

        assert "Tool desconocida" in result[0].text

    def test_value_error_handled(self, dispatch):
        call_tool, tools = dispatch
        tools.message_append = AsyncMock(
            side_effect=ValueError("Sesión no activa")
        )

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("message_append", {
                "session_id": "bad",
                "role": "user",
                "content": "hi",
            })
        )
        loop.close()

        data = json.loads(result[0].text)
        assert "error" in data
        assert "Sesión no activa" in data["error"]

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
            facts=[{"id": 1, "key": "k", "value": "v", "category": "personal", "confidence": 1.0}],
            related_sessions=[],
        ))

        loop = __import__("asyncio").new_event_loop()
        result = loop.run_until_complete(
            call_tool("memory_recall", {"query": "test"})
        )
        loop.close()

        data = json.loads(result[0].text)
        assert data["summary"] == "resumen"
        assert len(data["facts"]) == 1

    def test_memory_search_dispatched(self, dispatch):
        call_tool, tools = dispatch
        tools.memory_search = AsyncMock(return_value={"results": []})

        loop = __import__("asyncio").new_event_loop()
        loop.run_until_complete(
            call_tool("memory_search", {"query": "test"})
        )
        loop.close()

        tools.memory_search.assert_called_once_with("test")


class TestImportStructure:
    def test_version_importable(self):
        from alambique import __version__
        assert isinstance(__version__, str)

    def test_main_function(self):
        from alambique.server import main
        assert callable(main)
