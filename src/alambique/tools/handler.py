"""Composed ToolHandler — MCP tool entry point."""

from __future__ import annotations

from alambique.tools.base import ToolHandlerBase
from alambique.tools.consolidation import ConsolidationMixin
from alambique.tools.memory import MemoryMixin
from alambique.tools.sessions import SessionMixin
from alambique.tools.status import StatusMixin
from alambique.tools.workers import WorkerMixin


class ToolHandler(
    SessionMixin,
    MemoryMixin,
    ConsolidationMixin,
    StatusMixin,
    WorkerMixin,
    ToolHandlerBase,
):
    """Handles all MCP tool calls."""