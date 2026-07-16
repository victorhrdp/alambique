"""MCP tool implementations for Alambique."""

from alambique.tools.base import (
    API_KEY_NOTIFY_COOLDOWN_SECONDS,
    API_KEY_RETRY_INTERVAL_SECONDS,
    DEFAULT_STATUS_PORT,
)
from alambique.tools.handler import ToolHandler
from alambique.tools.text import consolidation_search_text, messages_for_consolidation
from alambique.vector_store import (
    embedding_rowid,
    has_embedding,
    insert_embedding,
    rowid_to_session_id,
    session_id_to_rowid,
    update_embedding,
    upsert_embedding,
)

# Backward-compatible aliases for tests and scripts
_insert_embedding = insert_embedding
_update_embedding = update_embedding
_upsert_embedding = upsert_embedding
_session_id_to_rowid = session_id_to_rowid
_rowid_to_session_id = rowid_to_session_id
_embedding_rowid = embedding_rowid
_has_embedding = has_embedding

__all__ = [
    "ToolHandler",
    "consolidation_search_text",
    "messages_for_consolidation",
    "API_KEY_RETRY_INTERVAL_SECONDS",
    "API_KEY_NOTIFY_COOLDOWN_SECONDS",
    "DEFAULT_STATUS_PORT",
    # Legacy private names re-exported for tests
    "_insert_embedding",
    "_update_embedding",
    "_upsert_embedding",
    "_session_id_to_rowid",
    "_rowid_to_session_id",
    "_embedding_rowid",
    "_has_embedding",
]