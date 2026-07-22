"""Pydantic models for Alambique data structures."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────


class SessionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    TRUNCATED = "truncated"


class ConsolidationAction(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    MERGE = "merge"
    CONTRADICT = "contradict"
    DISCARD = "discard"


_utc = timezone.utc


# ── Domain models ─────────────────────────────────────────────────


class Session(BaseModel):
    id: str
    status: SessionStatus = SessionStatus.OPEN
    consolidated: bool = False
    summary: Optional[str] = None
    client: Optional[str] = None
    conversation_id: Optional[str] = None
    expression: Optional[str] = None
    mood_text: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(_utc))
    ended_at: Optional[datetime] = None


class Message(BaseModel):
    id: Optional[int] = None
    session_id: str
    role: str
    content: str
    tool_calls: Optional[str] = None
    tool_results: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(_utc))


class Consolidation(BaseModel):
    id: Optional[int] = None
    session_id: str
    action: ConsolidationAction
    thread_id: Optional[int] = None
    capsule_scope: Optional[str] = None
    echo_id: Optional[int] = None
    previous_value: Optional[str] = None
    new_value: Optional[str] = None
    reason: Optional[str] = None


# ── MCP tool input/output models ───────────────────────────────────


class SessionStartOutput(BaseModel):
    session_id: Optional[str] = None
    status: str  # "ok"
    persona: Optional[str] = None
    client: Optional[str] = None
    conversation_id: Optional[str] = None
    session_reused: bool = False
    is_new: bool = False
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    initial_context: Optional[str] = None
    active_thread_keys: list[str] = Field(default_factory=list)


class SessionEndOutput(BaseModel):
    queued: bool = False
    pending_consolidation: int = 0


class MemoryRecallOutput(BaseModel):
    summary: str
    related_sessions: list[dict[str, Any]] = Field(default_factory=list)
    related_threads: list[dict[str, Any]] = Field(default_factory=list)
    related_capsules: list[dict[str, Any]] = Field(default_factory=list)
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)


class MemorySearchOutput(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list)


class MemoryRebuildVectorsOutput(BaseModel):
    dry_run: bool = True
    sessions_with_summary: int = 0
    vec0_sessions_before: int = 0
    would_remove_orphan_sessions: Optional[int] = None
    orphan_sessions_removed: int = 0
    vec0_sessions_cleared: int = 0
    sessions_embedded: int = 0
    sessions_failed: int = 0
    vec0_sessions_after: Optional[int] = None
    session_orphans_after: Optional[int] = None
    sessions_missing_after: Optional[int] = None
    threads_embedded: int = 0
    capsules_embedded: int = 0
    echoes_embedded: int = 0
    warnings: list[str] = Field(default_factory=list)


class MemoryStatusOutput(BaseModel):
    sessions: int = 0
    threads: int = 0
    capsules: int = 0
    echoes: int = 0
    pending_consolidation: int = 0
    last_consolidation: Optional[datetime] = None


class MemoryHealthCheck(BaseModel):
    status: str  # "ok" | "warning" | "error"
    detail: Optional[str] = None


class MemoryHealthOutput(BaseModel):
    healthy: bool
    mode: str  # "online" | "offline"
    checks: dict[str, MemoryHealthCheck] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ApiKeyState(BaseModel):
    status: str  # "loaded" | "missing" | "waiting_pass" | "failed"
    source: Optional[str] = None
    detail: Optional[str] = None
    last_attempt: Optional[datetime] = None
    attempt_count: int = 0
    retry_interval_seconds: int = 30
    pass_timeout_seconds: int = 120


class ActiveSessionState(BaseModel):
    id: Optional[str] = None
    client: Optional[str] = None
    conversation_id: Optional[str] = None
    expression: str = "normal"
    mood_text: str = ""
    bound: bool = False


class DaemonStatusOutput(BaseModel):
    version: str
    started_at: datetime
    uptime_seconds: float
    overall: str  # "ok" | "degraded" | "down"
    status_label: str
    status_summary: str
    system_message: str = ""
    system_message_level: str = "info"  # "ok" | "warning" | "error" | "info"
    mode: str  # "online" | "offline"
    online: bool = True
    db_exists: bool = True
    healthy: bool
    checks: dict[str, MemoryHealthCheck] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    api_key: ApiKeyState
    stats: MemoryStatusOutput
    active_session: ActiveSessionState
    open_sessions: list[ActiveSessionState] = Field(default_factory=list)
    sessions: int = 0
    threads: int = 0
    capsules: int = 0
    echoes: int = 0
    expression: str = "normal"
    mood_text: str = ""
    last_consolidation: Optional[datetime] = None


class MemoryContextOutput(BaseModel):
    session_summary: Optional[str] = None
    client: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 0


# ── Consolidator sub-models ────────────────────────────────────────


class ConsolidationResponse(BaseModel):
    threads: list[dict] = Field(default_factory=list)
    relationship_capsules: list[dict] = Field(default_factory=list)
    echoes: list[dict] = Field(default_factory=list)
    # Optional single future-oriented initiative for Lucy (MVP).
    # null / omitted when nothing genuine to propose.
    lucy_initiative: Optional[dict] = None
    model_config = {"extra": "ignore"}  # ignore legacy fields if any LLM still emits them
