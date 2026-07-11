"""Pydantic models for Alambique data structures."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from alambique.memory_config import (
    FLOOR_POSSESSIONS,
    FLOOR_PREFERENCE,
    LAMBDA_POSSESSIONS,
    LAMBDA_PREFERENCE,
)


# ── Enums ──────────────────────────────────────────────────────────


class SessionStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    TRUNCATED = "truncated"


class FactCategory(str, Enum):
    PERSONALITY = "personality"
    STATE = "state"
    PERSONAL = "personal"
    PREFERENCE = "preference"
    POSSESSIONS = "possessions"


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


class Fact(BaseModel):
    id: Optional[int] = None
    key: str
    value: str
    category: FactCategory
    ttl: Optional[int] = None
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(_utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(_utc))

    def get_decayed_confidence(self) -> float:
        import math
        # 1. Immune categories: stable facts, agent personality, temporal states (TTL-only)
        if self.category in (
            FactCategory.PERSONAL,
            FactCategory.PERSONALITY,
            FactCategory.STATE,
        ):
            return self.confidence

        # 2. Base decay rates (decay constant lambda)
        if self.category == FactCategory.POSSESSIONS:
            lambda_base = LAMBDA_POSSESSIONS
        else:
            lambda_base = LAMBDA_PREFERENCE

        # 3. Spaced reinforcement: more access count decreases rate of forgetting
        adjusted_lambda = lambda_base / (1.0 + math.log(self.access_count + 1))

        # 4. Calculate elapsed time in seconds
        # Handle timezone-aware vs naive created_at
        created = self.created_at
        if created.tzinfo is not None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()
        
        elapsed_seconds = (now - created).total_seconds()
        if elapsed_seconds < 0:
            elapsed_seconds = 0

        # 5. Exponential decay formula
        decayed_value = self.confidence * math.exp(-adjusted_lambda * elapsed_seconds)

        # 6. Apply floor to prevent complete deletion from database
        floor = (
            FLOOR_PREFERENCE
            if self.category == FactCategory.PREFERENCE
            else FLOOR_POSSESSIONS
        )
        return round(max(decayed_value, floor), 4)

    def is_ttl_expired(self) -> bool:
        """True if the fact's TTL has elapsed (state facts expire via TTL only)."""
        if self.ttl is None:
            return False
        created = self.created_at
        if created.tzinfo is not None:
            now = datetime.now(timezone.utc)
        else:
            now = datetime.now()
        elapsed = (now - created).total_seconds()
        return elapsed >= self.ttl

    def is_active(self) -> bool:
        """Visible for recall: positive confidence and TTL not expired."""
        return self.confidence > 0 and not self.is_ttl_expired()


class Consolidation(BaseModel):
    id: Optional[int] = None
    session_id: str
    action: ConsolidationAction
    fact_id: Optional[int] = None
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


class SessionEndOutput(BaseModel):
    queued: bool = True


class MemoryRecallOutput(BaseModel):
    summary: str
    facts: list[dict[str, Any]] = Field(default_factory=list)
    related_sessions: list[dict[str, Any]] = Field(default_factory=list)
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)


class MemorySearchOutput(BaseModel):
    results: list[dict[str, Any]] = Field(default_factory=list)


class MemoryForgetOutput(BaseModel):
    deleted: bool = True


class MemoryCleanupOutput(BaseModel):
    stale_embeddings_removed: int = 0


class MemoryReembedOutput(BaseModel):
    dry_run: bool = False
    missing_before: int = 0
    embedded: int = 0
    failed: int = 0
    fact_ids: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemoryDeduplicateOutput(BaseModel):
    dry_run: bool = True
    pairs_found: int = 0
    merged: int = 0
    pairs: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MemoryExportOutput(BaseModel):
    facts: list[dict[str, Any]] = Field(default_factory=list)
    sessions: list[dict[str, Any]] = Field(default_factory=list)


class MemoryStatusOutput(BaseModel):
    sessions: int = 0
    facts: int = 0
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


class MemoryContextOutput(BaseModel):
    session_summary: Optional[str] = None
    client: Optional[str] = None
    conversation_id: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 0


# ── Consolidator sub-models ────────────────────────────────────────


class ConsolidationFactItem(BaseModel):
    action: ConsolidationAction
    key: str
    value: str
    category: FactCategory
    confidence: float = 1.0
    ttl: Optional[int] = None
    related_fact_id: Optional[int] = None
    reason: str


class ConsolidationResponse(BaseModel):
    facts: list[ConsolidationFactItem] = Field(default_factory=list)
    session_summary: str
