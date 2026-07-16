"""Core ToolHandler state: DB lock, API key, LLM clients."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, AsyncIterator, Optional

from alambique.consolidator import (
    ApiKeyFetchResult,
    ConsolidatorClient,
    _pass_show_timeout,
    fetch_api_key,
)
from alambique.database import Database
from alambique.models import ApiKeyState
from alambique.notifications import (
    PASS_WAIT_BODY,
    PASS_WAIT_TITLE,
    is_waiting_pass_error,
    send_desktop_notification,
)
from alambique.ollama_client import OllamaClient
from alambique.recall import RecallClient
from alambique.memory_config import LLM_INSTABILITY_WINDOW_SECONDS

if TYPE_CHECKING:
    pass

logger = logging.getLogger("alambique.tools")

API_KEY_RETRY_INTERVAL_SECONDS = 30
API_KEY_NOTIFY_COOLDOWN_SECONDS = 300
DEFAULT_STATUS_PORT = 9042


class ToolHandlerBase:
    """Shared state and infrastructure for MCP tool handlers."""

    def __init__(self, db: Database, ollama: OllamaClient, api_key: str | None = None) -> None:
        self.db = db
        self.ollama = ollama
        self._api_key = api_key
        self._api_key_source: str | None = None
        self._api_key_last_error: str | None = None
        self._api_key_last_attempt: datetime | None = None
        self._api_key_attempt_count: int = 0
        self._api_key_notify_sent_at: datetime | None = None
        self._started_at = datetime.now(timezone.utc)
        self._api_key_retry_task: Optional[asyncio.Task] = None
        self._consolidator: Optional[ConsolidatorClient] = None
        self._consolidation_warnings: list[str] = []
        self._recall: Optional[RecallClient] = None
        self._db_lock: Optional[asyncio.Lock] = None
        self._llm_last_success_at: datetime | None = None
        self._llm_last_failure_at: datetime | None = None
        self._llm_last_error: str | None = None
        self._llm_recent_recovery: bool = False

    @asynccontextmanager
    async def _db_guard(self) -> AsyncIterator[Database]:
        """Serialize SQLite access across MCP tools and background workers."""
        if self._db_lock is None:
            self._db_lock = asyncio.Lock()
        async with self._db_lock:
            yield self.db

    @property
    def consolidator(self) -> ConsolidatorClient:
        if self._consolidator is None:
            if self._api_key is None:
                raise RuntimeError("API key no disponible")
            self._consolidator = ConsolidatorClient(
                self._api_key,
                on_outcome=self._note_llm_outcome,
            )
        return self._consolidator

    @property
    def recall(self) -> RecallClient:
        if self._recall is None:
            if self._api_key is None:
                raise RuntimeError("API key no disponible")
            self._recall = RecallClient(
                self._api_key,
                on_outcome=self._note_llm_outcome,
            )
        return self._recall

    def _note_llm_outcome(
        self,
        success: bool,
        error: str | None = None,
        retried: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc)
        if success:
            self._llm_last_success_at = now
            if retried or (
                self._llm_last_failure_at
                and (now - self._llm_last_failure_at).total_seconds()
                < LLM_INSTABILITY_WINDOW_SECONDS
            ):
                self._llm_recent_recovery = True
            else:
                self._llm_recent_recovery = False
            return

        self._llm_last_failure_at = now
        self._llm_last_error = error
        self._llm_recent_recovery = False

    def _build_system_message(
        self,
        *,
        overall: str,
        ollama_ok: bool,
        pending: int,
        api_runtime: ApiKeyState,
        health: MemoryHealthOutput,
    ) -> tuple[str, str]:
        now = datetime.now(timezone.utc)

        if not ollama_ok:
            return (
                "No estoy al 100% — Ollama no responde y la memoria semántica está parada.",
                "error",
            )

        if api_runtime.status == "waiting_pass":
            return (
                "Esperando tu contraseña GPG (pass). Sin ella funciono en modo limitado.",
                "warning",
            )

        if not self.online:
            detail = api_runtime.detail or "sin API key"
            return (
                f"Modo limitado: {detail}. La memoria local funciona; LLM cloud y consolidación esperan.",
                "warning",
            )

        if pending > 0:
            return (
                f"Casi perfecto — {pending} sesión(es) pendiente(s) de consolidar en memoria.",
                "warning",
            )

        if health.warnings:
            from alambique.warning_labels import (
                format_warnings_for_humans,
                warning_message_level,
            )

            return (
                format_warnings_for_humans(health.warnings),
                warning_message_level(health.warnings),
            )

        recent_failure = (
            self._llm_last_failure_at
            and (now - self._llm_last_failure_at).total_seconds()
            < LLM_INSTABILITY_WINDOW_SECONDS
        )
        recovered = self._llm_recent_recovery or (
            recent_failure
            and self._llm_last_success_at
            and self._llm_last_success_at > self._llm_last_failure_at
        )
        if recovered:
            return (
                "Hubo inestabilidad en el LLM cloud (OpenCode), pero ahora responde bien.",
                "warning",
            )

        if (
            recent_failure
            and (
                not self._llm_last_success_at
                or self._llm_last_failure_at > self._llm_last_success_at
            )
        ):
            detail = self._llm_last_error or "error desconocido"
            return (
                f"LLM cloud inestable — último fallo: {detail}. Usando fallbacks.",
                "error",
            )

        if overall == "ok":
            return (
                "Todo va perfecto. Memoria operativa, estable y sincronizada.",
                "ok",
            )

        return (
            "Funcionando con limitaciones — revisa el panel de salud.",
            "warning",
        )

    @property
    def online(self) -> bool:
        return self._api_key is not None

    def note_api_key_attempt(self, result: ApiKeyFetchResult) -> bool:
        """Record an API key fetch attempt and update daemon runtime state."""
        self._api_key_last_attempt = datetime.now(timezone.utc)
        self._api_key_attempt_count += 1

        if result.key:
            self._api_key = result.key
            self._api_key_source = result.source
            self._api_key_last_error = None
            self._api_key_notify_sent_at = None
            self._consolidator = None
            self._recall = None
            return True

        self._api_key_last_error = result.error or "API key no disponible"
        logger.warning(
            "API key no disponible (intento %d): %s — reintento en %ds",
            self._api_key_attempt_count,
            self._api_key_last_error,
            API_KEY_RETRY_INTERVAL_SECONDS,
        )
        self._maybe_notify_pass_waiting()
        return False

    def _maybe_notify_pass_waiting(self) -> None:
        if not is_waiting_pass_error(self._api_key_last_error):
            return

        now = datetime.now(timezone.utc)
        if self._api_key_notify_sent_at is not None:
            elapsed = now - self._api_key_notify_sent_at
            if elapsed < timedelta(seconds=API_KEY_NOTIFY_COOLDOWN_SECONDS):
                return

        if send_desktop_notification(
            PASS_WAIT_TITLE,
            PASS_WAIT_BODY,
            urgency="critical",
        ):
            self._api_key_notify_sent_at = now
            logger.info("Notificación de escritorio enviada: esperando contraseña GPG")

    def _api_key_runtime_state(self) -> ApiKeyState:
        if self.online:
            return ApiKeyState(
                status="loaded",
                source=self._api_key_source,
                detail=None,
                last_attempt=self._api_key_last_attempt,
                attempt_count=self._api_key_attempt_count,
                retry_interval_seconds=API_KEY_RETRY_INTERVAL_SECONDS,
                pass_timeout_seconds=_pass_show_timeout(),
            )

        detail = self._api_key_last_error
        status = "missing"
        if detail:
            lowered = detail.lower()
            if "pass no respondió" in lowered or "pinentry" in lowered:
                status = "waiting_pass"
            elif "pass" in lowered or "gpg" in lowered:
                status = "failed"

        return ApiKeyState(
            status=status,
            source=self._api_key_source,
            detail=detail,
            last_attempt=self._api_key_last_attempt,
            attempt_count=self._api_key_attempt_count,
            retry_interval_seconds=API_KEY_RETRY_INTERVAL_SECONDS,
            pass_timeout_seconds=_pass_show_timeout(),
        )


    async def shutdown_open_sessions(self) -> None:
        """Sync and close open sessions before daemon shutdown."""
        from alambique.models import SessionStatus

        async with self._db_guard():
            open_sessions = self.db.get_open_sessions()
        for session in open_sessions:
            logger.info("Shutdown: cerrando sesión abierta %s", session.id)
            await self._close_session(
                session.id,
                SessionStatus.TRUNCATED,
                conversation_id=session.conversation_id,
                client=session.client,
            )

