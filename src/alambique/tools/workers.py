"""Background workers (minimal - no consolidation loops to avoid CPU burn)."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from alambique.consolidator import fetch_api_key
from alambique.tools.base import API_KEY_RETRY_INTERVAL_SECONDS

logger = logging.getLogger("alambique.tools")


class WorkerMixin:
    """Lightweight background tasks. NO automatic consolidation loops.
    Consolidation is triggered explicitly from session_end or manual consolidate_session calls.
    """

    async def start_background_tasks(self) -> None:
        """Start only the lightweight API key retry task.
        There is deliberately NO consolidation background loop.
        """
        self._api_key_retry_task = asyncio.create_task(self._api_key_retry_loop())
        logger.info("Background tasks started (NO consolidation loop - explicit only).")

    async def stop_background_tasks(self) -> None:
        if self._api_key_retry_task:
            self._api_key_retry_task.cancel()
        if self._consolidator:
            await self._consolidator.close()
        if self._recall:
            await self._recall.close()

    async def _api_key_retry_loop(self) -> None:
        """Retry loading the API key when the daemon started offline."""
        while True:
            try:
                if not self.online:
                    result = await asyncio.to_thread(fetch_api_key)
                    if self.note_api_key_attempt(result):
                        logger.info("API key cargada dinámicamente. Pasando a modo online.")
                await asyncio.sleep(API_KEY_RETRY_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            except Exception as e:
                self._api_key_last_error = f"Error reintentando API key: {e}"
                self._api_key_last_attempt = datetime.now(timezone.utc)
                self._api_key_attempt_count += 1
                logger.error(
                    "API key no disponible (intento %d): %s — reintento en %ds",
                    self._api_key_attempt_count,
                    self._api_key_last_error,
                    API_KEY_RETRY_INTERVAL_SECONDS,
                )
                await asyncio.sleep(API_KEY_RETRY_INTERVAL_SECONDS)

