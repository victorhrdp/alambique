"""Daemon health and status endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from alambique import __version__
from alambique.models import (
    ActiveSessionState,
    DaemonStatusOutput,
    MemoryHealthCheck,
    MemoryHealthOutput,
    MemoryStatusOutput,
)
from alambique.tools.base import DEFAULT_STATUS_PORT

logger = logging.getLogger("alambique.tools")


class StatusMixin:

    async def memory_status(self) -> MemoryStatusOutput:
        async with self._db_guard():
            sessions = self.db.get_all_sessions()
            threads = self.db.conn.execute(
                "SELECT COUNT(*) FROM threads WHERE status = 'active'"
            ).fetchone()[0]
            capsules = self.db.conn.execute(
                "SELECT COUNT(*) FROM relationship_capsules"
            ).fetchone()[0]
            echoes = self.db.conn.execute(
                "SELECT COUNT(*) FROM echoes"
            ).fetchone()[0]
            pending = self.db.count_pending_consolidations_db()
            last = self.db.last_consolidation_time()

            return MemoryStatusOutput(
                sessions=len(sessions),
                threads=threads,
                capsules=capsules,
                echoes=echoes,
                pending_consolidation=pending,
                last_consolidation=datetime.fromisoformat(last) if last else None,
            )

    async def memory_health(self) -> MemoryHealthOutput:
        checks: dict[str, MemoryHealthCheck] = {}
        warnings: list[str] = []

        ollama_ok = await self.ollama.health()
        checks["ollama"] = MemoryHealthCheck(
            status="ok" if ollama_ok else "error",
            detail=None if ollama_ok else "Ollama no responde en :11434",
        )
        if not ollama_ok:
            warnings.append("ollama_unavailable")

        api_ok = self.online
        checks["api_key"] = MemoryHealthCheck(
            status="ok" if api_ok else "warning",
            detail=None if api_ok else "Sin API key — consolidación y recall LLM desactivados",
        )
        if not api_ok:
            warnings.append("offline_mode")

        async with self._db_guard():
            pending = self.db.count_pending_consolidations_db()
            consolidation_status = "ok"
            consolidation_detail = (
                f"{pending} sesiones pendientes" if pending else "Sin pendientes"
            )
            if pending > 0:
                consolidation_status = "warning"
                warnings.append("pending_consolidation")
            last = self.db.last_consolidation_time()
            checks["consolidation"] = MemoryHealthCheck(
                status=consolidation_status,
                detail=consolidation_detail
                + (f"; última: {last}" if last else "; sin consolidaciones previas"),
            )

            session_orphans = self.db.count_orphan_session_embeddings()
            sessions_missing = self.db.count_sessions_missing_embeddings()
            thread_orphans = self.db.count_orphan_thread_embeddings()
            threads_missing = self.db.count_threads_missing_embeddings()
            capsule_orphans = self.db.count_orphan_capsule_embeddings()
            capsules_missing = self.db.count_capsules_missing_embeddings()
            echo_orphans = self.db.count_orphan_echo_embeddings()
            echoes_missing = self.db.count_echoes_missing_embeddings()

            embedding_ok = (
                session_orphans == 0
                and sessions_missing == 0
                and thread_orphans == 0
                and threads_missing == 0
                and capsule_orphans == 0
                and capsules_missing == 0
                and echo_orphans == 0
                and echoes_missing == 0
            )
            embedding_status = "ok" if embedding_ok else "warning"
            embedding_parts = []
            if session_orphans:
                embedding_parts.append(
                    f"{session_orphans} vectores de sesión huérfanos (rebuild_vectors)"
                )
            if sessions_missing:
                embedding_parts.append(
                    f"{sessions_missing} sesiones con resumen sin vector"
                )
            if thread_orphans:
                embedding_parts.append(f"{thread_orphans} vectores de hilos huérfanos")
            if threads_missing:
                embedding_parts.append(f"{threads_missing} hilos sin vector")
            if capsule_orphans:
                embedding_parts.append(f"{capsule_orphans} vectores de cápsulas huérfanos")
            if capsules_missing:
                embedding_parts.append(f"{capsules_missing} cápsulas sin vector")
            if echo_orphans:
                embedding_parts.append(f"{echo_orphans} vectores de ecos huérfanos")
            if echoes_missing:
                embedding_parts.append(f"{echoes_missing} ecos sin vector")
            checks["embeddings"] = MemoryHealthCheck(
                status=embedding_status,
                detail="; ".join(embedding_parts) if embedding_parts else "Embeddings en orden",
            )
            if session_orphans > 0:
                warnings.append("orphan_session_embeddings")
            if sessions_missing > 0:
                warnings.append("sessions_missing_embeddings")
            if thread_orphans > 0 or capsule_orphans > 0 or echo_orphans > 0:
                warnings.append("orphan_new_embeddings")
            if threads_missing > 0 or capsules_missing > 0 or echoes_missing > 0:
                warnings.append("new_entities_missing_embeddings")

            if self._consolidation_warnings:
                warnings.extend(self._consolidation_warnings[-5:])

            healthy = (
                ollama_ok
                and api_ok
                and pending == 0
                and embedding_ok
            )

        return MemoryHealthOutput(
            healthy=healthy,
            mode="online" if api_ok else "offline",
            checks=checks,
            warnings=warnings,
        )

    async def daemon_status(self, port: int = DEFAULT_STATUS_PORT) -> DaemonStatusOutput:
        """Unified daemon status for HTTP /status and the KDE widget."""
        health = await self.memory_health()
        stats = await self.memory_status()
        async with self._db_guard():
            active_row = self.db.get_latest_open_session_row()
            open_rows = self.db.conn.execute(
                "SELECT id, client, conversation_id, expression, mood_text, created_at "
                "FROM sessions WHERE status = 'open' ORDER BY created_at DESC"
            ).fetchall()

        active_session = ActiveSessionState()
        expression = "normal"
        mood_text = ""
        if active_row:
            expression = active_row["expression"] or "normal"
            mood_text = active_row["mood_text"] or ""
            active_session = ActiveSessionState(
                id=active_row["id"],
                client=active_row["client"],
                conversation_id=active_row["conversation_id"],
                expression=expression,
                mood_text=mood_text,
                bound=bool(active_row["client"] and active_row["conversation_id"]),
            )

        # Build list of all open sessions for widget visibility and manual close
        open_sessions: list[ActiveSessionState] = []
        for row in open_rows:
            open_sessions.append(
                ActiveSessionState(
                    id=row["id"],
                    client=row["client"],
                    conversation_id=row["conversation_id"],
                    expression=row["expression"] or "normal",
                    mood_text=row["mood_text"] or "",
                    bound=bool(row["client"] and row["conversation_id"]),
                )
            )

        # Pending consolidations - for widget "consolidate" buttons (no auto loops)
        async with self._db_guard():
            pending_rows = self.db.conn.execute(
                "SELECT id, client, conversation_id, ended_at, created_at, status "
                "FROM sessions WHERE status IN ('closed','truncated') AND consolidated = 0 "
                "ORDER BY ended_at DESC LIMIT 50"
            ).fetchall()
        pending_for_widget = [
            {
                "id": r["id"],
                "client": r["client"],
                "conversation_id": r["conversation_id"],
                "status": r["status"],
                "ended_at": str(r["ended_at"]) if r["ended_at"] else None,
                "created_at": str(r["created_at"]) if r["created_at"] else None,
            }
            for r in pending_rows
        ]

        # Prefer central active_expression.json (direct agent writes for live face updates)
        try:
            import json
            from pathlib import Path
            expr_path = Path.home() / ".local" / "share" / "alambique" / "active_expression.json"
            if expr_path.exists():
                data = json.loads(expr_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    if data.get("expression"):
                        expression = data["expression"]
                    if data.get("mood_text") is not None:
                        mood_text = data["mood_text"]
        except Exception:
            pass

        # Refresh active_session with possibly overridden expression/mood from json
        if active_row:
            active_session = ActiveSessionState(
                id=active_row["id"],
                client=active_row["client"],
                conversation_id=active_row["conversation_id"],
                expression=expression,
                mood_text=mood_text,
                bound=bool(active_row["client"] and active_row["conversation_id"]),
            )

        # Write open sessions JSON for widget (easy polling + close buttons)
        try:
            import json
            from pathlib import Path
            from datetime import datetime as dt
            open_json_path = Path.home() / ".local" / "share" / "alambique" / "open_sessions.json"
            open_json_path.parent.mkdir(parents=True, exist_ok=True)
            open_data = [
                {
                    "id": s.id,
                    "client": s.client,
                    "conversation_id": s.conversation_id,
                    "expression": s.expression,
                    "mood_text": s.mood_text,
                    "bound": s.bound,
                }
                for s in open_sessions
            ]
            rich_open = []
            for row in open_rows:
                rich_open.append({
                    "id": row["id"],
                    "client": row["client"],
                    "conversation_id": row["conversation_id"],
                    "created_at": str(row["created_at"]) if row["created_at"] else None,
                })
            open_json_path.write_text(
                json.dumps({"open_sessions": open_data, "details": rich_open}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            open_json_path.write_text(json.dumps(open_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write open_sessions.json: %s", e)

        # Write pending consolidations JSON so the widget can show buttons to consolidate manually
        try:
            import json
            from pathlib import Path
            pending_json_path = Path.home() / ".local" / "share" / "alambique" / "pending_consolidations.json"
            pending_json_path.parent.mkdir(parents=True, exist_ok=True)
            pending_json_path.write_text(
                json.dumps({"pending": pending_for_widget}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning("Failed to write pending_consolidations.json: %s", e)

        api_runtime = self._api_key_runtime_state()
        api_check = health.checks.get(
            "api_key",
            MemoryHealthCheck(status="warning", detail="Sin API key"),
        )
        llm_detail = api_check.detail
        if not self.online and api_runtime.detail:
            llm_detail = api_runtime.detail

        widget_checks = {
            "daemon": MemoryHealthCheck(
                status="ok",
                detail=f"Puerto {port} · v{__version__}",
            ),
            "ollama": health.checks.get(
                "ollama",
                MemoryHealthCheck(status="error", detail="Sin datos"),
            ),
            "llm": MemoryHealthCheck(
                status=api_check.status,
                detail=llm_detail,
            ),
            "embeddings": health.checks.get(
                "embeddings",
                MemoryHealthCheck(status="warning", detail="Sin datos"),
            ),
            "consolidation": health.checks.get(
                "consolidation",
                MemoryHealthCheck(status="warning", detail="Sin datos"),
            ),
        }

        ollama_ok = widget_checks["ollama"].status == "ok"
        pending = stats.pending_consolidation
        embedding_check = widget_checks["embeddings"]
        orphans = 0
        stale = 0
        if embedding_check.detail:
            for part in embedding_check.detail.split(";"):
                part = part.strip()
                if "sin embedding" in part:
                    try:
                        orphans = int(part.split()[0])
                    except ValueError:
                        pass
                if "huérfanos" in part:
                    try:
                        stale = int(part.split()[0])
                    except ValueError:
                        pass

        if not ollama_ok:
            overall = "down"
        elif (
            self.online
            and pending == 0
            and embedding_check.status == "ok"
            and health.healthy
        ):
            overall = "ok"
        else:
            overall = "degraded"

        if overall == "ok":
            status_label = "Operativo"
            status_summary = "Operativo"
        elif not ollama_ok:
            status_label = "Caído"
            status_summary = widget_checks["ollama"].detail or "Ollama no responde"
        elif not self.online:
            status_label = "Degradado"
            status_summary = f"Degradado · {api_runtime.detail or 'Sin API key'}"
        elif pending > 0:
            status_label = "Degradado"
            status_summary = f"Degradado · {pending} sesión(es) pendiente(s) de consolidar"
        elif embedding_check.status != "ok":
            status_label = "Degradado"
            status_summary = f"Degradado · {embedding_check.detail}"
        elif health.warnings:
            from alambique.warning_labels import format_warnings_for_humans

            status_label = "Degradado"
            status_summary = format_warnings_for_humans(health.warnings)
        else:
            status_label = "Degradado"
            status_summary = "Degradado"

        system_message, system_message_level = self._build_system_message(
            overall=overall,
            ollama_ok=ollama_ok,
            pending=pending,
            api_runtime=api_runtime,
            health=health,
        )

        now = datetime.now(timezone.utc)
        uptime = (now - self._started_at).total_seconds()

        return DaemonStatusOutput(
            version=__version__,
            started_at=self._started_at,
            uptime_seconds=uptime,
            overall=overall,
            status_label=status_label,
            status_summary=status_summary,
            system_message=system_message,
            system_message_level=system_message_level,
            mode=health.mode,
            online=True,
            db_exists=True,
            healthy=health.healthy,
            checks=widget_checks,
            warnings=health.warnings,
            api_key=api_runtime,
            stats=stats,
            active_session=active_session,
            open_sessions=open_sessions,
            sessions=stats.sessions,
            threads=stats.threads,
            capsules=stats.capsules,
            echoes=stats.echoes,
            expression=expression,
            mood_text=mood_text,
            last_consolidation=stats.last_consolidation,
        )
