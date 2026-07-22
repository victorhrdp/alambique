"""Session lifecycle MCP tools."""

from __future__ import annotations

import asyncio
import logging
import os

from alambique.memory_config import (
    AGENT_NAME,
    RECALL_THRESHOLD_DEFAULT,
    VALID_EXPRESSIONS,
)
from alambique.models import (
    Message,
    SessionEndOutput,
    SessionStartOutput,
    SessionStatus,
)
logger = logging.getLogger("alambique.tools")


class SessionMixin:

    def _seed_persona_if_needed(self, persona_seed: str | None) -> None:
        """Persist initial personality capsule if none exists."""
        seed = (persona_seed or "").strip()
        if not seed:
            return
        existing = self.db.get_relevant_relationship_capsule("personality")
        if existing:
            return
        self.db.upsert_relationship_capsule("personality", seed)
        logger.info("Persona capsule sembrada.")

    async def _compose_session_persona(self) -> tuple[str | None, list[str]]:
        warnings: list[str] = []
        capsule = ""
        async with self._db_guard():
            capsule = self.db.get_relevant_relationship_capsule("personality") or self.db.get_relevant_relationship_capsule("general") or ""

        if not capsule.strip():
            return None, warnings

        if self.online:
            try:
                return await self.recall.compose_personality(capsule), warnings
            except Exception as e:
                self._note_llm_outcome(False, str(e))
                logger.warning("Error componiendo personalidad: %s", e)
                warnings.append("persona_llm_failed")

        # fallback to capsule text
        if not self.online:
            warnings.append("persona_offline_fallback")
        return capsule, warnings

    def _resolve_client_binding(
        self,
        client: str | None,
        conversation_id: str | None,
        workspace: str | None,
    ) -> tuple[str | None, str | None, list[str]]:
        import os

        warnings: list[str] = []
        if not client:
            warnings.append("binding_missing_client")
            return None, None, warnings

        if client == "grok":
            from alambique.transcripts.grok_cli import (
                normalize_workspace,
                resolve_grok_session_id,
            )

            workspace = normalize_workspace(workspace)
            if not workspace and not conversation_id:
                warnings.append("binding_missing_workspace")
            resolved, resolve_warnings = resolve_grok_session_id(
                conversation_id=conversation_id,
                workspace=workspace,
            )
            warnings.extend(resolve_warnings)
            if not resolved:
                warnings.append("binding_failed")
            return client, resolved, warnings

        if client == "antigravity_cli":
            from alambique.transcripts.antigravity_cli import (
                resolve_antigravity_conversation_id,
            )
            from alambique.transcripts.grok_cli import normalize_workspace as normalize_ws

            workspace = normalize_ws(workspace)
            if not workspace and not conversation_id:
                warnings.append("binding_missing_workspace")
            resolved, resolve_warnings = resolve_antigravity_conversation_id(
                conversation_id=conversation_id,
                workspace=workspace,
            )
            warnings.extend(resolve_warnings)
            if not resolved:
                warnings.append("binding_failed")
            return client, resolved, warnings

        if client == "opencode":
            from alambique.transcripts.grok_cli import normalize_workspace as normalize_ws
            from alambique.transcripts.opencode_cli import resolve_opencode_session_id

            workspace = normalize_ws(workspace)
            if not workspace and not conversation_id:
                warnings.append("binding_missing_workspace")
            resolved, resolve_warnings = resolve_opencode_session_id(
                conversation_id=conversation_id,
                workspace=workspace,
            )
            warnings.extend(resolve_warnings)
            if not resolved:
                warnings.append("binding_failed")
            return client, resolved, warnings

        resolved = conversation_id
        if not resolved:
            warnings.append("binding_failed")
        return client, resolved, warnings

    async def session_start(
        self,
        persona_seed: str | None = None,
        client: str | None = None,
        conversation_id: str | None = None,
        workspace: str | None = None,
    ) -> SessionStartOutput:
        warnings: list[str] = []
        if not self.online:
            warnings.append("offline_mode")
        if not await self.ollama.health():
            warnings.append("ollama_unavailable")

        bound_client, bound_conversation_id, bind_warnings = self._resolve_client_binding(
            client, conversation_id, workspace
        )
        warnings.extend(bind_warnings)

        binding_failed = bound_client is not None and bound_conversation_id is None
        if binding_failed:
            async with self._db_guard():
                self._seed_persona_if_needed(persona_seed)
            persona, persona_warnings = await self._compose_session_persona()
            warnings.extend(persona_warnings)
            return SessionStartOutput(
                session_id=None,
                status="error",
                persona=persona,
                client=bound_client,
                conversation_id=None,
                session_reused=False,
                is_new=False,
                degraded=True,
                warnings=warnings,
                initial_context=None,
                active_thread_keys=[],
            )

        session_reused = False
        async with self._db_guard():
            already_existed = len(self.db.get_all_sessions()) > 0
            self._seed_persona_if_needed(persona_seed)

            if bound_client and bound_conversation_id:
                # Prefer any previous session for the same binding (even if closed/truncated)
                # so that long Grok chats keep one Alambique session and accumulate threads.
                existing = self.db.get_session_by_binding(
                    bound_client, bound_conversation_id
                )
                if existing:
                    session = existing
                    session_reused = True
                    if existing.status != "open":
                        self.db.reopen_session(existing.id)
                        warnings.append("session_reopened")
                    # Close other open duplicates for the binding if any
                    for duplicate in self.db.get_open_sessions_by_binding(
                        bound_client, bound_conversation_id
                    ):
                        if duplicate.id != existing.id:
                            self.db.close_session(duplicate.id, SessionStatus.TRUNCATED)
                else:
                    session = self.db.create_session(
                        client=bound_client,
                        conversation_id=bound_conversation_id,
                    )
            else:
                session = self.db.create_session(
                    client=bound_client,
                    conversation_id=bound_conversation_id,
                )
        persona, persona_warnings = await self._compose_session_persona()
        warnings.extend(persona_warnings)

        # New memory activation (thematic threads + capsule + echoes)
        from alambique.activation import ActivationEngine
        engine = ActivationEngine(self.db, self.ollama)
        # Try to give a hint of the first user message for vector tier if transcript already has content.
        # Works for all clients (grok, antigravity_cli, opencode) via db or direct provider read.
        initial_hint = None
        try:
            async with self._db_guard():
                msgs = self.db.get_session_messages(session.id)
            if msgs:
                user_msgs = [m.content for m in msgs if m.role == "user"][:1]
                if user_msgs:
                    initial_hint = user_msgs[0][:300]
            else:
                # Fallback read from transcript provider (no db write) for fresh starts on other CLIs
                from alambique.transcripts import get_active_provider
                provider = get_active_provider(bound_conversation_id, bound_client)
                if provider:
                    raw = provider.get_messages(bound_conversation_id)
                    user_msgs = [m.get("content", "") for m in raw if m.get("role") == "user"][:1]
                    if user_msgs:
                        initial_hint = user_msgs[0][:300]
        except Exception:
            pass
        act = await engine.activate(initial_hint)

        # Write lightweight state for desktop widgets / companions (face is separate via active_expression.json)
        try:
            import json
            from pathlib import Path
            mem_path = Path.home() / ".local" / "share" / "alambique" / "active_memory.json"
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            initiative = act.get("initiative")
            mem_path.write_text(json.dumps({
                "active_thread_keys": act.get("active_thread_keys", []),
                "initial_context_snippet": (act.get("initial_context") or "")[:800],
                "pending_initiative": initiative,
                "updated_at": __import__("datetime").datetime.now().isoformat()
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write active_memory.json for widgets: %s", e)

        return SessionStartOutput(
            session_id=session.id,
            status="ok",
            persona=persona,
            client=bound_client,
            conversation_id=bound_conversation_id,
            session_reused=session_reused,
            is_new=not already_existed and not session_reused,
            degraded=bool(warnings),
            warnings=warnings,
            initial_context=act.get("initial_context"),
            active_thread_keys=act.get("active_thread_keys"),
        )

    async def _sync_session_transcript(
        self,
        session_id: str,
        conversation_id: str | None = None,
        client: str | None = None,
    ) -> int:
        """Import messages from the bound external transcript, if available."""
        import os
        from alambique.transcripts import get_active_provider

        async with self._db_guard():
            stored = self.db.get_session(session_id)

        effective_client = client or (stored.client if stored else None)
        conv_id = (
            conversation_id
            or (stored.conversation_id if stored else None)
            or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
            or os.environ.get("GROK_SESSION_ID")
            or os.environ.get("OPENCODE_SESSION_ID")
        )
        if not conv_id:
            if effective_client == "grok":
                from alambique.transcripts.grok_cli import resolve_grok_session_id

                conv_id, _ = resolve_grok_session_id()
            elif effective_client == "antigravity_cli":
                from alambique.transcripts.antigravity_cli import (
                    resolve_antigravity_conversation_id,
                )

                conv_id, _ = resolve_antigravity_conversation_id()
            elif effective_client == "opencode":
                from alambique.transcripts.opencode_cli import resolve_opencode_session_id

                conv_id, _ = resolve_opencode_session_id()
        provider = get_active_provider(conv_id, effective_client)
        if not provider:
            logger.info(
                "No active transcript provider for session %s; keeping existing messages.",
                session_id,
            )
            return 0

        try:
            raw_messages = provider.get_messages(conv_id)
            if not raw_messages:
                return 0

            db_messages = [
                Message(session_id=session_id, role=m["role"], content=m["content"])
                for m in raw_messages
            ]
            async with self._db_guard():
                self.db.clear_and_set_session_messages(session_id, db_messages)
            logger.info(
                "Sincronizados %d mensajes automáticamente usando %s",
                len(db_messages),
                provider.__class__.__name__,
            )
            return len(db_messages)
        except Exception as e:
            logger.error(
                "Error al sincronizar mensajes del transcript para sesión %s: %s",
                session_id,
                e,
            )
            return 0

    async def _close_session(
        self,
        session_id: str,
        status: SessionStatus,
        conversation_id: str | None = None,
        client: str | None = None,
    ) -> None:
        await self._sync_session_transcript(session_id, conversation_id, client)
        async with self._db_guard():
            self.db.close_session(session_id, status)

    # ── tool: session_end ───────────────────────────────────────

    async def session_end(
        self,
        session_id: str,
        truncated: bool = False,
        conversation_id: str | None = None,
        client: str | None = None,
    ) -> SessionEndOutput:
        status = SessionStatus.TRUNCATED if truncated else SessionStatus.CLOSED
        await self._close_session(session_id, status, conversation_id, client)

        async with self._db_guard():
            session = self.db.get_session(session_id)
            pending = self.db.count_pending_consolidations_db()
            queued = (
                session is not None
                and session.status in (SessionStatus.CLOSED, SessionStatus.TRUNCATED)
                and not session.consolidated
            )

        # Activate consolidation automatically when session ends.
        # Run in background task so session_end returns quickly to the client.
        # No persistent loops - only one-off per session_end.
        if queued:
            asyncio.create_task(
                self._trigger_consolidation_after_end(session_id),
                name=f"consolidate-after-end-{session_id}"
            )
            logger.info(f"Consolidation triggered in background after session_end for {session_id}")

        return SessionEndOutput(queued=queued, pending_consolidation=pending)

    async def _trigger_consolidation_after_end(self, session_id: str) -> None:
        """Background consolidation after session_end. Fire-and-forget, no polling loop.
        Uses full (non-light) mode for quality; widget buttons can use light=True to spare CPU.
        """
        try:
            result = await self.consolidate_session(session_id, light=False)
            logger.info(f"Auto-consolidation after session_end completed: {result}")
        except Exception as e:
            logger.error(f"Auto-consolidation after session_end failed for {session_id}: {e}")

    # ── tool: session_update ─────────────────────────────────────

    async def session_update(
        self,
        session_id: str,
        expression: str,
        mood_text: str,
    ) -> dict:
        if expression not in VALID_EXPRESSIONS:
            raise ValueError(
                f"Expresión inválida: {expression!r}. "
                f"Válidas: {', '.join(sorted(VALID_EXPRESSIONS))}"
            )
        async with self._db_guard():
            session = self.db.get_session(session_id)
            if not session:
                raise ValueError(f"Sesión no encontrada: {session_id}")
            self.db.update_session_expression(session_id, expression, mood_text)

        # Also write central json for widget / direct consumers (silent update path)
        try:
            import json
            from pathlib import Path
            expr_path = Path.home() / ".local" / "share" / "alambique" / "active_expression.json"
            expr_path.parent.mkdir(parents=True, exist_ok=True)
            expr_path.write_text(json.dumps({
                "expression": expression,
                "mood_text": mood_text
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write active_expression.json: %s", e)

        return {"status": "ok"}

    async def close_session(self, session_id: str) -> dict:
        """Manually close an open session (intended for widget/manual management).

        Performs transcript sync if bound, then closes it.
        Does not require the full session_end contract.
        """
        async with self._db_guard():
            sess = self.db.get_session(session_id)
            if not sess:
                return {"status": "error", "message": f"Sesión no encontrada: {session_id}"}
            if sess.status != "open":
                return {"status": "ok", "message": "La sesión ya no estaba abierta", "session_id": session_id}

        await self._close_session(
            session_id,
            SessionStatus.CLOSED,
            conversation_id=sess.conversation_id,
            client=sess.client,
        )
        return {"status": "ok", "session_id": session_id, "closed": True}
