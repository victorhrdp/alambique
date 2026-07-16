"""Memory MCP tools: recall, search, maintenance."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from alambique.memory_config import AGENT_NAME, RECALL_TOP_K
from alambique.models import (
    MemoryContextOutput,
    MemoryRebuildVectorsOutput,
    MemoryRecallOutput,
)
from alambique.vector_rebuild import rebuild_vectors

logger = logging.getLogger("alambique.tools")


class MemoryMixin:

    async def memory_recall(self, query: str) -> MemoryRecallOutput:
        sessions_out: list[dict] = []
        threads_out: list[dict] = []
        capsules_out: list[dict] = []
        warnings: list[str] = []
        used_fallback = False
        summary_llm_used = False

        if not await self.ollama.health():
            warnings.append("ollama_unavailable")

        try:
            query_emb = await self.ollama.embed(query)

            async with self._db_guard():
                # Threads (new model) — primary semantic search source
                thread_hits = self.db.vector_search_threads(query_emb, limit=5)
                seen_thread_keys: set[str] = set()
                for h in thread_hits:
                    t = h.get("thread", h) if isinstance(h, dict) else h
                    if t and t.get("key"):
                        seen_thread_keys.add(t["key"])
                        threads_out.append({
                            "key": t.get("key"),
                            "title": t.get("title"),
                            "snippet": (t.get("current_state") or "")[:200],
                            "salience": t.get("salience"),
                        })

                # Sessions — derived from thread hits (no longer relies on vec0_sessions + summary)
                if seen_thread_keys:
                    session_rows = self.db.get_sessions_for_thread_keys(
                        list(seen_thread_keys), limit=5
                    )
                    for s in session_rows:
                        snippet = s.get("summary")
                        if not snippet:
                            session_threads = self.db.get_threads_for_session(s["id"])
                            snippet = "; ".join(
                                t.get("title", "") for t in session_threads[:3]
                            )
                        sessions_out.append({
                            "id": s["id"],
                            "snippet": snippet or "",
                        })

                # Capsules via vector
                cap_hits = self.db.vector_search_capsules(query_emb, limit=3)
                for h in cap_hits:
                    c = h.get("capsule", h) if isinstance(h, dict) else h
                    if c and c.get("scope"):
                        capsules_out.append({
                            "scope": c.get("scope"),
                            "snippet": c.get("content", "")[:200],
                        })

                # Echoes via vector (top ones)
                echo_hits = self.db.vector_search_echoes(query_emb, limit=5)
                for h in echo_hits:
                    e = h.get("echo", h) if isinstance(h, dict) else h
                    if e and e.get("content"):
                        pass  # echoes available for future use
        except Exception as e:
            logger.warning("Vector search falló: %s.", e)
            warnings.append("vector_search_failed")
            used_fallback = True

        summary = ""
        context_for_summary = sessions_out + threads_out + capsules_out
        if self.online and context_for_summary:
            try:
                summary = await self.recall.compose_summary(query, AGENT_NAME, context_for_summary)
                if summary:
                    summary_llm_used = True
            except Exception as e:
                self._note_llm_outcome(False, str(e))
                logger.warning("Error en recall LLM: %s", e)
                warnings.append("summary_llm_failed")
        elif context_for_summary:
            warnings.append("summary_offline")

        if not summary and context_for_summary:
            summary = "Hilos y cápsulas relacionados: " + "; ".join(
                f"{s.get('id') or s.get('key', '')}: {s.get('snippet', '')}" for s in context_for_summary[:5]
            )
            if not summary_llm_used:
                warnings.append("summary_fallback_generic")

        if used_fallback:
            warnings.append("recall_degraded")

        return MemoryRecallOutput(
            summary=summary,
            related_sessions=sessions_out,
            related_threads=threads_out,
            related_capsules=capsules_out,
            degraded=bool(warnings),
            warnings=warnings,
        )

    # ── tool: memory_search ─────────────────────────────────────

    async def memory_search(self, query: str) -> dict[str, list]:
        async with self._db_guard():
            results = self.db.search_messages_fts(query, limit=20)
        return {"results": results}

    # ── tool: memory_context ────────────────────────────────────

    async def memory_context(self, session_id: str, offset: int = 0, limit: int = 15) -> MemoryContextOutput:
        async with self._db_guard():
            session = self.db.get_session(session_id)
            if session is None:
                raise ValueError(f"Sesión no encontrada: {session_id}")

            limit = min(limit, 30)
            msgs, total = self.db.get_session_messages_range(
                session_id, offset=offset, limit=limit
            )

            summary = session.summary
            if not summary:
                session_threads = self.db.get_threads_for_session(session_id)
                if session_threads:
                    parts = []
                    for t in session_threads[:3]:
                        state = (t.get("current_state") or "")[:150]
                        parts.append(f"[{t.get('title', t.get('key', ''))}] {state}")
                    summary = "\n".join(parts) if parts else None

            return MemoryContextOutput(
                session_summary=summary,
                client=session.client,
                conversation_id=session.conversation_id,
                messages=[
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                    for m in msgs
                ],
                total=total,
                offset=offset,
                limit=limit,
            )



    # ── tool: memory_rebuild_vectors ────────────────────────────

    async def memory_rebuild_vectors(
        self,
        dry_run: bool = True,
        facts_only: bool = False,  # legacy param, ignored
        sessions_only: bool = False,
    ) -> MemoryRebuildVectorsOutput:
        """Wipe vec0 tables and regenerate embeddings (sessions + new model entities)."""
        async with self._db_guard():
            report = await rebuild_vectors(
                self.db,
                self.ollama,
                dry_run=dry_run,
                sessions_only=sessions_only,
            )
        return MemoryRebuildVectorsOutput(**report)

    # ── tool: session_list ──────────────────────────────────────

    async def session_list(self, limit: int = 15, status: str | None = None) -> list[dict]:
        async with self._db_guard():
            sessions = self.db.get_sessions(limit=limit, status=status)
            return [
                {
                    "id": s.id,
                    "status": s.status.value,
                    "client": s.client,
                    "conversation_id": s.conversation_id,
                    "summary": s.summary,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                }
                for s in sessions
            ]

    # ── tool: memory_expand_thread ──────────────────────────────

    async def memory_expand_thread(self, thread_key: str, already_sent_echo_ids: list[int] = None) -> dict:
        from alambique.activation import ActivationEngine
        engine = ActivationEngine(self.db, self.ollama)
        return await engine.expand(thread_key, already_sent_echo_ids or [])
