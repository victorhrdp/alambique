"""MCP tool implementations for Alambique."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from alambique.consolidator import ConsolidatorClient, get_api_key
from alambique.memory_config import (
    AGENT_NAME,
    CONSOLIDATION_CANDIDATE_POOL,
    CONSOLIDATION_TOP_K,
    RANK_ACCESS_CAP,
    RANK_WEIGHT_CONFIDENCE,
    RANK_WEIGHT_REINFORCEMENT,
    RANK_WEIGHT_SIMILARITY,
    RECALL_CANDIDATE_POOL,
    RECALL_THRESHOLD_DEFAULT,
    RECALL_THRESHOLD_PREFERENCE,
    RECALL_TOP_K,
)
from alambique.memory_maintenance import (
    find_duplicate_pairs,
    merge_duplicate_pair,
    validate_fact_classification,
)
from alambique.database import Database
from alambique.models import (
    Consolidation,
    ConsolidationAction,
    ConsolidationFactItem,
    Fact,
    FactCategory,
    MemoryCleanupOutput,
    MemoryContextOutput,
    MemoryDeduplicateOutput,
    MemoryReembedOutput,
    MemoryHealthCheck,
    MemoryHealthOutput,
    MemoryRecallOutput,
    MemoryStatusOutput,
    Message,
    SessionEndOutput,
    SessionStartOutput,
    SessionStatus,
)
from alambique.ollama_client import OllamaClient
from alambique.recall import RecallClient

logger = logging.getLogger("alambique.tools")


def consolidation_search_text(messages: list[Message], *, max_chars: int = 8000) -> str:
    """Build embedding text from a session for consolidation fact retrieval."""
    lines = [f"{m.role}: {m.content}" for m in messages]
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def messages_for_consolidation(messages: list[Message]) -> list[Message]:
    """Messages eligible for fact extraction.

    Keeps user turns and assistant replies to the user. Drops LucyGame
    auto-commentary ([Auto] ...) — it stays in the session for in-game
    context but must not pollute long-term facts or session summaries.
    """
    eligible: list[Message] = []
    for m in messages:
        if m.role not in ("user", "assistant"):
            continue
        if m.role == "assistant" and m.content.lstrip().startswith("[Auto]"):
            continue
        eligible.append(m)
    return eligible


class ToolHandler:
    """Handles all MCP tool calls."""

    def __init__(self, db: Database, ollama: OllamaClient, api_key: str | None = None) -> None:
        self.db = db
        self.ollama = ollama
        self._api_key = api_key
        self._consolidator_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        self._consolidator: Optional[ConsolidatorClient] = None
        self._consolidation_warnings: list[str] = []
        self._recall: Optional[RecallClient] = None
        self._db_lock: Optional[asyncio.Lock] = None

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
            self._consolidator = ConsolidatorClient(self._api_key)
        return self._consolidator

    @property
    def recall(self) -> RecallClient:
        if self._recall is None:
            if self._api_key is None:
                raise RuntimeError("API key no disponible")
            self._recall = RecallClient(self._api_key)
        return self._recall

    @property
    def online(self) -> bool:
        return self._api_key is not None

    # ── lifecycle ──────────────────────────────────────────────

    async def start_background_tasks(self) -> None:
        """Start the consolidation worker and watchdog."""
        self._consolidator_task = asyncio.create_task(self._consolidation_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())
        logger.info("Background tasks started.")

    async def stop_background_tasks(self) -> None:
        if self._consolidator_task:
            self._consolidator_task.cancel()
        if self._watchdog_task:
            self._watchdog_task.cancel()
        if self._consolidator:
            await self._consolidator.close()
        if self._recall:
            await self._recall.close()

    async def shutdown_open_sessions(self) -> None:
        """Sync and close open sessions before daemon shutdown."""
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

    async def _consolidation_loop(self) -> None:
        """Process pending consolidations sequentially."""
        while True:
            try:
                async with self._db_guard():
                    sessions = self.db.get_pending_consolidations()
                if not sessions:
                    await asyncio.sleep(5)
                    continue

                if not self.online:
                    # Retry loading API key dynamically
                    api_key = get_api_key()
                    if api_key:
                        self._api_key = api_key
                        logger.info("API key cargada dinámicamente. Pasando a modo online.")
                    else:
                        logger.warning("Modo offline: posponiendo consolidación (reintento en 30s)")
                        await asyncio.sleep(30)
                        continue

                session = sessions[0]
                logger.info("Consolidando sesión %s...", session.id)
                await self._consolidate_session(session)
                await asyncio.sleep(2)  # Small break between consolidations
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Error en consolidación: %s", e)
                await asyncio.sleep(10)

    async def _watchdog_loop(self) -> None:
        """Detect stale sessions, sync transcript if bound, then mark truncated."""
        while True:
            try:
                async with self._db_guard():
                    stale = self.db.find_stale_sessions(timeout_minutes=30)
                for s in stale:
                    logger.info("Watchdog: cerrando sesión obsoleta %s (truncated)", s.id)
                    await self._close_session(
                        s.id,
                        SessionStatus.TRUNCATED,
                        conversation_id=s.conversation_id,
                        client=s.client,
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("Error en watchdog: %s", e)
            await asyncio.sleep(60)

    async def _consolidate_session(self, session) -> None:
        """Run consolidation on a session."""
        async with self._db_guard():
            messages = self.db.get_session_messages(session.id)

        if not messages and session.client and session.conversation_id:
            await self._sync_session_transcript(
                session.id,
                session.conversation_id,
                session.client,
            )
            async with self._db_guard():
                messages = self.db.get_session_messages(session.id)

        if not messages:
            async with self._db_guard():
                if session.client and session.conversation_id:
                    self.db.set_session_summary(
                        session.id, "(transcript vacío tras sync)"
                    )
                else:
                    self.db.set_session_summary(session.id, "(sesión vacía)")
            return

        consolidation_msgs = messages_for_consolidation(messages)
        if not consolidation_msgs:
            async with self._db_guard():
                if session.client and session.conversation_id:
                    self.db.set_session_summary(
                        session.id, "(sin mensajes consolidables tras sync)"
                    )
                else:
                    self.db.set_session_summary(session.id, "(sin mensajes consolidables)")
            return

        existing_facts, fact_warnings = await self._facts_for_consolidation(consolidation_msgs)
        self._consolidation_warnings.extend(fact_warnings)

        if not self.online:
            logger.warning("Offline: saltando consolidación de %s (se reintentará)", session.id)
            return

        try:
            response = await self.consolidator.consolidate(
                agent_name=AGENT_NAME,
                messages=consolidation_msgs,
                existing_facts=existing_facts,
            )
        except Exception as e:
            logger.error("Fallo la llamada al consolidador: %s (se reintentará)", e)
            return

        await self._apply_consolidation(session, response)

    async def _facts_for_consolidation(
        self, messages: list[Message]
    ) -> tuple[list[Fact], list[str]]:
        """Retrieve existing facts semantically relevant to the session."""
        warnings: list[str] = []
        async with self._db_guard():
            personality = [
                f
                for f in self.db.get_facts(categories=(FactCategory.PERSONALITY,))
                if f.is_active()
            ]

        if not await self.ollama.health():
            warnings.append("consolidation_facts_offline_fallback")
            async with self._db_guard():
                recent = self.db.get_recent_facts(limit=CONSOLIDATION_TOP_K)
            return self._merge_consolidation_facts(recent, personality), warnings

        try:
            query = consolidation_search_text(messages)
            query_emb = await self.ollama.embed(query)
            async with self._db_guard():
                ranked, _raw_hits = self._rank_facts_from_embedding(
                    query_emb,
                    CONSOLIDATION_TOP_K,
                    candidate_pool=CONSOLIDATION_CANDIDATE_POOL,
                )
            if ranked:
                return self._merge_consolidation_facts(ranked, personality), warnings
            warnings.append("consolidation_no_vector_hits")
        except Exception as e:
            logger.warning("Búsqueda vectorial de consolidación falló: %s", e)
            warnings.append("consolidation_facts_vector_fallback")

        async with self._db_guard():
            recent = self.db.get_recent_facts(limit=CONSOLIDATION_TOP_K)
        return self._merge_consolidation_facts(recent, personality), warnings

    @staticmethod
    def _merge_consolidation_facts(
        primary: list[Fact], extra: list[Fact]
    ) -> list[Fact]:
        seen = {f.id for f in primary}
        merged = list(primary)
        for fact in extra:
            if fact.id not in seen:
                merged.append(fact)
                seen.add(fact.id)
        return merged

    def _rank_facts_from_embedding(
        self,
        query_emb: list[float],
        top_k: int,
        *,
        candidate_pool: int = RECALL_CANDIDATE_POOL,
        record_access: bool = False,
    ) -> tuple[list[Fact], int]:
        fact_rows = self._vector_search(
            "vec0_facts",
            query_emb,
            limit=candidate_pool,
            active_facts_only=True,
        )

        candidates: list[tuple[float, Fact]] = []
        for row in fact_rows:
            fid = row["id"]
            fact = self.db.get_fact(fid)
            if fact is None or not fact.is_active():
                continue
            threshold = (
                RECALL_THRESHOLD_PREFERENCE
                if fact.category in (FactCategory.PREFERENCE, FactCategory.STATE)
                else RECALL_THRESHOLD_DEFAULT
            )
            if fact.confidence < threshold:
                continue
            similarity = 1.0 / (1.0 + row["distance"])
            reinforcement = min(1.0, fact.access_count / float(RANK_ACCESS_CAP))
            score = (
                (similarity * RANK_WEIGHT_SIMILARITY)
                + (fact.confidence * RANK_WEIGHT_CONFIDENCE)
                + (reinforcement * RANK_WEIGHT_REINFORCEMENT)
            )
            candidates.append((score, fact))

        candidates.sort(key=lambda item: item[0], reverse=True)
        facts: list[Fact] = []
        for _score, fact in candidates[:top_k]:
            if record_access:
                self.db.record_fact_access(fact.id)
            facts.append(fact)
        return facts, len(fact_rows)

    def _consolidation_db_phase(
        self, session, response
    ) -> list[tuple[str, int, str]]:
        """Apply fact mutations under the DB lock; return pending embedding jobs."""
        embed_requests: list[tuple[str, int, str]] = []

        for item in response.facts:
            if item.action != ConsolidationAction.DISCARD:
                for warning in validate_fact_classification(item.key, item.category):
                    logger.warning("Consolidación %s: %s", session.id, warning)
                    self._consolidation_warnings.append(warning)

            action = item.action
            fact_id = item.related_fact_id
            prev_val = None
            embed_text = None

            if action in (ConsolidationAction.UPDATE, ConsolidationAction.MERGE, ConsolidationAction.CONTRADICT):
                if fact_id:
                    prev = self.db.get_fact(fact_id)
                    prev_val = prev.value if prev else None

            if action == ConsolidationAction.CREATE:
                existing = self.db.get_fact_by_key(item.key)
                if existing:
                    self.db.update_fact(
                        existing.id,
                        item.value,
                        item.confidence,
                        category=item.category,
                        ttl=item.ttl,
                    )
                    fact_id = existing.id
                    embed_text = item.value
                else:
                    f = Fact(
                        key=item.key,
                        value=item.value,
                        category=item.category,
                        ttl=item.ttl,
                        confidence=item.confidence,
                    )
                    new_id = self.db.insert_fact(f)
                    fact_id = new_id
                    embed_text = item.value

            elif action == ConsolidationAction.UPDATE:
                if fact_id:
                    self.db.update_fact(
                        fact_id,
                        item.value,
                        item.confidence,
                        category=item.category,
                        ttl=item.ttl,
                    )
                    embed_text = item.value

            elif action == ConsolidationAction.MERGE:
                if fact_id:
                    prev = self.db.get_fact(fact_id)
                    if prev:
                        merged = f"{prev.value}; {item.value}"
                        self.db.update_fact(fact_id, merged, max(prev.confidence, item.confidence))
                        embed_text = merged

            elif action == ConsolidationAction.CONTRADICT:
                key = self._allocate_contradiction_key(item.key)
                f = Fact(
                    key=key,
                    value=item.value,
                    category=item.category,
                    ttl=item.ttl,
                    confidence=item.confidence,
                )
                new_id = self.db.insert_fact(f)
                fact_id = new_id
                embed_text = item.value

            elif action == ConsolidationAction.DISCARD:
                pass

            if embed_text is not None:
                embed_requests.append(("vec0_facts", fact_id, embed_text))

            # Record consolidation
            c = Consolidation(
                session_id=session.id,
                action=action,
                fact_id=fact_id,
                previous_value=prev_val,
                new_value=item.value,
                reason=item.reason,
            )
            self.db.insert_consolidation(c)

        return embed_requests

    async def _apply_consolidation(self, session, response) -> None:
        """Apply consolidation: DB mutations under lock, embeddings outside lock."""
        async with self._db_guard():
            embed_requests = self._consolidation_db_phase(session, response)

        if embed_requests:
            try:
                texts = [r[2] for r in embed_requests]
                embeddings = await self.ollama.embed_batch(texts)
                async with self._db_guard():
                    for (table, entity_id, text), emb in zip(embed_requests, embeddings):
                        _upsert_embedding(self.db.conn, table, entity_id, emb)
            except Exception as e:
                logger.warning("Batch embedding falló: %s", e)

        summary = response.session_summary
        async with self._db_guard():
            self.db.set_session_summary(session.id, summary)
        try:
            emb = await self.ollama.embed(summary)
            async with self._db_guard():
                _upsert_embedding(self.db.conn, "vec0_sessions", session.id, emb)
        except Exception as e:
            logger.warning("Embedding falló para summary de sesión %s: %s", session.id, e)

        async with self._db_guard():
            stale_removed = self.db.cleanup_stale_embeddings()
        if stale_removed:
            logger.info("Limpieza post-consolidación: %d embeddings huérfanos", stale_removed)

        logger.info("Consolidación completada para sesión %s: %d facts", session.id, len(response.facts))

    # ── tool: session_start ─────────────────────────────────────

    def _allocate_contradiction_key(self, base_key: str) -> str:
        """Reserve a unique key when contradict preserves both facts."""
        if self.db.get_fact_by_key(base_key) is None:
            return base_key
        candidate = f"{base_key}__alt"
        n = 2
        while self.db.get_fact_by_key(candidate):
            candidate = f"{base_key}__alt{n}"
            n += 1
        return candidate

    def _seed_persona_if_needed(self, persona_seed: str | None) -> None:
        """Persist initial personality when we have no traits yet."""
        seed = (persona_seed or "").strip()
        if not seed:
            return
        existing = self.db.get_facts(categories=(FactCategory.PERSONALITY,))
        if existing:
            return
        self.db.insert_fact(
            Fact(
                key="persona_seed",
                value=seed,
                category=FactCategory.PERSONALITY,
                confidence=1.0,
            )
        )
        logger.info("Persona sembrada.")

    async def _compose_session_persona(self) -> tuple[str | None, list[str]]:
        warnings: list[str] = []
        async with self._db_guard():
            traits = self.db.get_facts(categories=(FactCategory.PERSONALITY,))
            states = self.db.get_facts(categories=(FactCategory.STATE,))
        traits = [f for f in traits if f.confidence >= RECALL_THRESHOLD_DEFAULT]
        states = [
            f
            for f in states
            if f.is_active() and f.confidence >= RECALL_THRESHOLD_DEFAULT
        ]

        if not traits and not states:
            return None, warnings

        if self.online:
            try:
                return await self.recall.compose_personality(traits, states), warnings
            except Exception as e:
                logger.warning("Error componiendo personalidad: %s", e)
                warnings.append("persona_llm_failed")

        if traits:
            if not self.online:
                warnings.append("persona_offline_fallback")
            elif "persona_llm_failed" in warnings:
                warnings.append("persona_trait_fallback")
            return traits[0].value, warnings

        return None, warnings

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
            from alambique.transcripts.grok_cli import resolve_grok_session_id

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
            resolved = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
            if not resolved:
                warnings.append("antigravity_conversation_missing")
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

        session_reused = False
        async with self._db_guard():
            already_existed = len(self.db.get_all_sessions()) > 0
            self._seed_persona_if_needed(persona_seed)

            if bound_client and bound_conversation_id:
                existing = self.db.get_open_session_by_binding(
                    bound_client, bound_conversation_id
                )
                if existing:
                    session = existing
                    session_reused = True
                    warnings.append("session_reused")
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
        )
        if not conv_id and effective_client == "grok":
            from alambique.transcripts.grok_cli import resolve_grok_session_id

            conv_id, _ = resolve_grok_session_id()
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
        return SessionEndOutput()

    # ── tool: memory_recall ─────────────────────────────────────

    async def memory_recall(self, query: str) -> MemoryRecallOutput:
        facts_out: list[dict] = []
        sessions_out: list[dict] = []
        warnings: list[str] = []
        used_fallback = False
        summary_llm_used = False

        if not await self.ollama.health():
            warnings.append("ollama_unavailable")

        try:
            query_emb = await self.ollama.embed(query)

            async with self._db_guard():
                ranked, raw_hits = self._rank_facts_from_embedding(
                    query_emb,
                    RECALL_TOP_K,
                    record_access=True,
                )
                if raw_hits and not ranked:
                    warnings.append("no_candidates_after_filter")

                for f in ranked:
                    facts_out.append({
                        "id": f.id,
                        "key": f.key,
                        "value": f.value,
                        "category": f.category.value,
                        "confidence": f.confidence,
                    })

                session_rows = self._vector_search("vec0_sessions", query_emb, limit=5)
                for row in session_rows:
                    sid = row["session_id"] if "session_id" in row.keys() else row.get("id")
                    if sid:
                        s = (
                            self.db.get_session(str(sid))
                            if not isinstance(sid, str)
                            else self.db.get_session(sid)
                        )
                        if s and s.summary:
                            sessions_out.append({
                                "id": s.id,
                                "snippet": s.summary,
                            })
        except Exception as e:
            logger.warning("Vector search falló: %s. Usando solo facts directos.", e)
            warnings.append("vector_search_failed")
            used_fallback = True
            async with self._db_guard():
                facts = self.db.get_recent_facts(limit=RECALL_TOP_K)
            facts_out = [
                {"id": f.id, "key": f.key, "value": f.value, "category": f.category.value, "confidence": f.confidence}
                for f in facts
            ]

        summary = ""
        if self.online and (facts_out or sessions_out):
            try:
                summary = await self.recall.compose_summary(query, AGENT_NAME, facts_out, sessions_out)
                if summary:
                    summary_llm_used = True
            except Exception as e:
                logger.warning("Error en recall LLM: %s", e)
                warnings.append("summary_llm_failed")
        elif facts_out or sessions_out:
            warnings.append("summary_offline")

        if not summary and facts_out:
            summary = "Hechos encontrados: " + "; ".join(
                f"{f['key']}: {f['value']}" for f in facts_out[:5]
            )
            if not summary_llm_used:
                warnings.append("summary_fallback_generic")

        if used_fallback:
            warnings.append("recall_degraded_recent_facts")

        return MemoryRecallOutput(
            summary=summary,
            facts=facts_out,
            related_sessions=sessions_out,
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

            return MemoryContextOutput(
                session_summary=session.summary,
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



    # ── tool: memory_reembed ────────────────────────────────────

    async def memory_reembed(
        self,
        dry_run: bool = False,
        fact_ids: list[int] | None = None,
    ) -> MemoryReembedOutput:
        """Generate missing vec0 embeddings for active facts via Ollama."""
        warnings: list[str] = []
        async with self._db_guard():
            facts = self.db.get_facts_missing_embeddings()
            if fact_ids is not None:
                allowed = set(fact_ids)
                facts = [f for f in facts if f.id in allowed]

        missing_before = len(facts)
        target_ids = [f.id for f in facts]

        if missing_before == 0:
            return MemoryReembedOutput(
                dry_run=dry_run,
                missing_before=0,
                embedded=0,
                failed=0,
                fact_ids=[],
                warnings=warnings,
            )

        if not await self.ollama.health():
            warnings.append("ollama_unavailable")
            return MemoryReembedOutput(
                dry_run=dry_run,
                missing_before=missing_before,
                embedded=0,
                failed=missing_before,
                fact_ids=target_ids,
                warnings=warnings,
            )

        if dry_run:
            return MemoryReembedOutput(
                dry_run=True,
                missing_before=missing_before,
                embedded=0,
                failed=0,
                fact_ids=target_ids,
                warnings=warnings,
            )

        embedded = 0
        failed = 0
        batch_size = 32
        for offset in range(0, len(facts), batch_size):
            chunk = facts[offset : offset + batch_size]
            try:
                embeddings = await self.ollama.embed_batch([f.value for f in chunk])
                async with self._db_guard():
                    for fact, emb in zip(chunk, embeddings):
                        _upsert_embedding(self.db.conn, "vec0_facts", fact.id, emb)
                        embedded += 1
            except Exception as e:
                logger.warning("Batch re-embed falló (offset %d): %s", offset, e)
                failed += len(chunk)
                warnings.append(f"reembed_batch_failed:{offset}")

        return MemoryReembedOutput(
            dry_run=False,
            missing_before=missing_before,
            embedded=embedded,
            failed=failed,
            fact_ids=target_ids,
            warnings=warnings,
        )

    # ── tool: memory_deduplicate ────────────────────────────────

    async def memory_deduplicate(
        self,
        dry_run: bool = True,
    ) -> MemoryDeduplicateOutput:
        warnings: list[str] = []
        if not await self.ollama.health():
            warnings.append("ollama_unavailable")

        async with self._db_guard():
            pairs = find_duplicate_pairs(self.db)
        pair_reports: list[dict[str, object]] = []
        merged = 0

        for fact_a, fact_b, similarity in pairs:
            pair_reports.append({
                "fact_a": fact_a.id,
                "fact_b": fact_b.id,
                "similarity": round(similarity, 4),
                "key_a": fact_a.key,
                "key_b": fact_b.key,
            })
            if dry_run:
                continue
            if not fact_a.is_active() or not fact_b.is_active():
                continue
            async with self._db_guard():
                keeper_id = merge_duplicate_pair(self.db, fact_a, fact_b)
                keeper = self.db.get_fact(keeper_id)
            if keeper:
                try:
                    emb = await self.ollama.embed(keeper.value)
                    async with self._db_guard():
                        _update_embedding(self.db.conn, "vec0_facts", keeper_id, emb)
                except Exception as e:
                    logger.warning("Re-embed tras dedup falló para %s: %s", keeper_id, e)
                    warnings.append(f"reembed_failed:{keeper_id}")
            merged += 1

        return MemoryDeduplicateOutput(
            dry_run=dry_run,
            pairs_found=len(pairs),
            merged=merged,
            pairs=pair_reports,
            warnings=warnings,
        )

    # ── tool: memory_forget ─────────────────────────────────────

    async def memory_forget(self, fact_id: int | None = None, key: str | None = None) -> dict:
        async with self._db_guard():
            if fact_id:
                self.db.forget_fact(fact_id)
            elif key:
                f = self.db.get_fact_by_key(key)
                if f:
                    self.db.forget_fact(f.id)
                else:
                    raise ValueError(f"No se encontró fact: {key}")
            else:
                raise ValueError("Especifica fact_id o key")
        return {"deleted": True}

    # ── tool: memory_export ─────────────────────────────────────

    async def memory_export(self, format: str = "json") -> dict:
        async with self._db_guard():
            facts = self.db.get_all_facts()
            embedded_ids = self.db.get_embedded_fact_ids()
            sessions = [
                {
                    "id": s.id,
                    "status": s.status.value,
                    "client": s.client,
                    "conversation_id": s.conversation_id,
                    "summary": s.summary,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                }
                for s in self.db.get_all_sessions()
                if s.summary
            ]
            return {
                "facts": [
                    {
                        "id": f.id,
                        "key": f.key,
                        "value": f.value,
                        "category": f.category.value,
                        "confidence": f.confidence,
                        "ttl": f.ttl,
                        "is_expired": f.is_ttl_expired(),
                        "created_at": f.created_at.isoformat() if f.created_at else None,
                        "last_accessed": f.last_accessed.isoformat() if f.last_accessed else None,
                        "embedding_ok": f.id in embedded_ids,
                    }
                    for f in facts
                ],
                "sessions": sessions,
            }

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


    # ── tool: memory_status ─────────────────────────────────────

    async def memory_status(self) -> MemoryStatusOutput:
        async with self._db_guard():
            facts = self.db.get_all_facts()
            sessions = self.db.get_all_sessions()
            pending = self.db.count_pending_consolidations_db()
            last = self.db.last_consolidation_time()

            return MemoryStatusOutput(
                sessions=len(sessions),
                facts=len(facts),
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

            orphans = self.db.count_facts_missing_embeddings()
            stale = self.db.count_stale_embeddings()
            embedding_status = "ok" if orphans == 0 and stale == 0 else "warning"
            embedding_parts = []
            if orphans:
                embedding_parts.append(
                    f"{orphans} hechos activos sin embedding (usa memory_reembed)"
                )
            if stale:
                embedding_parts.append(f"{stale} embeddings huérfanos (confidence=0)")
            checks["embeddings"] = MemoryHealthCheck(
                status=embedding_status,
                detail="; ".join(embedding_parts) if embedding_parts else "Embeddings en orden",
            )
            if orphans > 0:
                warnings.append("embeddings_orphaned")
            if stale > 0:
                warnings.append("stale_embeddings")

            if self._consolidation_warnings:
                warnings.extend(self._consolidation_warnings[-5:])

            healthy = ollama_ok and api_ok and pending == 0 and orphans == 0 and stale == 0

        return MemoryHealthOutput(
            healthy=healthy,
            mode="online" if api_ok else "offline",
            checks=checks,
            warnings=warnings,
        )

    # ── vector helpers ──────────────────────────────────────────

    def _vector_search(
        self,
        table: str,
        embedding: list[float],
        limit: int = 10,
        *,
        active_facts_only: bool = False,
    ) -> list[dict]:
        """Perform KNN search on a vec0 virtual table."""
        emb_str = f"[{','.join(str(f) for f in embedding)}]"
        conn = self.db.conn

        try:
            if table == "vec0_facts" and active_facts_only:
                conditions = [
                    "f.confidence > 0",
                    "(f.ttl IS NULL OR (CAST(strftime('%s','now') AS INTEGER) "
                    "- CAST(strftime('%s', f.created_at) AS INTEGER)) < f.ttl)",
                ]
                where_extra = " AND " + " AND ".join(conditions)
                query = f"""
                    SELECT f.id AS rowid, v.distance
                    FROM vec0_facts v
                    JOIN facts f ON f.id = v.rowid
                    WHERE v.embedding MATCH '{emb_str}' AND k = {int(limit)}
                    {where_extra}
                    ORDER BY v.distance
                """
                rows = conn.execute(query).fetchall()
                return [
                    {"id": r["rowid"], "rowid": r["rowid"], "distance": r["distance"]}
                    for r in rows
                ]

            query = f"""
                SELECT rowid, distance
                FROM {table}
                WHERE embedding MATCH '{emb_str}' AND k = {int(limit)}
                ORDER BY distance
            """
            rows = conn.execute(query).fetchall()
            results = [dict(r) for r in rows]

            if table == "vec0_facts":
                return [
                    {"id": r["rowid"], "rowid": r["rowid"], "distance": r["distance"]}
                    for r in results
                ][:limit]
            if table == "vec0_sessions":
                return [
                    {"session_id": _rowid_to_session_id(r["rowid"]), "distance": r["distance"]}
                    for r in results
                ][:limit]

            return results[:limit] if results else []
        except Exception as e:
            logger.warning("Vector search error on %s: %s", table, e)
            return []


def _session_id_to_rowid(session_id: str) -> int:
    """Convert a session_id (sess_<hex>) to an integer rowid for vec0."""
    return int(session_id.split("_")[1], 16)


def _rowid_to_session_id(rowid: int) -> str:
    """Convert a vec0 rowid back to a session_id string."""
    return f"sess_{rowid:012x}"


def _embedding_rowid(table: str, entity_id) -> int:
    """Resolve vec0 rowid for a fact id or session id."""
    return _session_id_to_rowid(entity_id) if table == "vec0_sessions" else entity_id


def _has_embedding(conn, table: str, entity_id) -> bool:
    rowid = _embedding_rowid(table, entity_id)
    row = conn.execute(
        f"SELECT rowid FROM {table} WHERE rowid = ?", (rowid,)
    ).fetchone()
    return row is not None


def _insert_embedding(conn, table: str, entity_id, embedding: list[float], id_col: str = "id") -> None:
    """Insert an embedding into a vec0 virtual table."""
    emb_str = f"[{','.join(str(f) for f in embedding)}]"
    rowid = _embedding_rowid(table, entity_id)
    conn.execute(
        f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
        (rowid, emb_str),
    )
    conn.commit()


def _update_embedding(conn, table: str, entity_id, embedding: list[float]) -> None:
    """Update embedding in vec0 virtual table."""
    rowid = _embedding_rowid(table, entity_id)
    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    _insert_embedding(conn, table, entity_id, embedding)


def _upsert_embedding(conn, table: str, entity_id, embedding: list[float]) -> None:
    """Insert or replace an embedding (safe for consolidation UPDATE/MERGE)."""
    if _has_embedding(conn, table, entity_id):
        _update_embedding(conn, table, entity_id, embedding)
    else:
        _insert_embedding(conn, table, entity_id, embedding)
