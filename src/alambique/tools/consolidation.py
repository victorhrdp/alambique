"""Session consolidation: LLM thematic synthesis (threads, capsules, echoes) and DB apply."""

from __future__ import annotations

import json
import logging

from alambique.consolidator import ConsolidatorClient
from alambique.memory_config import (
    AGENT_NAME,
    INITIATIVE_MIN_PAYLOAD_LEN,
)
from alambique.models import (
    Consolidation,
    ConsolidationAction,
    Message,
)
from alambique.tools.text import messages_for_consolidation
from alambique.vector_store import upsert_embedding

logger = logging.getLogger("alambique.tools")


def _dumps_open_questions(open_questions: list | None) -> str | None:
    """Serialize open_questions as JSON with real UTF-8 (not \\uXXXX escapes).

    Historical bug: json.dumps default ensure_ascii=True stored Spanish as
    literal escapes in SQLite; the widget then showed '\\u00bfComo...' instead
    of '¿Cómo...'. Always write ensure_ascii=False.
    """
    if not open_questions:
        return None
    return json.dumps(open_questions, ensure_ascii=False)


def rewrite_open_questions_utf8(conn) -> int:
    """Re-serialize threads.open_questions with ensure_ascii=False. Returns rows fixed."""
    rows = conn.execute(
        "SELECT id, open_questions FROM threads WHERE open_questions IS NOT NULL"
    ).fetchall()
    fixed = 0
    for row in rows:
        raw = row["open_questions"] if hasattr(row, "keys") else row[1]
        tid = row["id"] if hasattr(row, "keys") else row[0]
        if not raw or not isinstance(raw, str):
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, list):
            continue
        new = json.dumps(parsed, ensure_ascii=False)
        if new != raw:
            conn.execute(
                "UPDATE threads SET open_questions = ? WHERE id = ?",
                (new, tid),
            )
            fixed += 1
    return fixed


class ConsolidationMixin:
    async def _consolidate_session(self, session, light: bool = False) -> None:
        """Run consolidation on a session.

        light: if True, skip the expensive local embedding + vector search for similar threads
        (saves one bge-m3 call). Still provides recent high-salience threads to the consolidator.
        """
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
            # No legacy summary. El estado quedará reflejado en threads (o vacío si no hay nada).
            async with self._db_guard():
                self.db.conn.execute("UPDATE sessions SET consolidated = 1 WHERE id = ?", (session.id,))
                self.db.conn.commit()
            return

        consolidation_msgs = messages_for_consolidation(messages)
        if not consolidation_msgs:
            # No legacy summary.
            async with self._db_guard():
                self.db.conn.execute("UPDATE sessions SET consolidated = 1 WHERE id = ?", (session.id,))
                self.db.conn.commit()
            return

        # Prepare dense context for new thematic consolidation (threads + capsules)
        threads_text = "(ninguno relevante)"
        capsules_text = "(ninguna)"
        try:
            async with self._db_guard():
                # Get recent/salient threads + (optionally) semantically similar ones.
                # light mode skips the Ollama embed + vec search to reduce CPU.
                recent_threads = self.db.get_high_salience_recent_threads(limit=15)
                similar_threads = []
                if not light:
                    try:
                        # Embed a short version of the conversation to find similar existing threads
                        conv_text = " ".join([m.content for m in consolidation_msgs[:5]])[:1500]
                        if conv_text:
                            emb = await self.ollama.embed(conv_text)
                            hits = self.db.vector_search_threads(emb, limit=10)
                            for h in hits:
                                t = h.get("thread", h) if isinstance(h, dict) else h
                                if t:
                                    similar_threads.append(t)
                    except Exception as e:
                        logger.warning("Similar threads search skipped due to error (light mode recommended if frequent): %s", e)

                all_threads = recent_threads + similar_threads
                # dedup by key
                seen = set()
                unique_threads = []
                for t in all_threads:
                    k = t.get("key") if isinstance(t, dict) else None
                    if k and k not in seen:
                        seen.add(k)
                        unique_threads.append(t)

                if unique_threads:
                    lines = []
                    for t in unique_threads[:25]:  # cap to not overwhelm prompt
                        state_snippet = (t.get('current_state','') or '')[:180].replace('\n', ' ')
                        desc = t.get('description','') or ''
                        desc_part = f" | description: {desc[:100]}" if desc else ""
                        oq = t.get('open_questions') or ''
                        oq_part = f" | open_questions: {oq[:80]}" if oq else ""
                        sal = t.get('salience', 0.5)
                        lines.append(f"- key={t.get('key')}: {t.get('title','')} {desc_part}{oq_part} | salience: {sal} | current_state: {state_snippet}")
                    threads_text = "\n".join(lines)
                cap = self.db.get_relevant_relationship_capsule()
                if cap:
                    capsules_text = cap[:600]
        except Exception:
            pass

        if not self.online:
            logger.warning("Offline: saltando consolidación de %s (se reintentará)", session.id)
            return

        try:
            response = await self.consolidator.consolidate(
                agent_name=AGENT_NAME,
                messages=consolidation_msgs,
                existing_threads=threads_text,
                existing_capsules=capsules_text,
            )
        except Exception as e:
            self._note_llm_outcome(False, str(e))
            logger.error("Fallo la llamada al consolidador: %s (se reintentará)", e)
            return

        await self._apply_consolidation(session, response)

    def _consolidation_db_phase(
        self, session, response
    ) -> list[tuple[str, int, str]]:
        """Apply mutations for new memory model under the DB lock; return pending embedding jobs."""
        embed_requests: list[tuple[str, int, str]] = []

        # New: handle threads, capsules, echoes first (core of the new model)
        for thread_item in getattr(response, 'threads', []):
            action = thread_item.get('action', 'create')
            key = thread_item.get('key')
            if action not in ('create', 'update', 'merge'):
                logger.warning(f"Consolidation {session.id}: invalid action {action} for thread {key or 'unknown'}, defaulting to update")
                action = 'update'
            if not key:
                logger.warning(f"Consolidation {session.id}: thread item missing key, skipping")
                continue

            # Stronger validation of LLM output for threads
            current_state = thread_item.get('current_state')
            title = thread_item.get('title')
            search_text = thread_item.get('search_text')
            tone_guidance = thread_item.get('tone_guidance')
            salience_raw = thread_item.get('salience', 0.5)

            if not current_state or not isinstance(current_state, str) or len(current_state.strip()) < 20:
                logger.warning(f"Consolidation {session.id}: thread {key} missing or too short current_state, skipping")
                continue
            if not title or not isinstance(title, str):
                logger.warning(f"Consolidation {session.id}: thread {key} missing title, using key as fallback")
                title = key
            if not search_text or not isinstance(search_text, str):
                logger.warning(f"Consolidation {session.id}: thread {key} missing search_text, using key")
                search_text = key
            if not tone_guidance or not isinstance(tone_guidance, str):
                logger.warning(f"Consolidation {session.id}: thread {key} missing tone_guidance, using empty")
                tone_guidance = ''
            try:
                salience = float(salience_raw)
                if not (0.0 <= salience <= 1.0):
                    raise ValueError
            except (ValueError, TypeError):
                logger.warning(f"Consolidation {session.id}: thread {key} invalid salience, defaulting to 0.5")
                salience = 0.5

            description = thread_item.get('description', '')
            open_questions = thread_item.get('open_questions', []) or []
            if not isinstance(open_questions, list):
                open_questions = []
            reason = thread_item.get('reason', '')

            # Additional checks for merge
            if action == 'merge':
                merged_from = thread_item.get('merged_from', []) or []
                if not isinstance(merged_from, list) or len(merged_from) == 0:
                    logger.warning(f"Consolidation {session.id}: merge for {key} without merged_from list")

            existing = self.db.get_thread_by_key(key)
            thread_id = None
            target_id = None
            # Existence wins over LLM action: "create" on an existing key used to
            # hit UNIQUE(threads.key) and abort the whole apply mid-flight.
            if not existing:
                thread_id = self.db.create_thread(
                    key=key,
                    title=title,
                    current_state=current_state,
                    tone_guidance=tone_guidance,
                    search_text=search_text,
                    salience=salience,
                    description=description,
                    open_questions=_dumps_open_questions(open_questions)
                )
                target_id = thread_id
                if action != 'create':
                    logger.info(
                        f"Consolidation {session.id}: action={action} for new key {key}, created thread"
                    )
            else:
                if action == 'create':
                    logger.warning(
                        f"Consolidation {session.id}: LLM said create for existing key {key}, updating instead"
                    )
                target_id = existing['id']
                self.db.update_thread(
                    key,
                    title=title,
                    current_state=current_state,
                    tone_guidance=tone_guidance,
                    search_text=search_text,
                    salience=salience,
                    description=description,
                    open_questions=_dumps_open_questions(open_questions)
                )
                thread_id = target_id

            if action == 'merge':
                logger.info(f"Consolidation {session.id}: merging into thread {key} (LLM combined state)")
                merged_from = thread_item.get('merged_from', []) or []
                for old_key in merged_from:
                    if old_key and old_key != key:
                        old = self.db.get_thread_by_key(old_key)
                        if old:
                            old_id = old['id']
                            # Drop participations on the old thread that would collide
                            # with UNIQUE(thread_id, session_id) on the survivor.
                            self.db.conn.execute(
                                """
                                DELETE FROM thread_participations
                                WHERE thread_id = ?
                                  AND session_id IN (
                                    SELECT session_id FROM thread_participations WHERE thread_id = ?
                                  )
                                """,
                                (old_id, target_id),
                            )
                            # Reassign remaining participations to the surviving thread
                            self.db.conn.execute(
                                "UPDATE thread_participations SET thread_id = ? WHERE thread_id = ?",
                                (target_id, old_id)
                            )
                            # Reassign echoes
                            self.db.conn.execute(
                                "UPDATE echoes SET thread_id = ? WHERE thread_id = ?",
                                (target_id, old_id)
                            )
                            # Mark old thread as merged
                            self.db.conn.execute(
                                "UPDATE threads SET status = 'merged', current_state = ?, last_active_at = datetime('now') WHERE id = ?",
                                (f"Fusionado en '{key}'. Ver el estado actual allí.", old_id)
                            )
                            self.db.conn.commit()
                            logger.info(f"Consolidation {session.id}: marked thread {old_key} as merged into {key}")

            if thread_id:
                # embed search_text
                embed_requests.append(("vec0_threads", thread_id, search_text))

                # Record participation so we know what this session contributed to the thread
                contribution = (reason or current_state or "Updated during consolidation")[:500]
                self.db.add_thread_participation(thread_id, session.id, contribution)

                # Record the thread action in consolidations table for complete audit trail
                if action == 'merge':
                    thread_action = ConsolidationAction.MERGE
                elif action == 'create':
                    thread_action = ConsolidationAction.CREATE
                else:
                    thread_action = ConsolidationAction.UPDATE
                c = Consolidation(
                    session_id=session.id,
                    action=thread_action,
                    thread_id=thread_id,
                    reason=reason or f"Thread {key} {action}",
                )
                self.db.insert_consolidation(c)

        for cap_item in getattr(response, 'relationship_capsules', []):
            scope = cap_item.get('scope')
            content = cap_item.get('content')
            if not scope or not isinstance(scope, str):
                logger.warning(f"Consolidation {session.id}: capsule missing or invalid scope, skipping")
                continue
            if not content or not isinstance(content, str) or len(content.strip()) < 10:
                logger.warning(f"Consolidation {session.id}: capsule {scope} missing or too short content, skipping")
                continue
            cap_id = self.db.upsert_relationship_capsule(scope, content)
            # embed content
            embed_requests.append(("vec0_relationship_capsules", cap_id, content))

            # Audit record for capsule
            c = Consolidation(
                session_id=session.id,
                action=ConsolidationAction.UPDATE,
                capsule_scope=scope,
                reason=f"Capsule {scope} updated",
            )
            self.db.insert_consolidation(c)

        for echo_item in getattr(response, 'echoes', []):
            thread_key = echo_item.get('thread_key')
            content = echo_item.get('content')
            context = echo_item.get('context', '')
            salience = echo_item.get('salience', 0.6)
            emotional_valence = echo_item.get('emotional_valence')
            if not content or not isinstance(content, str) or len(content.strip()) < 5:
                logger.warning(f"Consolidation {session.id}: echo missing or too short content, skipping")
                continue
            try:
                salience = float(salience)
                if not (0.0 <= salience <= 1.0):
                    salience = 0.6
            except (ValueError, TypeError):
                salience = 0.6
            if not isinstance(context, str):
                context = ''
            if emotional_valence is not None:
                try:
                    emotional_valence = float(emotional_valence)
                    if not (-1.0 <= emotional_valence <= 1.0):
                        emotional_valence = None
                except (ValueError, TypeError):
                    emotional_valence = None
            thread_id = None
            if thread_key:
                t = self.db.get_thread_by_key(thread_key)
                if t:
                    thread_id = t['id']
            self.db.conn.execute(
                "INSERT INTO echoes (thread_id, content, context, salience, emotional_valence) VALUES (?, ?, ?, ?, ?)",
                (thread_id, content, context, salience, emotional_valence)
            )
            self.db.conn.commit()
            echo_id = self.db.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            embed_requests.append(("vec0_echoes", echo_id, content))

            # Audit record for echo
            c = Consolidation(
                session_id=session.id,
                action=ConsolidationAction.CREATE,
                echo_id=echo_id,
                reason=f"Echo added for thread {thread_key or 'general'}",
            )
            self.db.insert_consolidation(c)

        # Lucy initiative MVP: single future-oriented slot (optional)
        initiative = getattr(response, "lucy_initiative", None)
        if isinstance(initiative, dict):
            payload = initiative.get("prompt_payload")
            if (
                isinstance(payload, str)
                and len(payload.strip()) >= INITIATIVE_MIN_PAYLOAD_LEN
            ):
                thread_key = initiative.get("thread_key")
                if thread_key is not None and not isinstance(thread_key, str):
                    thread_key = None
                reason = initiative.get("reason") or "Lucy initiative from consolidation"
                initiative_id = self.db.create_initiative(
                    payload.strip(),
                    thread_key=thread_key,
                    source_session_id=session.id,
                )
                c = Consolidation(
                    session_id=session.id,
                    action=ConsolidationAction.CREATE,
                    reason=f"Initiative #{initiative_id}: {reason}"[:500],
                    new_value=payload.strip()[:500],
                )
                self.db.insert_consolidation(c)
                logger.info(
                    "Consolidation %s: created initiative #%s",
                    session.id,
                    initiative_id,
                )
            else:
                logger.warning(
                    "Consolidation %s: lucy_initiative missing/short prompt_payload, skipped",
                    session.id,
                )

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
                        upsert_embedding(self.db.conn, table, entity_id, emb)
            except Exception as e:
                logger.warning("Batch embedding falló: %s", e)

        # Mark as consolidated (no more legacy summary)
        async with self._db_guard():
            self.db.conn.execute(
                "UPDATE sessions SET consolidated = 1 WHERE id = ?",
                (session.id,)
            )
            self.db.conn.commit()
            # Legacy cleanup_stale_embeddings removed; do targeted orphan cleanup instead (non-fatal)
            try:
                orphans = (
                    self.db.cleanup_orphan_thread_embeddings()
                    + self.db.cleanup_orphan_capsule_embeddings()
                    + self.db.cleanup_orphan_echo_embeddings()
                )
                if orphans:
                    logger.info("Limpieza post-consolidación: %d embeddings huérfanos limpiados", orphans)
            except Exception:
                pass

        logger.info("Consolidación completada para sesión %s", session.id)

    async def consolidate_session(self, session_id: str, force: bool = False, light: bool = False) -> dict:
        """Fuerza (o re-ejecuta) la consolidación de una sesión específica.

        Útil cuando session_end falló, hubo salida abrupta, o la consolidación
        previa no se completó. Sincroniza transcript si hay binding, cierra si
        está abierta, resetea flag si force=True, y ejecuta el consolidator.

        light=True: modo ligero que salta la búsqueda de threads similares por embedding
        (evita una llamada cara a Ollama bge-m3). Útil si la consolidación está pegando
        mucho la CPU. El consolidator aún recibe threads recientes de alta salience.

        Devuelve {"status": "ok", "session_id": ..., "consolidated": true} o error.
        """
        async with self._db_guard():
            session = self.db.get_session(session_id)
            if not session:
                return {"status": "error", "message": f"Sesión no encontrada: {session_id}"}

        original_status = session.status
        if session.status == "open":
            await self._close_session(
                session_id,
                SessionStatus.CLOSED,
                conversation_id=session.conversation_id,
                client=session.client,
            )
            async with self._db_guard():
                session = self.db.get_session(session_id)

        if session.consolidated and not force:
            return {
                "status": "ok",
                "message": "La sesión ya estaba consolidada (usa force=true para re-consolidar)",
                "session_id": session_id,
            }

        if force or not session.consolidated:
            async with self._db_guard():
                self.db.conn.execute(
                    "UPDATE sessions SET consolidated = 0 WHERE id = ?",
                    (session_id,),
                )
                self.db.conn.commit()
            # refresh
            async with self._db_guard():
                session = self.db.get_session(session_id)

        try:
            await self._consolidate_session(session, light=light)
            return {
                "status": "ok",
                "session_id": session_id,
                "consolidated": True,
                "was_open": original_status == "open",
                "forced": force,
                "light": light,
            }
        except Exception as e:
            logger.error("Fallo consolidación forzada para %s: %s", session_id, e)
            return {"status": "error", "message": str(e), "session_id": session_id}
