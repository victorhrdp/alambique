"""
ActivationEngine for Alambique memory redesign.
Implements the activation and expansion logic from the design.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import json

from alambique.database import Database
from alambique.ollama_client import OllamaClient


class ActivationEngine:
    """Core engine for context activation and thread expansion."""

    def __init__(self, db: Database, ollama: OllamaClient):
        self.db = db
        self.ollama = ollama

    async def activate(self, initial_user_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform activation for session start.
        Returns initial_context string and list of active thread keys.
        """
        # Get relationship capsule
        capsule = self.db.get_relevant_relationship_capsule() or ""

        # Tier 1: high salience + recent threads (cheap)
        tier1 = self.db.get_high_salience_recent_threads(limit=5)

        # Tier 2: vector if user text provided (controlled)
        candidates = list(tier1)
        vector_hits = []
        if initial_user_text:
            try:
                emb = await self.ollama.embed(initial_user_text)
                vector_hits = self.db.vector_search_threads(emb, limit=8)
                # merge unique by key or id
                seen = {t.get("key") or t.get("id") for t in tier1}
                for h in vector_hits:
                    t = h.get("thread", h) if isinstance(h, dict) else h
                    kid = t.get("key") or t.get("id")
                    if kid not in seen:
                        candidates.append(t)
                        seen.add(kid)
            except Exception:
                # Ollama down or embed fail -> just use tier1
                vector_hits = []
                candidates = list(tier1)
        threads = self._rerank(candidates, vector_hits)[:3] if vector_hits else tier1[:3]

        # Load a few high-value echoes for the selected threads
        echoes = []
        for t in threads:
            echoes.extend(self.db.get_top_echoes_for_thread(t["id"], limit=2))

        # Assemble
        context = self._assemble_context(capsule, threads, echoes)
        context = self._truncate_to_tokens(context, 1300)

        return {
            "initial_context": context,
            "active_thread_keys": [t["key"] for t in threads],
        }

    async def expand(self, thread_key: str, already_sent_echo_ids: List[int] = None) -> Dict[str, Any]:
        """
        Returns a richer expansion block for one specific thread.
        Budget: ~1000 tokens.
        """
        thread = self.db.get_thread_by_key(thread_key)
        if not thread:
            return {"error": "Thread no encontrado"}

        full_state = thread["current_state"]
        extra_echoes = self.db.get_top_echoes_for_thread(
            thread["id"], limit=6, exclude_ids=already_sent_echo_ids or []
        )
        parts = self.db.get_recent_participations(thread["id"], limit=3)

        block = self._build_expansion_block(thread, full_state, extra_echoes, parts)
        block = self._truncate_to_tokens(block, 1000)

        return {
            "thread_key": thread_key,
            "expanded_block": block,
            "has_more": bool(extra_echoes or parts),
        }

    # --- Internal helpers ---

    def _rerank(self, candidates: list, vector_hits: list = None) -> list:
        """Hybrid re-rank: vector sim (if avail) + salience + recency + access."""
        vector_hits = vector_hits or []
        # map key/id -> distance
        dist_map = {}
        for item in vector_hits:
            t = item.get("thread", item) if isinstance(item, dict) else item
            dist = item.get("distance", 0.0) if isinstance(item, dict) else 0.0
            kid = (t.get("key") if isinstance(t, dict) else None) or (t.get("id") if isinstance(t, dict) else t)
            if kid is not None:
                dist_map[kid] = dist

        scored = []
        for t in candidates:
            kid = t.get("key") or t.get("id")
            dist = dist_map.get(kid, 999.0)
            sim = 1.0 / (1.0 + dist) if dist < 900 else 0.3  # base sim for tier1 only

            # recency (simple linear decay)
            days = 0
            la = t.get("last_active_at")
            if la:
                try:
                    if isinstance(la, str):
                        la = datetime.fromisoformat(la.replace("Z", "+00:00"))
                    delta = datetime.now() - la
                    days = delta.days
                except:
                    days = 30
            recency = max(0.0, 1.0 - (days / 30.0))

            access = min(1.0, (t.get("access_count", 0) ** 0.5) / 5.0)

            score = (sim * 0.45) + (t.get("salience", 0.5) * 0.30) + (recency * 0.15) + (access * 0.10)
            scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]

    def _assemble_context(self, capsule: str, threads: list, echoes: list) -> str:
        parts = []
        if capsule:
            parts.append("=== RELACIÓN Y CARÁCTER ===\n" + capsule)
        if threads:
            parts.append("=== HILOS ACTIVOS ===")
            for t in threads:
                block = f"Key: {t['key']}\nEstado actual: {t.get('current_state', '')}\nTono recomendado: {t.get('tone_guidance', '')}"
                if t.get('description'):
                    block += f"\nDescripción: {t['description']}"
                oq = t.get('open_questions')
                if oq:
                    try:
                        oq_list = json.loads(oq) if isinstance(oq, str) else oq
                        if oq_list:
                            block += f"\nPreguntas abiertas: {', '.join(oq_list)}"
                    except:
                        pass
                parts.append(block)
        if echoes:
            parts.append("=== MATICES CLAVE ===")
            for e in echoes:
                line = "- " + e.get("content", "")
                ctx = e.get("context")
                if ctx:
                    line += f" (contexto: {ctx})"
                val = e.get("emotional_valence")
                if val is not None:
                    line += f" [valence: {val}]"
                parts.append(line)
        return "\n\n".join(parts)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        # Simple approximation (roughly 4 chars per token for Spanish/English mix)
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        # Try to cut at a clean paragraph or sentence boundary
        cut = text[:max_chars]
        for sep in ["\n\n", "\n", ". "]:
            idx = cut.rfind(sep)
            if idx > max_chars * 0.7:
                return cut[:idx].strip() + "\n[...]"
        return cut.strip() + "\n[...]"

    def _build_expansion_block(self, thread: dict, full_state: str, extra_echoes: list, parts: list) -> str:
        block = [f"=== EXPANSIÓN DEL HILO: {thread['key']} ==="]
        block.append("Estado actual completo:\n" + full_state)
        if thread.get('description'):
            block.append("Descripción del hilo:\n" + thread['description'])
        oq = thread.get('open_questions')
        if oq:
            try:
                oq_list = json.loads(oq) if isinstance(oq, str) else oq
                if oq_list:
                    block.append("Preguntas abiertas:\n- " + "\n- ".join(oq_list))
            except:
                pass
        if extra_echoes:
            block.append("Ecos adicionales:")
            for e in extra_echoes:
                line = "- " + e.get("content", "")
                ctx = e.get("context")
                if ctx:
                    line += f" (contexto: {ctx})"
                val = e.get("emotional_valence")
                if val is not None:
                    line += f" [valence: {val}]"
                block.append(line)
        if parts:
            block.append("Participaciones clave:")
            for p in parts:
                block.append("- " + p.get("contribution_summary", ""))
        return "\n\n".join(block)


# Convenience if someone imports directly
def get_activation_engine(db: Database, ollama: OllamaClient) -> ActivationEngine:
    return ActivationEngine(db, ollama)
