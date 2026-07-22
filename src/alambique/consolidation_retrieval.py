"""Select which existing threads to show the consolidator LLM.

Loncha B: stop dumping ~15 high-salience threads into every consolidation
prompt. Prefer semantic neighbors of *this* session; keep a small recency
floor so brand-new work is still visible without re-listing the whole hall of fame.
"""

from __future__ import annotations

from typing import Any, Iterable

from alambique.memory_config import (
    CONSOLIDATION_LIST_CAP,
    CONSOLIDATION_MAX_VECTOR_DISTANCE,
    CONSOLIDATION_SIMILAR_LIMIT,
)


def merge_thread_candidates(
    *,
    similar: Iterable[dict],
    recent: Iterable[dict],
    list_cap: int = CONSOLIDATION_LIST_CAP,
) -> list[dict]:
    """Dedup by key. Similar (session-relevant) first, then recent floor.

    Order matters for the prompt: most relevant first, cap total lines.
    """
    out: list[dict] = []
    seen: set[str] = set()

    for source in (similar, recent):
        for t in source:
            if not isinstance(t, dict):
                continue
            key = t.get("key")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= list_cap:
                return out
    return out


def threads_from_vector_hits(
    hits: Iterable[Any],
    *,
    max_distance: float = CONSOLIDATION_MAX_VECTOR_DISTANCE,
    limit: int = CONSOLIDATION_SIMILAR_LIMIT,
) -> list[dict]:
    """Unwrap vector_search_threads hits; drop weak neighbors."""
    out: list[dict] = []
    for h in hits:
        if not isinstance(h, dict):
            continue
        dist = h.get("distance")
        if dist is not None:
            try:
                if float(dist) > max_distance:
                    continue
            except (TypeError, ValueError):
                pass
        t = h.get("thread", h)
        if isinstance(t, dict) and t.get("key"):
            out.append(t)
        if len(out) >= limit:
            break
    return out


def format_threads_for_prompt(threads: list[dict]) -> str:
    """Compact one-line-per-thread block for CONSOLIDATION_PROMPT."""
    if not threads:
        return "(ninguno relevante)"
    lines: list[str] = []
    for t in threads:
        state_snippet = (t.get("current_state", "") or "")[:180].replace("\n", " ")
        desc = t.get("description", "") or ""
        desc_part = f" | description: {desc[:100]}" if desc else ""
        oq = t.get("open_questions") or ""
        oq_part = f" | open_questions: {oq[:80]}" if oq else ""
        sal = t.get("salience", 0.5)
        lines.append(
            f"- key={t.get('key')}: {t.get('title', '')}{desc_part}{oq_part} "
            f"| salience: {sal} | current_state: {state_snippet}"
        )
    return "\n".join(lines)


def select_threads_for_consolidation_prompt(
    *,
    similar_hits: list | None = None,
    recent_threads: list[dict] | None = None,
    list_cap: int = CONSOLIDATION_LIST_CAP,
    max_distance: float = CONSOLIDATION_MAX_VECTOR_DISTANCE,
    similar_limit: int = CONSOLIDATION_SIMILAR_LIMIT,
) -> list[dict]:
    """Pure selection used by consolidation (and tests)."""
    similar = threads_from_vector_hits(
        similar_hits or [],
        max_distance=max_distance,
        limit=similar_limit,
    )
    recent = list(recent_threads or [])
    return merge_thread_candidates(
        similar=similar,
        recent=recent,
        list_cap=list_cap,
    )


__all__ = [
    "format_threads_for_prompt",
    "merge_thread_candidates",
    "select_threads_for_consolidation_prompt",
    "threads_from_vector_hits",
]
