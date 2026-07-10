"""Memory data-quality helpers — deduplication, validation, embedding cleanup."""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

from alambique.memory_config import DEDUP_MAX_DISTANCE, DEDUP_SIMILARITY_THRESHOLD
from alambique.models import Fact, FactCategory

if TYPE_CHECKING:
    from alambique.database import Database

logger = logging.getLogger("alambique.maintenance")

_HEALTH_KEY_MARKERS = ("pain", "health", "dolor", "cuello", "neck", "enfermo")


def similarity_from_distance(distance: float) -> float:
    """Map vec0 KNN distance to a 0–1 similarity score (same formula as recall)."""
    return 1.0 / (1.0 + distance)


def parse_embedding_blob(blob: bytes) -> list[float]:
    """Decode a vec0 float32 embedding blob."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def validate_fact_classification(key: str, category: FactCategory) -> list[str]:
    """Light heuristics after consolidation — non-blocking warnings only."""
    warnings: list[str] = []
    key_lower = key.lower()
    if category == FactCategory.PERSONAL and any(m in key_lower for m in _HEALTH_KEY_MARKERS):
        warnings.append(
            f"classification_warning: key '{key}' looks temporal/health but category is personal"
        )
    return warnings


def _merge_fact_values(keeper: Fact, duplicate: Fact) -> str:
    """Pick the best combined value when merging near-duplicate facts."""
    a, b = keeper.value.strip(), duplicate.value.strip()
    if a == b:
        return a
    if a in b:
        return b
    if b in a:
        return a
    return f"{a}; {b}"


def _pick_keeper(fact_a: Fact, fact_b: Fact) -> tuple[Fact, Fact]:
    """Return (keeper, duplicate) using confidence, access, then lower id."""
    score_a = (fact_a.confidence, fact_a.access_count, -fact_a.id)
    score_b = (fact_b.confidence, fact_b.access_count, -fact_b.id)
    if score_a >= score_b:
        return fact_a, fact_b
    return fact_b, fact_a


def find_duplicate_pairs(db: Database) -> list[tuple[Fact, Fact, float]]:
    """Find active fact pairs with similarity >= threshold."""
    facts = db.get_active_embedded_facts()
    pairs: list[tuple[Fact, Fact, float]] = []
    seen: set[tuple[int, int]] = set()

    for fact in facts:
        embedding = db.get_fact_embedding(fact.id)
        if embedding is None:
            continue
        neighbors = db.search_similar_facts(
            embedding,
            limit=20,
            max_distance=DEDUP_MAX_DISTANCE,
        )
        for neighbor in neighbors:
            other_id = neighbor["id"]
            if other_id == fact.id:
                continue
            pair_key = (min(fact.id, other_id), max(fact.id, other_id))
            if pair_key in seen:
                continue
            other = db.get_fact(other_id)
            if other is None or not other.is_active():
                continue
            sim = similarity_from_distance(neighbor["distance"])
            if sim >= DEDUP_SIMILARITY_THRESHOLD:
                seen.add(pair_key)
                pairs.append((fact, other, sim))

    return pairs


def merge_duplicate_pair(db: Database, fact_a: Fact, fact_b: Fact) -> int:
    """Merge two facts in-place; returns the surviving fact id."""
    keeper, duplicate = _pick_keeper(fact_a, fact_b)
    merged_value = _merge_fact_values(keeper, duplicate)
    merged_confidence = max(keeper.confidence, duplicate.confidence)
    db.update_fact(keeper.id, merged_value, merged_confidence)
    db.forget_fact(duplicate.id)
    logger.info(
        "Merged duplicate facts %d ← %d",
        keeper.id,
        duplicate.id,
    )
    return keeper.id