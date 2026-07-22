"""Central configuration for Alambique memory tuning parameters.

All decay rates, recall thresholds, re-ranking weights, and TTL defaults
live here so behaviour can be adjusted without touching business logic.
"""

# Single-agent scope: all memory belongs to Lucy.
AGENT_NAME: str = "Lucy"

# Valid avatar expressions for session_update (KDE widget).
VALID_EXPRESSIONS: frozenset[str] = frozenset({
    "normal", "happy", "wink", "blushing", "thinking",
    "sad", "angry", "sleepy", "surprised",
})

# ── Ebbinghaus decay (λ in 1/seconds) ─────────────────────────────
# Formula: confidence * exp(-λ * elapsed_seconds), with spaced-reinforcement
# divisor log(access_count + 1). Approximate daily loss at access_count=0:
#   possessions ~0.1%/day, preference ~1%/day.

LAMBDA_POSSESSIONS: float = 1.1574e-8
LAMBDA_PREFERENCE: float = 1.1574e-7

FLOOR_PREFERENCE: float = 0.5
FLOOR_POSSESSIONS: float = 0.8

# ── TTL ───────────────────────────────────────────────────────────
# Default lifetime for temporal state facts (seconds). 86400 = 24 hours.

STATE_DEFAULT_TTL: int = 86400

# ── Lucy initiatives (MVP: single pending slot, future-oriented) ──
# Injected at session_start; expire after N starts or max age in days.

INITIATIVE_TTL_SESSIONS: int = 3
INITIATIVE_TTL_DAYS: int = 14
INITIATIVE_MIN_PAYLOAD_LEN: int = 20

# ── Hybrid re-ranking (memory_recall) ─────────────────────────────
# score = w_sim * similarity + w_conf * confidence + w_ref * reinforcement
# reinforcement = min(1.0, access_count / RANK_ACCESS_CAP)

RANK_WEIGHT_SIMILARITY: float = 0.6
RANK_WEIGHT_CONFIDENCE: float = 0.2
RANK_WEIGHT_REINFORCEMENT: float = 0.2
RANK_ACCESS_CAP: int = 20

# ── Recall thresholds and pool sizes ────────────────────────────────
# preference/state facts use the lower threshold; all others use default.

RECALL_THRESHOLD_PREFERENCE: float = 0.5
RECALL_THRESHOLD_DEFAULT: float = 0.8
RECALL_CANDIDATE_POOL: int = 25
RECALL_TOP_K: int = 10

# ── Consolidation context ──────────────────────────────────────────
# Facts passed to the consolidator LLM (vector search over session text).

CONSOLIDATION_CANDIDATE_POOL: int = 30
CONSOLIDATION_TOP_K: int = 15

# ── LLM calls (recall summary, personality composition) ───────────
# Retries transient OpenCode/network failures before surfacing to callers.

LLM_RETRY_MAX_ATTEMPTS: int = 3
LLM_RETRY_BASE_DELAY_SECONDS: float = 1.0
LLM_INSTABILITY_WINDOW_SECONDS: int = 600

# ── Data quality (Phase D) ─────────────────────────────────────────
# Cosine-like similarity via vec0 distance: sim = 1 / (1 + distance).

DEDUP_SIMILARITY_THRESHOLD: float = 0.85
DEDUP_MAX_DISTANCE: float = (1.0 / DEDUP_SIMILARITY_THRESHOLD) - 1.0