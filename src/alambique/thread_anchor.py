"""Anchor checks: refuse thread update/merge when the session never touched that topic.

Pure lexical overlap — no embeddings. Used at consolidation apply time so a
high-salience thread listed in the prompt cannot be rewritten from an unrelated chat.
"""

from __future__ import annotations

import re
import unicodedata

# Tokens that appear almost everywhere (Lucy, names, meta words) and must not
# alone justify an update.
_GENERIC_TOKENS: frozenset[str] = frozenset({
    "lucy", "victor", "víctor", "victor", "nosotros", "nuestro", "nuestra",
    "thread", "hilo", "tema", "sesion", "sesión", "session", "conversacion",
    "conversación", "charla", "update", "create", "merge", "estado", "state",
    "general", "about", "with", "from", "this", "that", "para", "como", "sobre",
    "entre", "desde", "hasta", "cuando", "donde", "porque", "también", "tambien",
    "much", "very", "have", "been", "will", "just", "solo", "más", "mas", "menos",
    "//", "http", "https", "www",
})

_TOKEN_RE = re.compile(r"[a-zà-öø-ÿ0-9]{4,}", re.IGNORECASE)


def _fold(text: str) -> str:
    """Lowercase and strip combining marks so inglés/ingles match more easily."""
    text = (text or "").lower()
    nk = unicodedata.normalize("NFD", text)
    return "".join(c for c in nk if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> set[str]:
    """Distinct content tokens (length ≥ 4), folded, minus generics."""
    folded = _fold(text)
    out: set[str] = set()
    for m in _TOKEN_RE.finditer(folded):
        tok = m.group(0)
        if tok in _GENERIC_TOKENS:
            continue
        if tok.isdigit():
            continue
        out.add(tok)
    return out


def key_tokens(key: str) -> set[str]:
    """Tokens from a thread key (split on _ -)."""
    if not key:
        return set()
    parts = re.split(r"[_\-\s]+", key)
    return tokenize(" ".join(parts))


def thread_anchor_tokens(
    key: str,
    *,
    title: str = "",
    search_text: str = "",
    current_state: str = "",
) -> set[str]:
    """Anchor vocabulary for a thread (what must appear in the session to update it)."""
    anchors = key_tokens(key)
    anchors |= tokenize(title or "")
    # Prefer search_text (compact); a short head of state for older rows without it.
    anchors |= tokenize((search_text or "")[:400])
    if len(anchors) < 3:
        anchors |= tokenize((current_state or "")[:280])
    return anchors


def session_tokens(session_text: str) -> set[str]:
    return tokenize(session_text or "")


def is_thread_anchored_to_session(
    session_text: str,
    key: str,
    *,
    title: str = "",
    search_text: str = "",
    current_state: str = "",
    min_hits: int = 2,
) -> bool:
    """True if the session shares enough anchor tokens with the thread identity.

    - Fewer than `min_hits` available anchors → require all of them (small keys).
    - Empty anchors after filtering → fail closed (do not update).
    """
    anchors = thread_anchor_tokens(
        key,
        title=title,
        search_text=search_text,
        current_state=current_state,
    )
    if not anchors:
        return False
    sess = session_tokens(session_text)
    hits = anchors & sess
    need = min(min_hits, len(anchors))
    return len(hits) >= need


def anchor_hit_count(
    session_text: str,
    key: str,
    *,
    title: str = "",
    search_text: str = "",
    current_state: str = "",
) -> tuple[int, set[str]]:
    """Debug helper: (hit count, hit tokens)."""
    anchors = thread_anchor_tokens(
        key,
        title=title,
        search_text=search_text,
        current_state=current_state,
    )
    hits = anchors & session_tokens(session_text)
    return len(hits), hits
