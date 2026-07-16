"""Text helpers for consolidation and recall."""

from __future__ import annotations

from alambique.models import Message

def consolidation_search_text(messages: list[Message], *, max_chars: int = 1500) -> str:
    """Build embedding text from a session for consolidation fact retrieval.
    Keep short to avoid overloading the embedding model (bge-m3 batch limits + CPU).
    Prefer recent context; full history tails bloat embeddings unnecessarily.
    """
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
