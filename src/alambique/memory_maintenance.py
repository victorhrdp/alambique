"""Memory data-quality helpers (legacy facts code removed; only pure utils remain if needed)."""

from __future__ import annotations

import logging
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alambique.database import Database

logger = logging.getLogger("alambique.maintenance")


def parse_embedding_blob(blob: bytes) -> list[float]:
    """Decode a vec0 float32 embedding blob (still used for sessions/threads vecs)."""
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))





