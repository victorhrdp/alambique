"""Rebuild vec0 tables (no facts - legacy removed)."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from alambique.database import Database
from alambique.ollama_client import OllamaClient
from alambique.vector_store import upsert_embedding

logger = logging.getLogger("alambique.vector_rebuild")

BATCH_SIZE = 32


def backup_db(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".db.bak-rebuild-vectors-{stamp}")
    shutil.copy2(db_path, backup)
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if side.exists():
            shutil.copy2(side, Path(str(backup) + suffix))
    return backup


async def rebuild_vectors(
    db: Database,
    ollama: OllamaClient,
    *,
    dry_run: bool,
    sessions_only: bool = False,
) -> dict:
    """Wipe and regenerate vec0 tables (sessions + new model entities; new model embeds at consolidation but rebuild supports full)."""
    report: dict = {
        "dry_run": dry_run,
        "sessions_with_summary": 0,
        "vec0_sessions_before": 0,
        "orphan_sessions_removed": 0,
        "vec0_sessions_cleared": 0,
        "sessions_embedded": 0,
        "sessions_failed": 0,
        "threads_embedded": 0,
        "capsules_embedded": 0,
        "echoes_embedded": 0,
        "warnings": [],
    }

    report["orphan_sessions_removed"] = db.cleanup_orphan_session_embeddings()

    summarized = db.conn.execute(
        "SELECT id, summary FROM sessions "
        "WHERE summary IS NOT NULL AND summary != '' ORDER BY created_at"
    ).fetchall()

    report["sessions_with_summary"] = len(summarized)
    report["vec0_sessions_before"] = (
        db.conn.execute("SELECT COUNT(*) FROM vec0_sessions").fetchone()[0]
        if db._table_exists("vec0_sessions")
        else 0
    )

    if dry_run:
        report["would_remove_orphan_sessions"] = db.count_orphan_session_embeddings()
        return report

    if not await ollama.health():
        report["warnings"].append("ollama_unavailable")
        return report

    report["orphan_sessions_removed"] = db.cleanup_orphan_session_embeddings()

    if db._table_exists("vec0_sessions"):
        before = db.conn.execute("SELECT COUNT(*) FROM vec0_sessions").fetchone()[0]
        db.conn.execute("DELETE FROM vec0_sessions")
        db.conn.commit()
        report["vec0_sessions_cleared"] = before

    for offset in range(0, len(summarized), BATCH_SIZE):
        chunk = summarized[offset : offset + BATCH_SIZE]
        try:
            embeddings = await ollama.embed_batch([r["summary"] for r in chunk])
            for row, emb in zip(chunk, embeddings):
                upsert_embedding(db.conn, "vec0_sessions", row["id"], emb)
                report["sessions_embedded"] += 1
        except Exception as e:
            logger.warning("Session batch failed at offset %d: %s", offset, e)
            report["sessions_failed"] += len(chunk)
            report["warnings"].append(f"sessions_batch_failed:{offset}")

    report["vec0_sessions_after"] = (
        db.conn.execute("SELECT COUNT(*) FROM vec0_sessions").fetchone()[0]
        if db._table_exists("vec0_sessions")
        else 0
    )
    report["session_orphans_after"] = db.count_orphan_session_embeddings()
    report["sessions_missing_after"] = db.count_sessions_missing_embeddings()

    if not sessions_only:
        # Rebuild threads
        threads = db.conn.execute(
            "SELECT id, search_text FROM threads WHERE search_text IS NOT NULL AND search_text != ''"
        ).fetchall()
        if db._table_exists("vec0_threads"):
            before = db.conn.execute("SELECT COUNT(*) FROM vec0_threads").fetchone()[0]
            db.conn.execute("DELETE FROM vec0_threads")
            db.conn.commit()
        for offset in range(0, len(threads), BATCH_SIZE):
            chunk = threads[offset : offset + BATCH_SIZE]
            try:
                embeddings = await ollama.embed_batch([r["search_text"] for r in chunk])
                for row, emb in zip(chunk, embeddings):
                    upsert_embedding(db.conn, "vec0_threads", row["id"], emb)
                    report["threads_embedded"] += 1
            except Exception as e:
                logger.warning("Thread batch failed at offset %d: %s", offset, e)
                report["warnings"].append(f"threads_batch_failed:{offset}")

        # Capsules
        capsules = db.conn.execute(
            "SELECT id, content FROM relationship_capsules WHERE content IS NOT NULL AND content != ''"
        ).fetchall()
        if db._table_exists("vec0_relationship_capsules"):
            before = db.conn.execute("SELECT COUNT(*) FROM vec0_relationship_capsules").fetchone()[0]
            db.conn.execute("DELETE FROM vec0_relationship_capsules")
            db.conn.commit()
        for offset in range(0, len(capsules), BATCH_SIZE):
            chunk = capsules[offset : offset + BATCH_SIZE]
            try:
                embeddings = await ollama.embed_batch([r["content"] for r in chunk])
                for row, emb in zip(chunk, embeddings):
                    upsert_embedding(db.conn, "vec0_relationship_capsules", row["id"], emb)
                    report["capsules_embedded"] += 1
            except Exception as e:
                logger.warning("Capsule batch failed at offset %d: %s", offset, e)
                report["warnings"].append(f"capsules_batch_failed:{offset}")

        # Echoes
        echoes = db.conn.execute(
            "SELECT id, content FROM echoes WHERE content IS NOT NULL AND content != ''"
        ).fetchall()
        if db._table_exists("vec0_echoes"):
            before = db.conn.execute("SELECT COUNT(*) FROM vec0_echoes").fetchone()[0]
            db.conn.execute("DELETE FROM vec0_echoes")
            db.conn.commit()
        for offset in range(0, len(echoes), BATCH_SIZE):
            chunk = echoes[offset : offset + BATCH_SIZE]
            try:
                embeddings = await ollama.embed_batch([r["content"] for r in chunk])
                for row, emb in zip(chunk, embeddings):
                    upsert_embedding(db.conn, "vec0_echoes", row["id"], emb)
                    report["echoes_embedded"] += 1
            except Exception as e:
                logger.warning("Echo batch failed at offset %d: %s", offset, e)
                report["warnings"].append(f"echoes_batch_failed:{offset}")

    return report