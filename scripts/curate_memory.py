#!/usr/bin/env python3
"""Conservative memory curation for Lucy-only Alambique DB."""

from __future__ import annotations

import argparse
from pathlib import Path

from alambique.database import Database
from alambique.models import Fact
from alambique.server import DB_PATH

CANONICAL_ALAMBIQUE_KEY = "alambique_transcript_sync_architecture"
CANONICAL_ALAMBIQUE_VALUE = (
    "Alambique v0.2.3 es la memoria unificada de Lucy: binding en session_start "
    "(client + workspace), sync de transcript en session_end para Grok, Antigravity CLI "
    "y OpenCode, sin message_append. Consolidación asíncrona y watchdog a 30 min."
)

FACTS_TO_FORGET = [
    141,  # user_name_victor duplicate
    46,   # user_grok_cli_subscription mislabeled
    143,  # expired state
    22,   # silent_error_handling — resolved
    123,  # peer dynamic duplicate
    144,  # historical dev session dump
    145,  # obsolete roadmap cleanup
    148,
    149,
    150,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    164,
    165,
]


def curate(db: Database, *, dry_run: bool) -> dict:
    report: dict = {"legacy_sessions": 0, "legacy_messages": 0, "facts_forgotten": 0}

    legacy_ids = [
        row["id"]
        for row in db.conn.execute(
            "SELECT id FROM sessions WHERE client IS NULL"
        ).fetchall()
    ]
    report["legacy_sessions"] = len(legacy_ids)
    if legacy_ids:
        placeholders = ",".join("?" for _ in legacy_ids)
        report["legacy_messages"] = db.conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({placeholders})",
            legacy_ids,
        ).fetchone()[0]

    if not dry_run:
        for sid in legacy_ids:
            db.conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
            db.conn.execute("DELETE FROM consolidations WHERE session_id = ?", (sid,))
            db.conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))

        for fact_id in FACTS_TO_FORGET:
            if fact_id == 151:
                continue
            db.forget_fact(fact_id)
            report["facts_forgotten"] += 1

        db.update_fact(
            151,
            CANONICAL_ALAMBIQUE_VALUE,
            confidence=1.0,
        )
        db.cleanup_stale_embeddings()
        db.conn.commit()

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.connect()
    try:
        before = {
            "sessions": db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "messages": db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "facts": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE confidence > 0"
            ).fetchone()[0],
        }
        report = curate(db, dry_run=args.dry_run)
        after = {
            "sessions": db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "messages": db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "facts": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE confidence > 0"
            ).fetchone()[0],
        }
        print("dry_run", args.dry_run)
        print("before", before)
        print("report", report)
        print("after", after)
    finally:
        db.close()


if __name__ == "__main__":
    main()