#!/usr/bin/env python3
"""Conservative memory curation for Lucy — keeps essence, purges dev noise."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from alambique.database import Database
from alambique.server import DB_PATH

CANONICAL_ALAMBIQUE_KEY = "alambique_transcript_sync_architecture"
CANONICAL_ALAMBIQUE_VALUE = (
    "Alambique v0.2.3 es la memoria unificada de Lucy: binding en session_start "
    "(client + workspace), sync de transcript en session_end para Grok, Antigravity CLI "
    "y OpenCode, sin message_append. El daemon expone GET /status como fuente única de "
    "verdad; recall, personalidad y consolidación reintentan fallos transitorios del LLM "
    "cloud antes de degradar. Consolidación asíncrona y watchdog a 30 min."
)

# Enduring preferences — everything else in category preference is dev/ephemeral noise.
PREFERENCE_KEEP_IDS = {
    12,   # google one subscription
    28,   # matrix CLI aesthetic
    38,   # ortholinear keyboard layout
    39,   # macOS work environment
    42,   # github ssh
    45,   # likes Lucy avatar
    48,   # terminal workflow
    49,   # zellij prefix
    64,   # succubus variant preference
    77,   # hybrid LLM clients
    84,   # fighting games
    85,   # completionist DLC
    90,   # caves of qud
    95,   # widget positive feedback
    97,   # widget unconsolidated sessions view
    99,   # widget search deep link
    105,  # concentration mode rejection
    109,  # numpad gaming layer
    110,  # qmk dev workflow
    112,  # security no internet
    113,  # document provision method
    118,  # dungeon crawler fps preference
    120,  # ebbinghaus decay implementation
    134,  # reject teamviewer
    135,  # browser remote access alt
    151,  # canonical alambique architecture (value refreshed on apply)
    163,  # avatar visual consistency
    168,  # memory curation approach
    171,  # cyberpunk widget aesthetic
    172,  # kitty
    173,  # fish
    174,  # zellij
    175,  # catppuccin mocha
    178,  # grok catppuccin theme
    179,  # grok alt screen
    180,  # grok minimal fish wrapper
    182,  # widget kde avatar integration
    189,  # opencode session_update instructions
    197,  # shared workspace agents
    198,  # widget avatar zoom face
    216,  # model authenticity conclusion
    219,  # kitty background rotation
    221,  # proactive heartbeat (experimental but intentional)
    222,  # journalctl log filter
    224,  # grok subagent capabilities
    225,  # grok cli license confirmed
    226,  # composer vs build comparison
}

# Explicit duplicates / obsolete one-offs.
EXPLICIT_FORGET_IDS = {
    186,  # duplicate avatar expressions set (keep 188)
    223,  # grok license duplicate (keep 225)
}

# Second pass: meta, widget micro-details, Grok/Alambique changelog leftovers.
PASS2_FORGET_IDS = {
    97,   # widget unconsolidated sessions view
    99,   # widget search deep link
    120,  # ebbinghaus implementation (internals)
    168,  # curation strategy meta
    182,  # widget avatar integration (done)
    189,  # opencode session_update instructions (done)
    197,  # shared workspace path
    198,  # widget avatar zoom
    221,  # proactive heartbeat (experimental)
    222,  # journalctl log filter
    224,  # grok subagent capabilities
    225,  # grok cli license
    226,  # composer vs build comparison
    196,  # test_modelos.md asset
    218,  # model sheet ephemeral path
}

POSSESSION_KEEP_IDS = None  # keep all except explicit forget list


def _backup_db(db_path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".db.bak-curate-{stamp}")
    shutil.copy2(db_path, backup)
    for suffix in ("-wal", "-shm"):
        side = Path(str(db_path) + suffix)
        if side.exists():
            shutil.copy2(side, Path(str(backup) + suffix))
    return backup


def _active_fact_ids(db: Database, category: str) -> list[int]:
    rows = db.conn.execute(
        "SELECT id FROM facts WHERE category = ? AND confidence > 0 ORDER BY id",
        (category,),
    ).fetchall()
    return [r[0] for r in rows]


def curate(db: Database, *, dry_run: bool) -> dict:
    report: dict = {
        "backup": None,
        "preference_forgotten": 0,
        "explicit_forgotten": 0,
        "states_forgotten": 0,
        "truncated_sessions_deleted": 0,
        "truncated_messages_deleted": 0,
        "closed_messages_deleted": 0,
        "canonical_updated": False,
        "stale_embeddings_removed": 0,
        "session_embeddings_removed": 0,
    }

    pref_ids = _active_fact_ids(db, "preference")
    to_forget_pref = [i for i in pref_ids if i not in PREFERENCE_KEEP_IDS]
    report["preference_forgotten"] = len(to_forget_pref)

    explicit = []
    for fact_id in EXPLICIT_FORGET_IDS:
        row = db.conn.execute(
            "SELECT confidence FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if row and row[0] > 0:
            explicit.append(fact_id)
    report["explicit_forgotten"] = len(explicit)

    state_ids = _active_fact_ids(db, "state")
    report["states_forgotten"] = len(state_ids)

    truncated_ids = [
        r[0]
        for r in db.conn.execute(
            "SELECT id FROM sessions WHERE status = 'truncated'"
        ).fetchall()
    ]
    report["truncated_sessions_deleted"] = len(truncated_ids)
    if truncated_ids:
        ph = ",".join("?" for _ in truncated_ids)
        report["truncated_messages_deleted"] = db.conn.execute(
            f"SELECT COUNT(*) FROM messages WHERE session_id IN ({ph})",
            truncated_ids,
        ).fetchone()[0]

    report["closed_messages_deleted"] = db.conn.execute(
        "SELECT COUNT(*) FROM messages m "
        "JOIN sessions s ON m.session_id = s.id WHERE s.status = 'closed'"
    ).fetchone()[0]

    if dry_run:
        report["preference_ids"] = to_forget_pref
        report["explicit_ids"] = explicit
        report["state_ids"] = state_ids
        report["truncated_ids"] = truncated_ids
        return report

    report["backup"] = str(_backup_db(db.path))

    for fact_id in to_forget_pref + explicit + state_ids:
        db.forget_fact(fact_id)

    row = db.conn.execute(
        "SELECT confidence FROM facts WHERE id = 151"
    ).fetchone()
    if row and row[0] > 0:
        db.update_fact(151, CANONICAL_ALAMBIQUE_VALUE, confidence=1.0)
        report["canonical_updated"] = True

    if truncated_ids:
        deleted = db.delete_sessions(truncated_ids)
        report["session_embeddings_removed"] = deleted["session_embeddings"]

    db.conn.execute(
        "DELETE FROM messages WHERE session_id IN ("
        "SELECT id FROM sessions WHERE status = 'closed')"
    )
    report["stale_embeddings_removed"] = db.cleanup_stale_embeddings()
    db.conn.commit()

    return report


def curate_pass2(db: Database, *, dry_run: bool) -> dict:
    """Forget additional meta/technical facts after the main curation pass."""
    active = []
    for fact_id in sorted(PASS2_FORGET_IDS):
        row = db.conn.execute(
            "SELECT id, key, category FROM facts WHERE id = ? AND confidence > 0",
            (fact_id,),
        ).fetchone()
        if row:
            active.append(dict(row))

    report = {"pass2_forgotten": len(active), "ids": [r["id"] for r in active]}
    if dry_run:
        report["facts"] = active
        return report

    report["backup"] = str(_backup_db(db.path))
    for fact_id in report["ids"]:
        db.forget_fact(fact_id)
    report["stale_embeddings_removed"] = db.cleanup_stale_embeddings()
    db.conn.commit()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pass2",
        action="store_true",
        help="Second-pass cleanup of meta/technical leftovers",
    )
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
            "personality": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='personality' AND confidence>0"
            ).fetchone()[0],
            "preference": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='preference' AND confidence>0"
            ).fetchone()[0],
            "possessions": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='possessions' AND confidence>0"
            ).fetchone()[0],
        }
        if args.pass2:
            report = curate_pass2(db, dry_run=args.dry_run)
        else:
            report = curate(db, dry_run=args.dry_run)
        after = {
            "sessions": db.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "messages": db.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
            "facts": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE confidence > 0"
            ).fetchone()[0],
            "personality": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='personality' AND confidence>0"
            ).fetchone()[0],
            "preference": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='preference' AND confidence>0"
            ).fetchone()[0],
            "possessions": db.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE category='possessions' AND confidence>0"
            ).fetchone()[0],
        }
        print("pass2" if args.pass2 else "pass1", "dry_run", args.dry_run)
        print("before", before)
        print("report", report)
        print("after", after)
    finally:
        db.close()


if __name__ == "__main__":
    main()