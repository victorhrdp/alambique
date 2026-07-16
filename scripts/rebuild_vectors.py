#!/usr/bin/env python3
"""Rebuild vec0_facts and vec0_sessions from canonical tables (daemon stopped)."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys

from alambique.database import Database
from alambique.ollama_client import OllamaClient
from alambique.server import DB_PATH
from alambique.vector_rebuild import backup_db, rebuild_vectors


def _daemon_running() -> bool:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "alambique.service"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() == "active"
    except FileNotFoundError:
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild vec0 embeddings from facts and session summaries."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts only; do not modify the database",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip automatic DB backup before rebuild",
    )
    parser.add_argument(
        "--facts-only",
        action="store_true",
        help="Rebuild vec0_facts only",
    )
    parser.add_argument(
        "--sessions-only",
        action="store_true",
        help="Rebuild vec0_sessions only",
    )
    args = parser.parse_args()

    if args.facts_only and args.sessions_only:
        raise SystemExit("Use at most one of --facts-only / --sessions-only")

    if _daemon_running() and not args.dry_run:
        raise SystemExit(
            "alambique.service está activo. Para rebuild por CLI: "
            "systemctl --user stop alambique.service "
            "(o usa memory_rebuild_vectors vía MCP con el daemon en marcha)"
        )

    db = Database(DB_PATH)
    db.connect()
    ollama = OllamaClient()
    try:
        if not args.dry_run and not args.no_backup:
            backup = backup_db(db.path)
            print(f"Backup: {backup}")

        report = await rebuild_vectors(
            db,
            ollama,
            dry_run=args.dry_run,
            facts_only=args.facts_only,
            sessions_only=args.sessions_only,
        )
        if report.get("warnings"):
            for w in report["warnings"]:
                print(f"warning: {w}", file=sys.stderr)
        for key, value in report.items():
            if key != "warnings":
                print(f"{key}: {value}")
    finally:
        await ollama.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())