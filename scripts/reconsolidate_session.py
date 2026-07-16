#!/usr/bin/env python3
"""Re-sync transcript and re-queue consolidation for one session (daemon stopped)."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from alambique.consolidator import get_api_key
from alambique.database import Database
from alambique.ollama_client import OllamaClient
from alambique.server import DB_PATH
from alambique.tools import ToolHandler


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument(
        "--consolidate",
        action="store_true",
        help="Run consolidation immediately (requires API key; daemon may be running)",
    )
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.connect()
    ollama = OllamaClient()
    try:
        session = db.get_session(args.session_id)
        if not session:
            raise SystemExit(f"Session not found: {args.session_id}")

        handler = ToolHandler(db, ollama, api_key=get_api_key())
        count = await handler._sync_session_transcript(
            args.session_id,
            session.conversation_id,
            session.client,
        )
        db.conn.execute(
            "UPDATE sessions SET consolidated = 0, summary = NULL WHERE id = ?",
            (args.session_id,),
        )
        db.conn.commit()
        print(f"Re-synced {count} messages; consolidated reset for {args.session_id}")

        if args.consolidate:
            session = db.get_session(args.session_id)
            await handler._consolidate_session(session)
            session = db.get_session(args.session_id)
            print(f"Done: consolidated={session.consolidated}")
            if session.summary:
                print(session.summary)
    finally:
        await ollama.close()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())