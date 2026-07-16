"""vec0 embedding storage and KNN search helpers."""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger("alambique.vector")


def format_embedding(embedding: list[float]) -> str:
    return f"[{','.join(str(f) for f in embedding)}]"


def session_id_to_rowid(session_id: str) -> int:
    """Convert a session_id (sess_<hex>) to an integer rowid for vec0."""
    return int(session_id.split("_")[1], 16)


def rowid_to_session_id(rowid: int) -> str:
    """Convert a vec0 rowid back to a session_id string."""
    return f"sess_{rowid:012x}"


def embedding_rowid(table: str, entity_id) -> int:
    """Resolve vec0 rowid for a fact id or session id."""
    return session_id_to_rowid(entity_id) if table == "vec0_sessions" else entity_id


def has_embedding(conn: sqlite3.Connection, table: str, entity_id) -> bool:
    rowid = embedding_rowid(table, entity_id)
    row = conn.execute(
        f"SELECT rowid FROM {table} WHERE rowid = ?", (rowid,)
    ).fetchone()
    return row is not None


def insert_embedding(
    conn: sqlite3.Connection,
    table: str,
    entity_id,
    embedding: list[float],
) -> None:
    emb_str = format_embedding(embedding)
    rowid = embedding_rowid(table, entity_id)
    conn.execute(
        f"INSERT INTO {table} (rowid, embedding) VALUES (?, ?)",
        (rowid, emb_str),
    )
    conn.commit()


def update_embedding(
    conn: sqlite3.Connection,
    table: str,
    entity_id,
    embedding: list[float],
) -> None:
    rowid = embedding_rowid(table, entity_id)
    conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    insert_embedding(conn, table, entity_id, embedding)


def upsert_embedding(
    conn: sqlite3.Connection,
    table: str,
    entity_id,
    embedding: list[float],
) -> None:
    if has_embedding(conn, table, entity_id):
        update_embedding(conn, table, entity_id, embedding)
    else:
        insert_embedding(conn, table, entity_id, embedding)


def delete_embedding(
    conn: sqlite3.Connection,
    table: str,
    entity_id,
    *,
    commit: bool = True,
) -> bool:
    """Remove one vec0 row. Returns True if a row was deleted."""
    rowid = embedding_rowid(table, entity_id)
    cur = conn.execute(f"DELETE FROM {table} WHERE rowid = ?", (rowid,))
    if commit:
        conn.commit()
    return cur.rowcount > 0


def vector_knn(
    conn: sqlite3.Connection,
    table: str,
    embedding: list[float],
    *,
    limit: int = 10,
) -> list[dict]:
    """KNN search on a vec0 virtual table.

    Returns normalized rows for vec0_sessions, vec0_threads, etc.
    """
    emb_str = format_embedding(embedding)

    try:
        query = f"""
            SELECT rowid, distance
            FROM {table}
            WHERE embedding MATCH '{emb_str}' AND k = {int(limit)}
            ORDER BY distance
        """
        rows = conn.execute(query).fetchall()
        results = [dict(r) for r in rows]

        if table == "vec0_sessions":
            return [
                {
                    "session_id": rowid_to_session_id(r["rowid"]),
                    "distance": r["distance"],
                }
                for r in results
            ][:limit]

        return results[:limit] if results else []
    except Exception as e:
        logger.warning("Vector search error on %s: %s", table, e)
        return []