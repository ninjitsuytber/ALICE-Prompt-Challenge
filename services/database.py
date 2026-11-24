from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "challenge.db"

_DB_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _get_conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                round_id TEXT PRIMARY KEY,
                player_name TEXT NOT NULL,
                shuffled_ids_json TEXT NOT NULL,
                picked_slot INTEGER,
                picked_challenge_id TEXT,
                status TEXT NOT NULL DEFAULT 'picking',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS leaderboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                round_id TEXT UNIQUE NOT NULL,
                player_name TEXT NOT NULL,
                challenge_id TEXT NOT NULL,
                challenge_code TEXT NOT NULL,
                picked_slot INTEGER NOT NULL,
                prompt_text TEXT NOT NULL,
                generated_image_url TEXT NOT NULL,
                similarity_score REAL NOT NULL,
                semantic_score REAL NOT NULL,
                style_score REAL NOT NULL,
                color_score REAL NOT NULL,
                composition_score REAL NOT NULL,
                generator_source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_leaderboard_score
                ON leaderboard (similarity_score DESC, id ASC);
            """
        )


def create_round(player_name: str, shuffled_ids: list[str]) -> dict:
    round_id = f"R-{uuid.uuid4().hex[:8].upper()}"
    created_at = _utc_now_iso()
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO rounds (round_id, player_name, shuffled_ids_json, status, created_at)
            VALUES (?, ?, ?, 'picking', ?)
            """,
            (round_id, player_name, json.dumps(shuffled_ids), created_at),
        )
    return {
        "round_id": round_id,
        "player_name": player_name,
        "status": "picking",
        "slot_count": len(shuffled_ids),
        "created_at": created_at,
    }


def get_round(round_id: str) -> dict | None:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM rounds WHERE round_id = ?",
            (round_id,),
        ).fetchone()
    if not row:
        return None
    data = dict(row)
    data["shuffled_ids"] = json.loads(data.pop("shuffled_ids_json"))
    return data


def reshuffle_round(round_id: str, new_shuffled_ids: list[str]) -> bool:
    with _get_conn() as conn:
        cur = conn.execute(
            """
            UPDATE rounds
               SET shuffled_ids_json = ?
             WHERE round_id = ? AND status = 'picking' AND picked_slot IS NULL
            """,
            (json.dumps(new_shuffled_ids), round_id),
        )
        return cur.rowcount > 0


def pick_slot(round_id: str, slot_index: int) -> dict | None:
    if slot_index < 1 or slot_index > 10:
        raise ValueError("slot_index must be between 1 and 10")

    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM rounds WHERE round_id = ?",
            (round_id,),
        ).fetchone()
        if not row:
            return None
        if row["picked_slot"] is not None:
            return {
                "round_id": row["round_id"],
                "player_name": row["player_name"],
                "picked_slot": row["picked_slot"],
                "picked_challenge_id": row["picked_challenge_id"],
                "status": row["status"],
            }

        shuffled_ids = json.loads(row["shuffled_ids_json"])
        challenge_id = shuffled_ids[slot_index - 1]
        conn.execute(
            """
            UPDATE rounds
               SET picked_slot = ?,
                   picked_challenge_id = ?,
                   status = 'prompting'
             WHERE round_id = ?
            """,
            (slot_index, challenge_id, round_id),
        )
        return {
            "round_id": row["round_id"],
            "player_name": row["player_name"],
            "picked_slot": slot_index,
            "picked_challenge_id": challenge_id,
            "status": "prompting",
        }


def save_submission(
    *,
    round_id: str,
    player_name: str,
    challenge_id: str,
    challenge_code: str,
    picked_slot: int,
    prompt_text: str,
    generated_image_url: str,
    similarity_score: float,
    semantic_score: float,
    style_score: float,
    color_score: float,
    composition_score: float,
    generator_source: str,
) -> dict:
    created_at = _utc_now_iso()
    with _get_conn() as conn:
        existing = conn.execute(
            "SELECT * FROM leaderboard WHERE round_id = ?",
            (round_id,),
        ).fetchone()
        if existing:
            entry = dict(existing)
        else:
            cur = conn.execute(
                """
                INSERT INTO leaderboard (
                    round_id, player_name, challenge_id, challenge_code, picked_slot,
                    prompt_text, generated_image_url, similarity_score,
                    semantic_score, style_score, color_score, composition_score,
                    generator_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    round_id,
                    player_name,
                    challenge_id,
                    challenge_code,
                    picked_slot,
                    prompt_text,
                    generated_image_url,
                    similarity_score,
                    semantic_score,
                    style_score,
                    color_score,
                    composition_score,
                    generator_source,
                    created_at,
                ),
            )
            entry_id = cur.lastrowid
            conn.execute(
                "UPDATE rounds SET status = 'completed' WHERE round_id = ?",
                (round_id,),
            )
            entry = dict(
                conn.execute(
                    "SELECT * FROM leaderboard WHERE id = ?",
                    (entry_id,),
                ).fetchone()
            )

        rank_row = conn.execute(
            """
            SELECT COUNT(*) + 1 AS rank
              FROM leaderboard
             WHERE similarity_score > ?
                OR (similarity_score = ? AND id < ?)
            """,
            (entry["similarity_score"], entry["similarity_score"], entry["id"]),
        ).fetchone()
        total_row = conn.execute("SELECT COUNT(*) AS total FROM leaderboard").fetchone()

    entry["rank"] = int(rank_row["rank"])
    entry["total_entries"] = int(total_row["total"])
    return entry


def list_leaderboard(limit: int | None = None) -> list[dict]:
    query = """
            SELECT id, round_id, player_name, challenge_code, picked_slot,
                   similarity_score, semantic_score, style_score,
                   color_score, composition_score, generator_source, created_at
              FROM leaderboard
             ORDER BY similarity_score DESC, id ASC
            """
    params: tuple = ()
    if limit is not None:
        query += "\n             LIMIT ?"
        params = (limit,)
    with _get_conn() as conn:
        rows = conn.execute(query, params).fetchall()

    results = []
    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        item["rank"] = idx
        results.append(item)
    return results


def count_leaderboard() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) AS total FROM leaderboard").fetchone()
    return int(row["total"]) if row else 0


def clear_leaderboard() -> int:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM leaderboard")
        return cur.rowcount
