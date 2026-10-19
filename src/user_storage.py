
"""Local, single-user persistent storage for CineMatch."""

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "data" / "user_preferences.db"

BUCKETS = ("saved", "watched", "dismissed", "liked")


def connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS movie_preferences (
            bucket TEXT NOT NULL,
            movie_id INTEGER NOT NULL,
            PRIMARY KEY (bucket, movie_id)
        )
        """
    )
    return connection


def load_preferences():
    """Load all four preference lists from SQLite."""
    preferences = {bucket: [] for bucket in BUCKETS}

    with connect() as connection:
        rows = connection.execute(
            """
            SELECT bucket, movie_id
            FROM movie_preferences
            ORDER BY rowid
            """
        ).fetchall()

    for bucket, movie_id in rows:
        if bucket in preferences:
            preferences[bucket].append(movie_id)

    return preferences


def save_preferences(preferences):
    """Atomically replace all saved preference lists."""
    with connect() as connection:
        for bucket in BUCKETS:
            values = list(dict.fromkeys(
                int(movie_id)
                for movie_id in preferences.get(bucket, [])
            ))

            connection.execute(
                "DELETE FROM movie_preferences WHERE bucket = ?",
                (bucket,),
            )

            connection.executemany(
                """
                INSERT INTO movie_preferences (bucket, movie_id)
                VALUES (?, ?)
                """,
                [(bucket, movie_id) for movie_id in values],
            )


def toggle_movie(bucket, movie_id):
    """Add or remove a movie and return its new saved status."""
    if bucket not in BUCKETS:
        raise ValueError(f"Unknown preference bucket: {bucket}")

    movie_id = int(movie_id)

    with connect() as connection:
        existing = connection.execute(
            """
            SELECT 1 FROM movie_preferences
            WHERE bucket = ? AND movie_id = ?
            """,
            (bucket, movie_id),
        ).fetchone()

        if existing:
            connection.execute(
                """
                DELETE FROM movie_preferences
                WHERE bucket = ? AND movie_id = ?
                """,
                (bucket, movie_id),
            )
            return False

        connection.execute(
            """
            INSERT INTO movie_preferences (bucket, movie_id)
            VALUES (?, ?)
            """,
            (bucket, movie_id),
        )
        return True


def clear_preferences():
    """Remove all locally stored preferences."""
    with connect() as connection:
        connection.execute("DELETE FROM movie_preferences")
