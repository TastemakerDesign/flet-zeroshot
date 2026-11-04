import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Recording:
    id: int
    # Unix timestamps in seconds.
    started_at: float
    ended_at: float
    prompts: list[str] = field(default_factory=list)


@dataclass
class QuerySession:
    prompt: str
    baseline_prompt: str | None
    started_at: float
    ended_at: float | None = None
    # (Unix timestamp, cosine similarity) pairs in the order they were measured.
    scores: list[tuple[float, float]] = field(default_factory=list)


class RecordingsDB:
    _SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS query_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    baseline_prompt TEXT,
    started_at REAL NOT NULL,
    ended_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scores (
    session_id INTEGER NOT NULL REFERENCES query_sessions(id) ON DELETE CASCADE,
    timestamp REAL NOT NULL,
    similarity REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS query_sessions_recording
    ON query_sessions (recording_id);
CREATE INDEX IF NOT EXISTS scores_session ON scores (session_id);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

    def __init__(self, path: Path):
        self._path = path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(self._SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(query_sessions)")}
        if "baseline_prompt" not in columns:
            conn.execute("ALTER TABLE query_sessions ADD COLUMN baseline_prompt TEXT")
        return conn

    def save_recording(self, started_at: float, ended_at: float, sessions: list[QuerySession]) -> None:
        with closing(self._connect()) as conn, conn:
            recording_id = conn.execute(
                "INSERT INTO recordings (started_at, ended_at) VALUES (?, ?)",
                (started_at, ended_at),
            ).lastrowid
            for session in sessions:
                session_id = conn.execute(
                    "INSERT INTO query_sessions (recording_id, prompt, baseline_prompt, started_at, ended_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        recording_id,
                        session.prompt,
                        session.baseline_prompt,
                        session.started_at,
                        session.ended_at if session.ended_at is not None else ended_at,
                    ),
                ).lastrowid
                conn.executemany(
                    "INSERT INTO scores (session_id, timestamp, similarity) VALUES (?, ?, ?)",
                    [(session_id, t, s) for t, s in session.scores],
                )

    def list_recordings(self) -> list[Recording]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id, started_at, ended_at FROM recordings ORDER BY started_at DESC").fetchall()
            sessions = conn.execute("SELECT recording_id, prompt FROM query_sessions ORDER BY started_at").fetchall()
        recordings = {row[0]: Recording(*row) for row in rows}
        for recording_id, prompt in sessions:
            prompts = recordings[recording_id].prompts
            if prompt not in prompts:
                prompts.append(prompt)
        return list(recordings.values())

    def get_recording(self, recording_id: int) -> tuple[Recording, list[QuerySession]] | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, started_at, ended_at FROM recordings WHERE id = ?",
                (recording_id,),
            ).fetchone()
            if row is None:
                return None
            sessions = []
            for session_id, prompt, baseline_prompt, started_at, ended_at in conn.execute(
                "SELECT id, prompt, baseline_prompt, started_at, ended_at FROM query_sessions "
                "WHERE recording_id = ? ORDER BY started_at",
                (recording_id,),
            ).fetchall():
                scores = conn.execute(
                    "SELECT timestamp, similarity FROM scores WHERE session_id = ? ORDER BY timestamp",
                    (session_id,),
                ).fetchall()
                sessions.append(QuerySession(prompt, baseline_prompt, started_at, ended_at, scores))
        return Recording(*row), sessions

    def delete_recording(self, recording_id: int) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))

    def get_setting(self, key: str, default: str) -> str:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return default if row is None else row[0]

    def set_setting(self, key: str, value: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )


recordings_db = RecordingsDB(Path(os.environ.get("FLET_APP_STORAGE_DATA", ".")) / "recordings.db")
