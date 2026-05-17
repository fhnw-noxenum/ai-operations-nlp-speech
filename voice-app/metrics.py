"""
Per-turn tracing für die Voice-Pipeline.

Eine TurnTrace-Instanz pro User-Turn. Stempelt Zeitpunkte für jede
Stufe (STT done, LLM first token = TTFT, LLM done, TTS first chunk = TTFA).
Persistiert nach SQLite, damit das Dashboard sie auswerten kann.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock


DB_PATH = os.environ.get("METRICS_DB", "/data/metrics.sqlite")

_DB_LOCK = Lock()


def _ensure_db() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                ts_unix      REAL    NOT NULL,
                mode         TEXT    NOT NULL,
                t_stt_done   REAL    NOT NULL,
                t_llm_first  REAL    NOT NULL,
                t_llm_done   REAL    NOT NULL,
                t_tts_first  REAL    NOT NULL,
                t_total      REAL    NOT NULL,
                transcript   TEXT    NOT NULL,
                response     TEXT    NOT NULL,
                payload      TEXT    NOT NULL
            )
            """
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts_unix)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_turns_mode ON turns(mode)")
    print(f"[metrics] db ready at {DB_PATH}", flush=True)


_ensure_db()


@dataclass
class TurnTrace:
    """Zeitstempel pro Stufe in Sekunden relativ zu t0 (Audio-Start)."""

    mode: str = "streaming"
    t0: float = field(default_factory=time.perf_counter)

    t_stt_done: float = 0.0
    t_llm_first: float = 0.0   # TTFT
    t_llm_done: float = 0.0
    t_tts_first: float = 0.0   # TTFA
    t_total: float = 0.0

    transcript: str = ""
    response: str = ""

    def stamp(self, name: str) -> float:
        """Setzt das benannte Feld auf die seit t0 vergangene Zeit (s)."""
        elapsed = time.perf_counter() - self.t0
        setattr(self, name, elapsed)
        return elapsed

    def persist(self) -> None:
        self.t_total = max(
            self.t_total,
            self.t_tts_first,
            self.t_llm_done,
            self.t_stt_done,
        )
        payload = asdict(self)
        with _DB_LOCK, sqlite3.connect(DB_PATH) as con:
            con.execute(
                """
                INSERT INTO turns
                (ts_unix, mode, t_stt_done, t_llm_first, t_llm_done,
                 t_tts_first, t_total, transcript, response, payload)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    time.time(),
                    self.mode,
                    self.t_stt_done,
                    self.t_llm_first,
                    self.t_llm_done,
                    self.t_tts_first,
                    self.t_total,
                    self.transcript,
                    self.response,
                    json.dumps(payload),
                ),
            )
        print(
            f"[metrics] turn  mode={self.mode}  "
            f"stt={self.t_stt_done*1000:.0f}ms  "
            f"ttft={self.t_llm_first*1000:.0f}ms  "
            f"ttfa={self.t_tts_first*1000:.0f}ms  "
            f"total={self.t_total*1000:.0f}ms",
            flush=True,
        )


def recent_turns(limit: int = 50) -> list[dict]:
    with _DB_LOCK, sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM turns ORDER BY ts_unix DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
