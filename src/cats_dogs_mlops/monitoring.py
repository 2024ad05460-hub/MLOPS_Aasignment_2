from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class PerformanceSummary:
    labeled_count: int
    correct_count: int
    accuracy: float | None
    label_distribution: dict[str, int]
    prediction_distribution: dict[str, int]


class PredictionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    request_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    true_label TEXT,
                    feedback_timestamp_utc TEXT
                )
                """
            )
            connection.commit()

    def record_prediction(
        self,
        request_id: str,
        predicted_label: str,
        confidence: float,
        probabilities: dict[str, float],
        latency_ms: float,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions
                (request_id, timestamp_utc, predicted_label, confidence, probabilities_json, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    datetime.now(timezone.utc).isoformat(),
                    predicted_label,
                    confidence,
                    json.dumps(probabilities, sort_keys=True),
                    latency_ms,
                ),
            )
            connection.commit()

    def add_feedback(self, request_id: str, true_label: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE predictions
                SET true_label = ?, feedback_timestamp_utc = ?
                WHERE request_id = ?
                """,
                (true_label, datetime.now(timezone.utc).isoformat(), request_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def performance(self) -> PerformanceSummary:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT predicted_label, true_label FROM predictions WHERE true_label IS NOT NULL"
            ).fetchall()
        labeled_count = len(rows)
        correct_count = sum(row["predicted_label"] == row["true_label"] for row in rows)
        label_distribution: dict[str, int] = {}
        prediction_distribution: dict[str, int] = {}
        for row in rows:
            label_distribution[row["true_label"]] = label_distribution.get(row["true_label"], 0) + 1
            prediction_distribution[row["predicted_label"]] = prediction_distribution.get(row["predicted_label"], 0) + 1
        return PerformanceSummary(
            labeled_count=labeled_count,
            correct_count=correct_count,
            accuracy=(correct_count / labeled_count) if labeled_count else None,
            label_distribution=label_distribution,
            prediction_distribution=prediction_distribution,
        )

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions ORDER BY timestamp_utc DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]
