"""SQLite persistence layer for ARGUS-X long-term identity memory.

Stores global identity records, observation metadata (camera ID, local track ID,
timestamp, embedding ID), and the mapping from FAISS vector IDs to observations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from config import MemoryConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObservationRecord:
    """Metadata for a single stored appearance embedding observation."""

    embedding_id: str
    vector_id: int
    identity_id: str
    camera_id: str
    local_track_id: Optional[int]
    timestamp: float
    embedding_hash: str


@dataclass(frozen=True)
class LifecycleSnapshot:
    """Persisted global identity lifecycle state."""

    global_id: str
    state: str
    created_at: float
    last_seen_at: float
    current_camera_id: Optional[str]
    camera_track_history: dict[str, list[int]]
    camera_sequence: list[str]


@dataclass(frozen=True)
class AuditLogEntry:
    """Structured audit record for an association or lifecycle decision."""

    timestamp: float
    camera_id: str
    local_track_id: int
    global_id: Optional[str]
    decision: str
    appearance_score: Optional[float]
    temporal_score: Optional[float]
    camera_score: Optional[float]
    motion_score: Optional[float]
    combined_confidence: Optional[float]
    reason: str


@dataclass(frozen=True)
class UnresolvedObservationRecord:
    """Persisted unresolved/uncertain tracklet awaiting evidence or manual review."""

    tracklet_uid: str
    camera_id: str
    local_track_id: int
    first_seen_at: float
    last_seen_at: float
    candidate_ids: list[str]
    confidence: float
    reason: str
    status: str
    resolved_global_id: Optional[str] = None


class IdentityStorage:
    """SQLite-backed metadata store for FAISS identity embeddings."""

    def __init__(self, db_path: Union[str, Path] = MemoryConfig.db_path) -> None:
        """Open (or create) the SQLite database and initialize schema.

        Args:
            db_path: Path to the SQLite database file, or ``":memory:"``.
        """
        self.db_path_str = str(db_path)
        if self.db_path_str != ":memory:":
            Path(self.db_path_str).parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA busy_timeout = 5000;")
        self._init_schema()
        logger.debug("Initialized IdentityStorage at %s", self.db_path_str)

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identities (
                    identity_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    embedding_id TEXT UNIQUE NOT NULL,
                    vector_id INTEGER UNIQUE NOT NULL,
                    identity_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    local_track_id INTEGER,
                    timestamp REAL NOT NULL,
                    embedding_hash TEXT NOT NULL,
                    FOREIGN KEY (identity_id) REFERENCES identities(identity_id)
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_identity
                ON observations(identity_id);
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_vector_id
                ON observations(vector_id);
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS identity_lifecycles (
                    global_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    current_camera_id TEXT,
                    camera_track_history_json TEXT NOT NULL,
                    camera_sequence_json TEXT NOT NULL,
                    FOREIGN KEY (global_id) REFERENCES identities(identity_id)
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    local_track_id INTEGER NOT NULL,
                    global_id TEXT,
                    decision TEXT NOT NULL,
                    appearance_score REAL,
                    temporal_score REAL,
                    camera_score REAL,
                    motion_score REAL,
                    combined_confidence REAL,
                    reason TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unresolved_observations (
                    tracklet_uid TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    local_track_id INTEGER NOT NULL,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    candidate_ids_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    resolved_global_id TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_observations_dedup
                ON observations(identity_id, embedding_hash);
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ObservationRecord:
        return ObservationRecord(
            embedding_id=str(row["embedding_id"]),
            vector_id=int(row["vector_id"]),
            identity_id=str(row["identity_id"]),
            camera_id=str(row["camera_id"]),
            local_track_id=(
                int(row["local_track_id"])
                if row["local_track_id"] is not None
                else None
            ),
            timestamp=float(row["timestamp"]),
            embedding_hash=str(row["embedding_hash"]),
        )

    def ensure_identity(
        self, identity_id: str, created_at: Optional[float] = None
    ) -> None:
        """Register an identity ID if it does not already exist."""
        if not isinstance(identity_id, str) or not identity_id.strip():
            raise ValueError("identity_id must be a non-empty string.")
        ts = float(time.time() if created_at is None else created_at)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO identities (identity_id, created_at)
                    VALUES (?, ?);
                    """,
                    (identity_id, ts),
                )

    def add_observation(
        self,
        embedding_id: str,
        vector_id: int,
        identity_id: str,
        camera_id: str,
        local_track_id: Optional[int],
        timestamp: float,
        embedding_hash: str,
    ) -> ObservationRecord:
        """Persist a new observation record and link it to its FAISS vector ID."""
        self.ensure_identity(identity_id, created_at=timestamp)
        try:
            with self._lock:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO observations (
                            embedding_id,
                            vector_id,
                            identity_id,
                            camera_id,
                            local_track_id,
                            timestamp,
                            embedding_hash
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            embedding_id,
                            int(vector_id),
                            identity_id,
                            camera_id,
                            local_track_id,
                            float(timestamp),
                            embedding_hash,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Failed to store observation (embedding_id={embedding_id!r}, "
                f"vector_id={vector_id}): {exc}"
            ) from exc

        return ObservationRecord(
            embedding_id=embedding_id,
            vector_id=int(vector_id),
            identity_id=identity_id,
            camera_id=camera_id,
            local_track_id=local_track_id,
            timestamp=float(timestamp),
            embedding_hash=embedding_hash,
        )

    def find_duplicate(
        self, identity_id: str, embedding_hash: str
    ) -> Optional[ObservationRecord]:
        """Return an existing observation if the same identity+hash already exists."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT embedding_id, vector_id, identity_id, camera_id,
                       local_track_id, timestamp, embedding_hash
                FROM observations
                WHERE identity_id = ? AND embedding_hash = ?
                LIMIT 1;
                """,
                (identity_id, embedding_hash),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_by_vector_id(self, vector_id: int) -> Optional[ObservationRecord]:
        """Look up an observation by its FAISS vector ID."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT embedding_id, vector_id, identity_id, camera_id,
                       local_track_id, timestamp, embedding_hash
                FROM observations
                WHERE vector_id = ?;
                """,
                (int(vector_id),),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_by_vector_ids(
        self, vector_ids: Sequence[int]
    ) -> dict[int, ObservationRecord]:
        """Batch lookup of observations by FAISS vector IDs."""
        clean_ids = [int(vid) for vid in vector_ids if int(vid) >= 0]
        if not clean_ids:
            return {}
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT embedding_id, vector_id, identity_id, camera_id,
                       local_track_id, timestamp, embedding_hash
                FROM observations
                WHERE vector_id IN ({placeholders});
                """,
                clean_ids,
            ).fetchall()
            return {int(row["vector_id"]): self._row_to_record(row) for row in rows}

    def get_by_embedding_id(self, embedding_id: str) -> Optional[ObservationRecord]:
        """Look up an observation by its unique embedding ID."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT embedding_id, vector_id, identity_id, camera_id,
                       local_track_id, timestamp, embedding_hash
                FROM observations
                WHERE embedding_id = ?;
                """,
                (embedding_id,),
            ).fetchone()
            return self._row_to_record(row) if row is not None else None

    def get_observations_for_identity(
        self, identity_id: str
    ) -> list[ObservationRecord]:
        """Return all stored observations for a given identity in insertion order."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT embedding_id, vector_id, identity_id, camera_id,
                       local_track_id, timestamp, embedding_hash
                FROM observations
                WHERE identity_id = ?
                ORDER BY id ASC;
                """,
                (identity_id,),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_vector_identity_map(self) -> dict[int, str]:
        """Return the full mapping of FAISS vector_id -> identity_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT vector_id, identity_id FROM observations ORDER BY vector_id ASC;"
            ).fetchall()
            return {int(row["vector_id"]): str(row["identity_id"]) for row in rows}

    def list_identities(self) -> list[str]:
        """Return all registered identity IDs."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT identity_id FROM identities ORDER BY created_at ASC, identity_id ASC;"
            ).fetchall()
            return [str(row["identity_id"]) for row in rows]

    def count_observations(self) -> int:
        """Return total number of stored observations."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS cnt FROM observations;").fetchone()
            return int(row["cnt"]) if row is not None else 0

    def get_observation_counts(self) -> dict[str, int]:
        """Return total observation count per identity in one query."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT identity_id, COUNT(*) AS cnt FROM observations GROUP BY identity_id;"
            ).fetchall()
            return {str(row["identity_id"]): int(row["cnt"]) for row in rows}

    def upsert_lifecycle(self, snapshot: LifecycleSnapshot) -> None:
        """Persist or update a global identity lifecycle snapshot."""
        with self._lock:
            self.ensure_identity(snapshot.global_id, created_at=snapshot.created_at)
            with self._conn:
                self._conn.execute(
                """
                INSERT INTO identity_lifecycles (
                    global_id, state, created_at, last_seen_at,
                    current_camera_id, camera_track_history_json, camera_sequence_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(global_id) DO UPDATE SET
                    state = excluded.state,
                    last_seen_at = excluded.last_seen_at,
                    current_camera_id = excluded.current_camera_id,
                    camera_track_history_json = excluded.camera_track_history_json,
                    camera_sequence_json = excluded.camera_sequence_json;
                """,
                (
                    snapshot.global_id,
                    snapshot.state,
                    float(snapshot.created_at),
                    float(snapshot.last_seen_at),
                    snapshot.current_camera_id,
                    json.dumps(snapshot.camera_track_history),
                    json.dumps(snapshot.camera_sequence),
                ),
            )

    def load_lifecycles(self) -> list[LifecycleSnapshot]:
        """Load all persisted global identity lifecycle snapshots."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT global_id, state, created_at, last_seen_at,
                       current_camera_id, camera_track_history_json, camera_sequence_json
                FROM identity_lifecycles
                ORDER BY created_at ASC, global_id ASC;
                """
            ).fetchall()
        return [
            LifecycleSnapshot(
                global_id=str(row["global_id"]),
                state=str(row["state"]),
                created_at=float(row["created_at"]),
                last_seen_at=float(row["last_seen_at"]),
                current_camera_id=(
                    str(row["current_camera_id"])
                    if row["current_camera_id"] is not None
                    else None
                ),
                camera_track_history=json.loads(row["camera_track_history_json"]),
                camera_sequence=json.loads(row["camera_sequence_json"]),
            )
            for row in rows
        ]

    def add_audit_log(self, entry: AuditLogEntry) -> None:
        """Persist an audit log entry."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO audit_logs (
                        timestamp, camera_id, local_track_id, global_id, decision,
                        appearance_score, temporal_score, camera_score, motion_score,
                        combined_confidence, reason
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        float(entry.timestamp),
                        entry.camera_id,
                        int(entry.local_track_id),
                        entry.global_id,
                        entry.decision,
                        entry.appearance_score,
                        entry.temporal_score,
                        entry.camera_score,
                        entry.motion_score,
                        entry.combined_confidence,
                        entry.reason,
                    ),
                )

    def get_audit_logs(
        self,
        global_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> list[AuditLogEntry]:
        """Query persisted audit logs with optional filters."""
        clauses: list[str] = []
        params: list[object] = []
        if global_id is not None:
            clauses.append("global_id = ?")
            params.append(global_id)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if decision is not None:
            clauses.append("decision = ?")
            params.append(decision)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT timestamp, camera_id, local_track_id, global_id, decision,
                       appearance_score, temporal_score, camera_score, motion_score,
                       combined_confidence, reason
                FROM audit_logs
                {where_sql}
                ORDER BY id ASC;
                """,
                params,
            ).fetchall()
        return [
            AuditLogEntry(
                timestamp=float(row["timestamp"]),
                camera_id=str(row["camera_id"]),
                local_track_id=int(row["local_track_id"]),
                global_id=str(row["global_id"]) if row["global_id"] is not None else None,
                decision=str(row["decision"]),
                appearance_score=(
                    float(row["appearance_score"])
                    if row["appearance_score"] is not None
                    else None
                ),
                temporal_score=(
                    float(row["temporal_score"])
                    if row["temporal_score"] is not None
                    else None
                ),
                camera_score=(
                    float(row["camera_score"])
                    if row["camera_score"] is not None
                    else None
                ),
                motion_score=(
                    float(row["motion_score"])
                    if row["motion_score"] is not None
                    else None
                ),
                combined_confidence=(
                    float(row["combined_confidence"])
                    if row["combined_confidence"] is not None
                    else None
                ),
                reason=str(row["reason"]),
            )
            for row in rows
        ]

    def upsert_unresolved(self, record: UnresolvedObservationRecord) -> None:
        """Persist or update an unresolved observation for evidence or manual review."""
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO unresolved_observations (
                        tracklet_uid, camera_id, local_track_id, first_seen_at,
                        last_seen_at, candidate_ids_json, confidence, reason,
                        status, resolved_global_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tracklet_uid) DO UPDATE SET
                        last_seen_at = excluded.last_seen_at,
                        candidate_ids_json = excluded.candidate_ids_json,
                        confidence = excluded.confidence,
                        reason = excluded.reason,
                        status = excluded.status,
                        resolved_global_id = excluded.resolved_global_id;
                    """,
                    (
                        record.tracklet_uid,
                        record.camera_id,
                        int(record.local_track_id),
                        float(record.first_seen_at),
                        float(record.last_seen_at),
                        json.dumps(record.candidate_ids),
                        float(record.confidence),
                        record.reason,
                        record.status,
                        record.resolved_global_id,
                    ),
                )

    def get_unresolved(
        self, status: Optional[str] = "PENDING"
    ) -> list[UnresolvedObservationRecord]:
        """Return unresolved observation records filtered by status (or all if None)."""
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM unresolved_observations ORDER BY first_seen_at ASC;"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM unresolved_observations WHERE status = ? ORDER BY first_seen_at ASC;",
                    (status,),
                ).fetchall()
            return [
                UnresolvedObservationRecord(
                    tracklet_uid=str(row["tracklet_uid"]),
                    camera_id=str(row["camera_id"]),
                    local_track_id=int(row["local_track_id"]),
                    first_seen_at=float(row["first_seen_at"]),
                    last_seen_at=float(row["last_seen_at"]),
                    candidate_ids=json.loads(row["candidate_ids_json"]),
                    confidence=float(row["confidence"]),
                    reason=str(row["reason"]),
                    status=str(row["status"]),
                    resolved_global_id=(
                        str(row["resolved_global_id"])
                        if row["resolved_global_id"] is not None
                        else None
                    ),
                )
                for row in rows
            ]

    def close(self) -> None:
        """Close the SQLite database connection."""
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "IdentityStorage":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
