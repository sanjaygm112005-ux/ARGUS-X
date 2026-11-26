"""FAISS-backed long-term identity memory for ARGUS-X.

Stores L2-normalized 512-dimensional OSNet appearance embeddings in a FAISS
``IndexFlatIP`` index for cosine similarity search, paired with SQLite metadata
persistence.

Architectural invariants:
- Local BoT-SORT track IDs are stored as metadata only and remain separate from
  global ``identity_id`` values.
- Multiple observations per identity are retained without discarding older
  embeddings.
- Identities are never automatically merged based solely on FAISS similarity.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

# macOS: PyTorch and faiss-cpu each bundle libomp.dylib. Ensure torch
# initializes its OpenMP runtime first if installed, and allow coexistence.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch as _torch  # noqa: F401
except ImportError:
    pass

import faiss
import numpy as np

from config import LOGS_DIR, MemoryConfig, ReIDConfig
from identity_memory.storage import IdentityStorage, ObservationRecord

logger = logging.getLogger(__name__)

_DEFAULT_INDEX_PATH = LOGS_DIR / "argus_memory.faiss"


@dataclass(frozen=True)
class SearchMatch:
    """Single top-K similarity search result from IdentityMemory."""

    embedding_id: str
    vector_id: int
    identity_id: str
    camera_id: str
    local_track_id: Optional[int]
    timestamp: float
    similarity: float


def _normalize_single_embedding(
    embedding: Union[np.ndarray, Sequence[float]],
    expected_dim: int,
) -> np.ndarray:
    """Validate and L2-normalize a single 1D (or 1xD) embedding to shape (1, D)."""
    if embedding is None:
        raise ValueError("Embedding cannot be None.")

    arr = np.asarray(embedding, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Embedding cannot be empty.")

    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(
            f"Expected 1D embedding of length {expected_dim}, got shape {arr.shape}."
        )

    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"Invalid embedding dimension: expected {expected_dim}, got {arr.shape[0]}."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError("Embedding contains NaN or infinite values.")

    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Embedding has zero (or near-zero) norm and cannot be normalized.")

    normalized = np.ascontiguousarray((arr / norm).reshape(1, expected_dim), dtype=np.float32)
    return normalized


class IdentityMemory:
    """Persistent appearance-feature memory using FAISS IndexFlatIP and SQLite."""

    def __init__(
        self,
        embedding_dim: int = ReIDConfig.embedding_dim,
        db_path: Union[str, Path] = MemoryConfig.db_path,
        index_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize FAISS IndexFlatIP and SQLite metadata storage.

        Args:
            embedding_dim: Dimensionality of OSNet embeddings (default 512).
            db_path: Path to SQLite database file, or ``":memory:"``.
            index_path: Optional path to an existing FAISS index file to load,
                or default save location.
        """
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.embedding_dim = int(embedding_dim)
        self.index_path = Path(index_path) if index_path is not None else None
        self._lock = threading.RLock()
        self._storage = IdentityStorage(db_path=db_path)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)
        self._vector_to_identity: dict[int, str] = {}

        if self.index_path is not None and self.index_path.exists():
            self.load_index(self.index_path)
        else:
            self._vector_to_identity = self._storage.get_vector_identity_map()
            if self._vector_to_identity and self._index.ntotal == 0:
                logger.warning(
                    "SQLite contains %d observations but no FAISS index was loaded.",
                    len(self._vector_to_identity),
                )

        logger.info(
            "IdentityMemory ready (dim=%d, ntotal=%d, db=%s)",
            self.embedding_dim,
            self._index.ntotal,
            self._storage.db_path_str,
        )

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the FAISS index."""
        with self._lock:
            return int(self._index.ntotal)

    @property
    def storage(self) -> IdentityStorage:
        """Underlying SQLite metadata and audit storage."""
        return self._storage

    @property
    def vector_to_identity(self) -> dict[int, str]:
        """Read-only copy of the FAISS vector_id -> identity_id mapping."""
        with self._lock:
            return dict(self._vector_to_identity)

    def create_identity(self, identity_id: Optional[str] = None) -> str:
        """Create and persist a new unique global identity ID."""
        gid = identity_id.strip() if identity_id is not None else f"id_{uuid.uuid4().hex[:12]}"
        if not gid:
            raise ValueError("identity_id cannot be empty.")
        with self._lock:
            self._storage.ensure_identity(gid)
        return gid

    def add_observation(
        self,
        embedding: Union[np.ndarray, Sequence[float]],
        identity_id: str,
        camera_id: str = "cam_0",
        local_track_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        embedding_id: Optional[str] = None,
        deduplicate: bool = True,
    ) -> ObservationRecord:
        """Normalize and store a single appearance embedding with metadata.

        Args:
            embedding: 512-dim appearance vector (will be L2-normalized).
            identity_id: Global identity ID (kept separate from local_track_id).
            camera_id: Camera identifier string.
            local_track_id: Optional BoT-SORT local track ID.
            timestamp: Observation epoch timestamp (defaults to current time).
            embedding_id: Optional unique ID for this observation.
            deduplicate: If True, skip adding if the exact same normalized vector
                already exists for this ``identity_id``.

        Returns:
            The persisted ``ObservationRecord``.
        """
        if not isinstance(identity_id, str) or not identity_id.strip():
            raise ValueError("identity_id must be a non-empty string.")
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string.")

        norm_vec = _normalize_single_embedding(embedding, self.embedding_dim)
        emb_hash = hashlib.sha256(norm_vec.tobytes()).hexdigest()

        with self._lock:
            if deduplicate:
                existing = self._storage.find_duplicate(identity_id, emb_hash)
                if existing is not None:
                    logger.info(
                        "Duplicate embedding ignored for identity=%s (existing embedding_id=%s)",
                        identity_id,
                        existing.embedding_id,
                    )
                    return existing

            emb_id = embedding_id or f"emb_{uuid.uuid4().hex[:12]}"
            ts = float(time.time() if timestamp is None else timestamp)
            vector_id = int(self._index.ntotal)

            record = self._storage.add_observation(
                embedding_id=emb_id,
                vector_id=vector_id,
                identity_id=identity_id,
                camera_id=camera_id,
                local_track_id=local_track_id,
                timestamp=ts,
                embedding_hash=emb_hash,
            )
            self._index.add(norm_vec)
            self._vector_to_identity[vector_id] = identity_id
            logger.debug(
                "Added observation embedding_id=%s vector_id=%d identity_id=%s camera=%s",
                emb_id,
                vector_id,
                identity_id,
                camera_id,
            )
            return record

    def add_observations_batch(
        self,
        embeddings: Union[np.ndarray, Sequence[Sequence[float]]],
        identity_ids: Sequence[str],
        camera_ids: Optional[Sequence[str]] = None,
        local_track_ids: Optional[Sequence[Optional[int]]] = None,
        timestamps: Optional[Sequence[Optional[float]]] = None,
        deduplicate: bool = True,
    ) -> list[ObservationRecord]:
        """Store a batch of embeddings and their metadata.

        An empty batch returns ``[]`` without error.
        """
        if embeddings is None:
            raise ValueError("embeddings cannot be None.")

        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.size == 0:
            if len(identity_ids) != 0:
                raise ValueError("Empty embeddings provided with non-empty identity_ids.")
            return []

        if arr.ndim != 2 or arr.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected 2D array of shape (N, {self.embedding_dim}), got {arr.shape}."
            )

        n = arr.shape[0]
        if len(identity_ids) != n:
            raise ValueError("identity_ids length must match number of embeddings.")

        cams = list(camera_ids) if camera_ids is not None else ["cam_0"] * n
        tracks = list(local_track_ids) if local_track_ids is not None else [None] * n
        times = list(timestamps) if timestamps is not None else [None] * n

        if len(cams) != n or len(tracks) != n or len(times) != n:
            raise ValueError("Metadata sequence lengths must match number of embeddings.")

        records: list[ObservationRecord] = []
        for idx in range(n):
            records.append(
                self.add_observation(
                    embedding=arr[idx],
                    identity_id=identity_ids[idx],
                    camera_id=cams[idx],
                    local_track_id=tracks[idx],
                    timestamp=times[idx],
                    deduplicate=deduplicate,
                )
            )
        return records

    def search(
        self,
        query_embedding: Union[np.ndarray, Sequence[float]],
        top_k: int = 5,
    ) -> list[SearchMatch]:
        """Retrieve the top-K visually similar stored observations by cosine similarity.

        Note: High similarity indicates visual resemblance only; it does NOT
        automatically merge identities or prove identity.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if query_embedding is None:
            raise ValueError("query_embedding cannot be None.")

        arr = np.asarray(query_embedding, dtype=np.float32)
        if arr.size == 0:
            return []

        norm_vec = _normalize_single_embedding(arr, self.embedding_dim)

        with self._lock:
            if self._index.ntotal == 0:
                return []

            k = min(int(top_k), int(self._index.ntotal))
            scores, indices = self._index.search(norm_vec, k)

            raw_ids = [int(vid) for vid in indices[0] if int(vid) >= 0]
            metadata_map = self._storage.get_by_vector_ids(raw_ids)

            matches: list[SearchMatch] = []
            for score, vid in zip(scores[0], indices[0]):
                vector_id = int(vid)
                if vector_id < 0:
                    continue
                record = metadata_map.get(vector_id)
                if record is None:
                    logger.debug("Missing SQLite metadata for FAISS vector_id=%d", vector_id)
                    continue
                similarity = float(np.clip(float(score), -1.0, 1.0))
                matches.append(
                    SearchMatch(
                        embedding_id=record.embedding_id,
                        vector_id=vector_id,
                        identity_id=record.identity_id,
                        camera_id=record.camera_id,
                        local_track_id=record.local_track_id,
                        timestamp=record.timestamp,
                        similarity=similarity,
                    )
                )
            return matches

    def get_identity_observations(self, identity_id: str) -> list[ObservationRecord]:
        """Return all stored observations for a given identity."""
        with self._lock:
            return self._storage.get_observations_for_identity(identity_id)

    def list_identities(self) -> list[str]:
        """Return all registered global identity IDs."""
        with self._lock:
            return self._storage.list_identities()

    def save_index(self, index_path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the FAISS index to disk."""
        with self._lock:
            target = Path(index_path) if index_path is not None else (self.index_path or _DEFAULT_INDEX_PATH)
            target.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(target))
            self.index_path = target
            logger.info("Saved FAISS index (%d vectors) to %s", self._index.ntotal, target)
            return target

    def load_index(self, index_path: Union[str, Path]) -> None:
        """Load a persisted FAISS index from disk and sync vector-to-identity map."""
        with self._lock:
            target = Path(index_path)
            if not target.exists():
                raise FileNotFoundError(f"FAISS index file not found: {target}")

            loaded = faiss.read_index(str(target))
            if loaded.d != self.embedding_dim:
                raise ValueError(
                    f"Loaded FAISS index dimension ({loaded.d}) does not match "
                    f"expected embedding_dim ({self.embedding_dim})."
                )

            self._index = loaded
            self.index_path = target
            self._vector_to_identity = self._storage.get_vector_identity_map()
            db_count = self._storage.count_observations()
            if self._index.ntotal != db_count:
                logger.info(
                    "FAISS index vector count (%d) differs from SQLite observation count (%d).",
                    self._index.ntotal,
                    db_count,
                )
            logger.info("Loaded FAISS index (%d vectors) from %s", self._index.ntotal, target)

    def close(self) -> None:
        """Close underlying SQLite storage."""
        self._storage.close()

    def __enter__(self) -> "IdentityMemory":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
