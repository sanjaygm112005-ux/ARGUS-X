"""Tracklet management for single-camera identity recovery in ARGUS-X.

Collects BoT-SORT local track observations, tracks start/end timestamps,
detects when local tracks become inactive, and extracts OSNet appearance
embeddings on a configurable schedule without reloading models.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence

import numpy as np

from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import TrackedObject

if TYPE_CHECKING:
    from association.cost_matrix import KalmanMotionEstimate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackletSample:
    """Single timestamped observation within a local tracklet."""

    timestamp: float
    bbox: tuple[float, float, float, float]
    confidence: float
    embedding: Optional[np.ndarray] = None
    motion_state: Optional[KalmanMotionEstimate] = None


@dataclass
class Tracklet:
    """Short-horizon local tracklet aggregating consecutive BoT-SORT detections."""

    tracklet_uid: str
    camera_id: str
    local_track_id: int
    start_timestamp: float
    end_timestamp: float
    is_active: bool = True
    samples: list[TrackletSample] = field(default_factory=list)
    last_extracted_timestamp: Optional[float] = None

    @property
    def latest_sample(self) -> TrackletSample:
        """Return the most recent observation in this tracklet."""
        if not self.samples:
            raise ValueError(f"Tracklet '{self.tracklet_uid}' has no samples.")
        return self.samples[-1]

    @property
    def embeddings(self) -> list[np.ndarray]:
        """Return all extracted appearance embeddings for this tracklet."""
        return [s.embedding for s in self.samples if s.embedding is not None]

    @property
    def latest_embedding(self) -> Optional[np.ndarray]:
        """Return the most recently extracted embedding, or None."""
        for sample in reversed(self.samples):
            if sample.embedding is not None:
                return sample.embedding
        return None


@dataclass(frozen=True)
class TrackletUpdateResult:
    """Output of a single TrackletManager.update() call."""

    active_tracklets: list[Tracklet]
    newly_started: list[Tracklet]
    newly_inactive: list[Tracklet]


class TrackletManager:
    """Manages local BoT-SORT tracklets and periodic OSNet feature extraction."""

    def __init__(
        self,
        camera_id: str = "cam_0",
        feature_extractor: Optional[PersonFeatureExtractor] = None,
        inactive_timeout_sec: float = 1.0,
        extraction_interval_sec: float = 0.5,
        max_samples_per_tracklet: int = 30,
    ) -> None:
        """Initialize TrackletManager for a single camera.

        Args:
            camera_id: Single-camera identifier.
            feature_extractor: Optional reusable ``PersonFeatureExtractor``.
            inactive_timeout_sec: Seconds without updates before an active
                tracklet is marked inactive.
            extraction_interval_sec: Minimum interval between OSNet embedding
                extractions for the same active tracklet.
            max_samples_per_tracklet: Maximum samples retained per tracklet.
        """
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string.")
        if inactive_timeout_sec < 0.0:
            raise ValueError("inactive_timeout_sec must be >= 0.")
        if extraction_interval_sec < 0.0:
            raise ValueError("extraction_interval_sec must be >= 0.")
        if max_samples_per_tracklet < 1:
            raise ValueError("max_samples_per_tracklet must be >= 1.")

        self.camera_id = camera_id
        self.feature_extractor = feature_extractor
        self.inactive_timeout_sec = float(inactive_timeout_sec)
        self.extraction_interval_sec = float(extraction_interval_sec)
        self.max_samples_per_tracklet = int(max_samples_per_tracklet)

        self._active_by_local_id: dict[int, Tracklet] = {}
        self._inactive_tracklets: dict[str, Tracklet] = {}
        self._uid_counter: int = 0

    def _next_uid(self, local_track_id: int) -> str:
        self._uid_counter += 1
        return f"{self.camera_id}:trk_{local_track_id}:{self._uid_counter}"

    @staticmethod
    def _is_person(obj: TrackedObject) -> bool:
        return obj.class_id == 0 or obj.class_name.strip().lower() == "person"

    def _needs_extraction(self, tracklet: Optional[Tracklet], timestamp: float) -> bool:
        if tracklet is None or tracklet.last_extracted_timestamp is None:
            return True
        return (timestamp - tracklet.last_extracted_timestamp) >= self.extraction_interval_sec

    def update(
        self,
        tracked_objects: Sequence[TrackedObject],
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        precomputed_embeddings: Optional[Mapping[int, np.ndarray]] = None,
        motion_states: Optional[Mapping[int, KalmanMotionEstimate]] = None,
    ) -> TrackletUpdateResult:
        """Ingest current-frame BoT-SORT detections and update tracklet states.

        Args:
            tracked_objects: Detections with local ``track_id`` from BoT-SORT.
            timestamp: Current frame timestamp in seconds.
            frame: Optional BGR frame for OSNet crop extraction.
            precomputed_embeddings: Optional ``{local_track_id: embedding}`` map.
            motion_states: Optional ``{local_track_id: KalmanMotionEstimate}`` map.
        """
        ts = float(time.time() if timestamp is None else timestamp)
        person_objs = [obj for obj in tracked_objects if self._is_person(obj)]

        # Determine which active tracklets need OSNet extraction this frame.
        extracted_map: dict[int, np.ndarray] = {}
        if precomputed_embeddings:
            for tid, emb in precomputed_embeddings.items():
                extracted_map[int(tid)] = np.asarray(emb, dtype=np.float32)

        if frame is not None and self.feature_extractor is not None:
            objs_to_extract = [
                obj
                for obj in person_objs
                if obj.track_id not in extracted_map
                and self._needs_extraction(self._active_by_local_id.get(obj.track_id), ts)
            ]
            if objs_to_extract:
                valid_crops = []
                valid_tids = []
                for obj in objs_to_extract:
                    try:
                        valid_crops.append(self.feature_extractor.crop_person(frame, obj.bbox))
                        valid_tids.append(obj.track_id)
                    except ValueError:
                        continue
                if valid_crops:
                    feats = self.feature_extractor.extract_batch(valid_crops)
                    for feat_row, tid in zip(feats, valid_tids):
                        extracted_map[tid] = feat_row

        seen_local_ids: set[int] = set()
        newly_started: list[Tracklet] = []

        for obj in person_objs:
            tid = int(obj.track_id)
            seen_local_ids.add(tid)
            emb = extracted_map.get(tid)
            motion = motion_states.get(tid) if motion_states else None
            sample = TrackletSample(
                timestamp=ts,
                bbox=tuple(float(v) for v in obj.bbox),  # type: ignore[arg-type]
                confidence=float(obj.confidence),
                embedding=emb,
                motion_state=motion,
            )

            existing = self._active_by_local_id.get(tid)
            if existing is None:
                tracklet = Tracklet(
                    tracklet_uid=self._next_uid(tid),
                    camera_id=self.camera_id,
                    local_track_id=tid,
                    start_timestamp=ts,
                    end_timestamp=ts,
                    is_active=True,
                    samples=[sample],
                    last_extracted_timestamp=ts if emb is not None else None,
                )
                self._active_by_local_id[tid] = tracklet
                newly_started.append(tracklet)
            else:
                existing.end_timestamp = ts
                existing.samples.append(sample)
                if len(existing.samples) > self.max_samples_per_tracklet:
                    existing.samples.pop(0)
                if emb is not None:
                    existing.last_extracted_timestamp = ts

        newly_inactive: list[Tracklet] = []
        for tid in list(self._active_by_local_id.keys()):
            if tid in seen_local_ids:
                continue
            tracklet = self._active_by_local_id[tid]
            if (ts - tracklet.end_timestamp) >= self.inactive_timeout_sec:
                tracklet.is_active = False
                self._active_by_local_id.pop(tid)
                self._inactive_tracklets[tracklet.tracklet_uid] = tracklet
                newly_inactive.append(tracklet)
                logger.debug(
                    "Tracklet %s (local_id=%d) marked inactive at t=%.2f",
                    tracklet.tracklet_uid,
                    tid,
                    ts,
                )

        return TrackletUpdateResult(
            active_tracklets=list(self._active_by_local_id.values()),
            newly_started=newly_started,
            newly_inactive=newly_inactive,
        )

    def get_active_tracklets(self) -> list[Tracklet]:
        """Return all currently active tracklets."""
        return list(self._active_by_local_id.values())

    def get_inactive_tracklets(self) -> list[Tracklet]:
        """Return all retired/inactive tracklets."""
        return list(self._inactive_tracklets.values())

    def retire_all(self) -> list[Tracklet]:
        """Mark all currently active tracklets inactive (e.g., on camera disconnect)."""
        retired: list[Tracklet] = []
        for tid in list(self._active_by_local_id.keys()):
            trk = self._active_by_local_id.pop(tid)
            trk.is_active = False
            self._inactive_tracklets[trk.tracklet_uid] = trk
            retired.append(trk)
        return retired
