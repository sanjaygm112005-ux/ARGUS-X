"""Single-camera global identity management and recovery for ARGUS-X.

Connects BoT-SORT tracklets, OSNet embeddings, FAISS IdentityMemory, and
Hungarian DataAssociator to recover identities after occlusion or temporary
disappearance within a single camera.

Architectural invariants:
- Single-camera recovery only (cross-camera recovery is a separate milestone).
- Local BoT-SORT track IDs remain separate from global identity IDs.
- Existing global identities are never merged automatically.
- Historical observations are never deleted when a local track disappears.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from identity_memory.memory import IdentityMemory
from identity_memory.tracklet_manager import Tracklet, TrackletManager
from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import ObjectTracker

if TYPE_CHECKING:
    from association.associator import DataAssociator
    from association.cost_matrix import IdentityCandidate, TrackletObservation

logger = logging.getLogger(__name__)


class IdentityState(str, Enum):
    """Lifecycle states for a tracked global identity."""

    ACTIVE = "ACTIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    RECOVERED = "RECOVERED"
    UNCERTAIN = "UNCERTAIN"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class GlobalIdentityRecord:
    """State and history metadata for a single global identity."""

    global_id: str
    camera_id: str
    state: IdentityState
    created_at: float
    last_seen_at: float
    active_local_track_id: Optional[int] = None
    active_tracklet_uid: Optional[str] = None
    local_track_history: list[int] = field(default_factory=list)
    last_stored_embedding_ts: Optional[float] = None


@dataclass(frozen=True)
class RecoveryOutput:
    """Structured per-tracklet output for visualization and downstream APIs."""

    camera_id: str
    local_track_id: int
    global_id: Optional[str]
    bbox: tuple[float, float, float, float]
    confidence: float
    identity_state: IdentityState
    recovery_decision: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        data = asdict(self)
        data["identity_state"] = self.identity_state.value
        return data


class GlobalIdentityManager:
    """Manages global identity lifecycles and single-camera tracklet recovery."""

    def __init__(
        self,
        camera_id: str = "cam_0",
        memory: Optional[IdentityMemory] = None,
        associator: Optional[DataAssociator] = None,
        lost_timeout_sec: float = 30.0,
        expire_timeout_sec: float = 300.0,
        db_path: Union[str, Path] = ":memory:",
    ) -> None:
        """Initialize single-camera GlobalIdentityManager.

        Args:
            camera_id: Single-camera identifier enforced for all tracklets.
            memory: Optional ``IdentityMemory`` instance.
            associator: Optional ``DataAssociator`` instance.
            lost_timeout_sec: Threshold distinguishing brief occlusion from
                longer temporary loss (informational/monitoring boundary).
            expire_timeout_sec: Seconds after which a lost identity becomes
                ``EXPIRED`` and is no longer eligible for automatic recovery.
            db_path: SQLite path if creating an internal ``IdentityMemory``.
        """
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ValueError("camera_id must be a non-empty string.")
        if lost_timeout_sec < 0.0:
            raise ValueError("lost_timeout_sec must be >= 0.")
        if expire_timeout_sec < lost_timeout_sec:
            raise ValueError("expire_timeout_sec must be >= lost_timeout_sec.")

        self.camera_id = camera_id
        self.lost_timeout_sec = float(lost_timeout_sec)
        self.expire_timeout_sec = float(expire_timeout_sec)
        self._owns_memory = memory is None
        self.memory = memory or IdentityMemory(db_path=db_path)

        if associator is not None:
            self.associator = associator
        else:
            from association.associator import DataAssociator
            from association.cost_matrix import AssociationConfig, CameraTransitionRule

            config = AssociationConfig(max_time_gap_sec=self.expire_timeout_sec)
            same_cam_rule = CameraTransitionRule(
                source_camera=self.camera_id,
                target_camera=self.camera_id,
                min_travel_sec=0.0,
                max_travel_sec=self.expire_timeout_sec,
                allowed=True,
            )
            self.associator = DataAssociator(
                memory=self.memory,
                config=config,
                transition_rules=[same_cam_rule],
            )

        self._identities: dict[str, GlobalIdentityRecord] = {}
        self._tracklet_to_gid: dict[str, str] = {}

    def _generate_global_id(self) -> str:
        gid = f"gid_{uuid.uuid4().hex[:12]}"
        self.memory.create_identity(gid)
        return gid

    def get_identity(self, global_id: str) -> Optional[GlobalIdentityRecord]:
        """Return the lifecycle record for ``global_id``, if present."""
        return self._identities.get(global_id)

    def list_identities(self) -> list[GlobalIdentityRecord]:
        """Return all managed global identity records."""
        return list(self._identities.values())

    def update_lifecycles(
        self,
        current_timestamp: float,
        active_tracklet_uids: Sequence[str],
    ) -> None:
        """Transition disappeared tracks to TEMPORARILY_LOST and stale ones to EXPIRED.

        Never deletes historical observations from ``IdentityMemory``.
        """
        active_set = set(active_tracklet_uids)
        for record in self._identities.values():
            if record.state in (IdentityState.ACTIVE, IdentityState.RECOVERED):
                if record.active_tracklet_uid not in active_set:
                    record.state = IdentityState.TEMPORARILY_LOST
                    record.active_local_track_id = None
                    record.active_tracklet_uid = None
                    logger.debug(
                        "Identity %s transitioned to TEMPORARILY_LOST at t=%.2f",
                        record.global_id,
                        current_timestamp,
                    )

            if record.state == IdentityState.TEMPORARILY_LOST:
                elapsed = float(current_timestamp) - record.last_seen_at
                if elapsed > self.expire_timeout_sec:
                    record.state = IdentityState.EXPIRED
                    logger.info(
                        "Identity %s transitioned to EXPIRED (elapsed=%.2fs)",
                        record.global_id,
                        elapsed,
                    )

    def _create_new_identity(
        self,
        tracklet: Tracklet,
        embedding: np.ndarray,
        timestamp: float,
        decision_label: str,
    ) -> RecoveryOutput:
        gid = self._generate_global_id()
        self.memory.add_observation(
            embedding=embedding,
            identity_id=gid,
            camera_id=self.camera_id,
            local_track_id=tracklet.local_track_id,
            timestamp=timestamp,
        )
        record = GlobalIdentityRecord(
            global_id=gid,
            camera_id=self.camera_id,
            state=IdentityState.ACTIVE,
            created_at=timestamp,
            last_seen_at=timestamp,
            active_local_track_id=tracklet.local_track_id,
            active_tracklet_uid=tracklet.tracklet_uid,
            local_track_history=[tracklet.local_track_id],
            last_stored_embedding_ts=timestamp,
        )
        self._identities[gid] = record
        self._tracklet_to_gid[tracklet.tracklet_uid] = gid
        return RecoveryOutput(
            camera_id=self.camera_id,
            local_track_id=tracklet.local_track_id,
            global_id=gid,
            bbox=tracklet.latest_sample.bbox,
            confidence=1.0,
            identity_state=IdentityState.ACTIVE,
            recovery_decision=decision_label,
        )

    def _build_lost_candidates(
        self, tracklet_obs: Sequence[TrackletObservation]
    ) -> list[IdentityCandidate]:
        """Query FAISS for TEMPORARILY_LOST same-camera identities only."""
        from association.cost_matrix import IdentityCandidate

        if not tracklet_obs:
            return []

        best_sim_by_gid: dict[str, float] = {}
        for obs in tracklet_obs:
            hits = self.memory.search(obs.embedding, top_k=10)
            for hit in hits:
                if hit.camera_id != self.camera_id:
                    continue
                rec = self._identities.get(hit.identity_id)
                if rec is None or rec.state != IdentityState.TEMPORARILY_LOST:
                    continue
                prev = best_sim_by_gid.get(hit.identity_id, -1.0)
                if hit.similarity > prev:
                    best_sim_by_gid[hit.identity_id] = hit.similarity

        candidates: list[IdentityCandidate] = []
        for gid, best_sim in best_sim_by_gid.items():
            hist_obs = self.memory.get_identity_observations(gid)
            if not hist_obs:
                continue
            candidates.append(
                IdentityCandidate(
                    identity_id=gid,
                    observations=hist_obs,
                    best_similarity=best_sim,
                )
            )
        return candidates

    def process_tracklets(
        self,
        active_tracklets: Sequence[Tracklet],
        current_timestamp: Optional[float] = None,
    ) -> list[RecoveryOutput]:
        """Evaluate active tracklets against global identity memory.

        Args:
            active_tracklets: Currently active local tracklets from ``TrackletManager``.
            current_timestamp: Current frame timestamp in seconds.

        Returns:
            List of ``RecoveryOutput`` records for visualization/logging.
        """
        ts = float(time.time() if current_timestamp is None else current_timestamp)

        for tracklet in active_tracklets:
            if tracklet.camera_id != self.camera_id:
                raise ValueError(
                    f"Single-camera GlobalIdentityManager({self.camera_id!r}) "
                    f"received tracklet from camera {tracklet.camera_id!r}."
                )

        self.update_lifecycles(ts, [t.tracklet_uid for t in active_tracklets])

        outputs_by_uid: dict[str, RecoveryOutput] = {}
        unbound_tracklets: list[Tracklet] = []

        for tracklet in active_tracklets:
            gid = self._tracklet_to_gid.get(tracklet.tracklet_uid)
            if gid is not None:
                rec = self._identities[gid]
                rec.last_seen_at = ts
                rec.active_local_track_id = tracklet.local_track_id
                latest_emb = tracklet.latest_embedding
                if (
                    latest_emb is not None
                    and tracklet.last_extracted_timestamp is not None
                    and (
                        rec.last_stored_embedding_ts is None
                        or tracklet.last_extracted_timestamp > rec.last_stored_embedding_ts
                    )
                ):
                    self.memory.add_observation(
                        embedding=latest_emb,
                        identity_id=gid,
                        camera_id=self.camera_id,
                        local_track_id=tracklet.local_track_id,
                        timestamp=tracklet.last_extracted_timestamp,
                    )
                    rec.last_stored_embedding_ts = tracklet.last_extracted_timestamp

                outputs_by_uid[tracklet.tracklet_uid] = RecoveryOutput(
                    camera_id=self.camera_id,
                    local_track_id=tracklet.local_track_id,
                    global_id=gid,
                    bbox=tracklet.latest_sample.bbox,
                    confidence=tracklet.latest_sample.confidence,
                    identity_state=rec.state,
                    recovery_decision="ACTIVE_TRACK",
                )
            else:
                unbound_tracklets.append(tracklet)

        # Evaluate unbound tracklets that have an extracted embedding.
        from association.cost_matrix import ConfidenceDecision, TrackletObservation

        ready_tracklets: list[Tracklet] = []
        obs_list: list[TrackletObservation] = []
        for tracklet in unbound_tracklets:
            emb = tracklet.latest_embedding
            if emb is None:
                outputs_by_uid[tracklet.tracklet_uid] = RecoveryOutput(
                    camera_id=self.camera_id,
                    local_track_id=tracklet.local_track_id,
                    global_id=None,
                    bbox=tracklet.latest_sample.bbox,
                    confidence=0.0,
                    identity_state=IdentityState.UNKNOWN,
                    recovery_decision="AWAITING_EMBEDDING",
                )
                continue
            ready_tracklets.append(tracklet)
            obs_list.append(
                TrackletObservation(
                    local_track_id=tracklet.local_track_id,
                    camera_id=self.camera_id,
                    timestamp=ts,
                    embedding=emb,
                    motion_state=tracklet.latest_sample.motion_state,
                )
            )

        if ready_tracklets:
            candidates = self._build_lost_candidates(obs_list)
            assoc_result = self.associator.associate(obs_list, candidates=candidates)

            matched_by_local_id = {m.local_track_id: m for m in assoc_result.matches}
            for tracklet in ready_tracklets:
                match = matched_by_local_id.get(tracklet.local_track_id)
                emb = tracklet.latest_embedding
                assert emb is not None

                if match is None:
                    decision = "NEW_IDENTITY" if not candidates else "REJECT_NEW_IDENTITY"
                    outputs_by_uid[tracklet.tracklet_uid] = self._create_new_identity(
                        tracklet, emb, ts, decision
                    )
                elif match.decision == ConfidenceDecision.ACCEPT:
                    gid = match.identity_id
                    rec = self._identities[gid]
                    rec.state = IdentityState.RECOVERED
                    rec.last_seen_at = ts
                    rec.active_local_track_id = tracklet.local_track_id
                    rec.active_tracklet_uid = tracklet.tracklet_uid
                    rec.local_track_history.append(tracklet.local_track_id)
                    rec.last_stored_embedding_ts = ts
                    self._tracklet_to_gid[tracklet.tracklet_uid] = gid
                    self.memory.add_observation(
                        embedding=emb,
                        identity_id=gid,
                        camera_id=self.camera_id,
                        local_track_id=tracklet.local_track_id,
                        timestamp=ts,
                    )
                    outputs_by_uid[tracklet.tracklet_uid] = RecoveryOutput(
                        camera_id=self.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=gid,
                        bbox=tracklet.latest_sample.bbox,
                        confidence=match.confidence,
                        identity_state=IdentityState.RECOVERED,
                        recovery_decision="RECOVERED_ACCEPT",
                    )
                else:
                    # UNCERTAIN: keep unresolved for additional evidence.
                    outputs_by_uid[tracklet.tracklet_uid] = RecoveryOutput(
                        camera_id=self.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=None,
                        bbox=tracklet.latest_sample.bbox,
                        confidence=match.confidence,
                        identity_state=IdentityState.UNKNOWN,
                        recovery_decision="UNCERTAIN",
                    )

        return [outputs_by_uid[t.tracklet_uid] for t in active_tracklets]

    def close(self) -> None:
        """Close IdentityMemory if owned by this manager."""
        if self._owns_memory:
            self.memory.close()


class SingleCameraRecoveryPipeline:
    """End-to-end single-camera pipeline connecting YOLO+BoT-SORT, OSNet, FAISS,
    Hungarian DataAssociator, and GlobalIdentityManager.
    """

    def __init__(
        self,
        camera_id: str = "cam_0",
        tracker: Optional[ObjectTracker] = None,
        feature_extractor: Optional[PersonFeatureExtractor] = None,
        tracklet_manager: Optional[TrackletManager] = None,
        identity_manager: Optional[GlobalIdentityManager] = None,
    ) -> None:
        self.camera_id = camera_id
        self.tracker = tracker or ObjectTracker()
        self.feature_extractor = feature_extractor or PersonFeatureExtractor()
        self.tracklet_manager = tracklet_manager or TrackletManager(
            camera_id=camera_id,
            feature_extractor=self.feature_extractor,
        )
        self.identity_manager = identity_manager or GlobalIdentityManager(
            camera_id=camera_id,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> list[RecoveryOutput]:
        """Run detection/tracking, feature extraction, and identity recovery on one frame."""
        ts = float(time.time() if timestamp is None else timestamp)
        tracked = self.tracker.track_frame(frame)
        update_res = self.tracklet_manager.update(tracked, timestamp=ts, frame=frame)
        return self.identity_manager.process_tracklets(
            update_res.active_tracklets,
            current_timestamp=ts,
        )
