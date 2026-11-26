"""Multi-camera global identity recovery for ARGUS-X.

Coordinates shared FAISS IdentityMemory, directed CameraTransitionRule
validation, and Hungarian DataAssociator across multiple independent camera
streams without merging global identities or comparing uncalibrated pixel
coordinates across cameras.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np

from identity_memory.identity_manager import IdentityState
from identity_memory.memory import IdentityMemory
from identity_memory.storage import (
    AuditLogEntry,
    LifecycleSnapshot,
    UnresolvedObservationRecord,
)
from identity_memory.tracklet_manager import Tracklet

if TYPE_CHECKING:
    from association.associator import AssociationMatch, DataAssociator
    from association.cost_matrix import (
        AssociationConfig,
        CameraTransitionRule,
        IdentityCandidate,
        PairEvaluation,
        TrackletObservation,
    )

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConflictEvent:
    """Recorded conflict when an assignment would violate single-person physics."""

    timestamp: float
    camera_id: str
    local_track_id: int
    conflicting_global_id: str
    conflict_type: str
    reason: str


@dataclass
class MultiCameraIdentityRecord:
    """Global identity lifecycle and per-camera local track history."""

    global_id: str
    state: IdentityState
    created_at: float
    last_seen_at: float
    current_camera_id: Optional[str] = None
    active_tracks_by_camera: dict[str, tuple[int, str]] = field(default_factory=dict)
    camera_track_history: dict[str, list[int]] = field(default_factory=dict)
    camera_sequence: list[str] = field(default_factory=list)
    last_stored_ts_by_tracklet: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiCameraRecoveryOutput:
    """Structured output for multi-camera visualization and downstream APIs."""

    camera_id: str
    local_track_id: int
    global_id: Optional[str]
    bounding_box: tuple[float, float, float, float]
    confidence: float
    identity_state: IdentityState
    recovery_decision: str
    previous_camera_id: Optional[str] = None
    previous_observation_timestamp: Optional[float] = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        data = asdict(self)
        data["identity_state"] = self.identity_state.value
        return data


class MultiCameraIdentityManager:
    """Manages cross-camera global identities using shared FAISS + Hungarian association."""

    def __init__(
        self,
        memory: Optional[IdentityMemory] = None,
        associator: Optional[DataAssociator] = None,
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
        config: Optional[AssociationConfig] = None,
        expire_timeout_sec: float = 3600.0,
        db_path: Union[str, Path] = ":memory:",
        index_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize MultiCameraIdentityManager.

        Args:
            memory: Shared ``IdentityMemory`` across all cameras.
            associator: Optional preconfigured ``DataAssociator``.
            transition_rules: Directed ``CameraTransitionRule`` constraints.
            config: Optional ``AssociationConfig``.
            expire_timeout_sec: Seconds after last observation before a
                ``TEMPORARILY_LOST`` identity transitions to ``EXPIRED``.
            db_path: SQLite path if creating an internal ``IdentityMemory``.
            index_path: Optional FAISS index path for persistence/restart recovery.
        """
        if expire_timeout_sec <= 0.0:
            raise ValueError("expire_timeout_sec must be positive.")

        self.expire_timeout_sec = float(expire_timeout_sec)
        self._owns_memory = memory is None
        self.memory = memory or IdentityMemory(db_path=db_path, index_path=index_path)

        if associator is not None:
            self.associator = associator
        else:
            from association.associator import DataAssociator
            from association.cost_matrix import AssociationConfig

            assoc_cfg = config or AssociationConfig(max_time_gap_sec=self.expire_timeout_sec)
            self.associator = DataAssociator(
                memory=self.memory,
                config=assoc_cfg,
                transition_rules=transition_rules,
            )

        self._identities: dict[str, MultiCameraIdentityRecord] = {}
        self._tracklet_to_gid: dict[str, str] = {}
        self._conflict_events: list[ConflictEvent] = []
        self._restore_from_storage()

    def _persist_lifecycle(self, record: MultiCameraIdentityRecord) -> None:
        self.memory.storage.upsert_lifecycle(
            LifecycleSnapshot(
                global_id=record.global_id,
                state=record.state.value,
                created_at=record.created_at,
                last_seen_at=record.last_seen_at,
                current_camera_id=record.current_camera_id,
                camera_track_history=record.camera_track_history,
                camera_sequence=record.camera_sequence,
            )
        )

    def _restore_from_storage(self) -> None:
        """Restore persisted global identity lifecycles after application restart."""
        snapshots = self.memory.storage.load_lifecycles()
        for snap in snapshots:
            restored_state = IdentityState(snap.state)
            if restored_state in (IdentityState.ACTIVE, IdentityState.RECOVERED):
                # Live camera streams are gone after restart; mark TEMPORARILY_LOST
                # so identities can be cleanly recovered when streams resume.
                restored_state = IdentityState.TEMPORARILY_LOST
            self._identities[snap.global_id] = MultiCameraIdentityRecord(
                global_id=snap.global_id,
                state=restored_state,
                created_at=snap.created_at,
                last_seen_at=snap.last_seen_at,
                current_camera_id=snap.current_camera_id,
                active_tracks_by_camera={},
                camera_track_history=snap.camera_track_history,
                camera_sequence=snap.camera_sequence,
            )
        if snapshots:
            logger.info("Restored %d global identity lifecycles from storage.", len(snapshots))

    def _record_audit(
        self,
        timestamp: float,
        camera_id: str,
        local_track_id: int,
        global_id: Optional[str],
        decision: str,
        reason: str,
        evaluation: Optional[PairEvaluation] = None,
        combined_confidence: Optional[float] = None,
    ) -> None:
        self.memory.storage.add_audit_log(
            AuditLogEntry(
                timestamp=float(timestamp),
                camera_id=camera_id,
                local_track_id=int(local_track_id),
                global_id=global_id,
                decision=decision,
                appearance_score=evaluation.appearance_score if evaluation else None,
                temporal_score=evaluation.temporal_score if evaluation else None,
                camera_score=evaluation.camera_score if evaluation else None,
                motion_score=evaluation.motion_score if evaluation else None,
                combined_confidence=(
                    evaluation.confidence if evaluation else combined_confidence
                ),
                reason=reason,
            )
        )

    def _generate_global_id(self) -> str:
        gid = f"gid_{uuid.uuid4().hex[:12]}"
        self.memory.create_identity(gid)
        return gid

    def get_identity(self, global_id: str) -> Optional[MultiCameraIdentityRecord]:
        """Return the multi-camera record for ``global_id``, if present."""
        return self._identities.get(global_id)

    def list_identities(self) -> list[MultiCameraIdentityRecord]:
        """Return all managed global identity records."""
        return list(self._identities.values())

    def update_lifecycles(
        self,
        current_timestamp: float,
        active_tracklets: Sequence[Tracklet],
    ) -> None:
        """Update ACTIVE -> TEMPORARILY_LOST -> EXPIRED states without deleting history."""
        active_uids = {t.tracklet_uid for t in active_tracklets}
        for record in self._identities.values():
            for cam_id in list(record.active_tracks_by_camera.keys()):
                _, uid = record.active_tracks_by_camera[cam_id]
                if uid not in active_uids:
                    record.active_tracks_by_camera.pop(cam_id)

            if record.state in (IdentityState.ACTIVE, IdentityState.RECOVERED):
                if not record.active_tracks_by_camera:
                    record.state = IdentityState.TEMPORARILY_LOST
                    self._persist_lifecycle(record)
                    logger.debug(
                        "Global identity %s is TEMPORARILY_LOST at t=%.2f",
                        record.global_id,
                        current_timestamp,
                    )

            if record.state == IdentityState.TEMPORARILY_LOST:
                elapsed = float(current_timestamp) - record.last_seen_at
                if elapsed > self.expire_timeout_sec:
                    record.state = IdentityState.EXPIRED
                    self._persist_lifecycle(record)
                    self._record_audit(
                        timestamp=float(current_timestamp),
                        camera_id=record.current_camera_id or "unknown",
                        local_track_id=-1,
                        global_id=record.global_id,
                        decision="EXPIRED",
                        reason=f"expired_after_{elapsed:.2f}s",
                    )
                    logger.info(
                        "Global identity %s expired after %.2fs",
                        record.global_id,
                        elapsed,
                    )

    def _create_new_identity(
        self,
        tracklet: Tracklet,
        embedding: np.ndarray,
        timestamp: float,
        decision_label: str,
    ) -> MultiCameraRecoveryOutput:
        gid = self._generate_global_id()
        self.memory.add_observation(
            embedding=embedding,
            identity_id=gid,
            camera_id=tracklet.camera_id,
            local_track_id=tracklet.local_track_id,
            timestamp=timestamp,
        )
        record = MultiCameraIdentityRecord(
            global_id=gid,
            state=IdentityState.ACTIVE,
            created_at=timestamp,
            last_seen_at=timestamp,
            current_camera_id=tracklet.camera_id,
            active_tracks_by_camera={
                tracklet.camera_id: (tracklet.local_track_id, tracklet.tracklet_uid)
            },
            camera_track_history={tracklet.camera_id: [tracklet.local_track_id]},
            camera_sequence=[tracklet.camera_id],
            last_stored_ts_by_tracklet={tracklet.tracklet_uid: timestamp},
        )
        self._identities[gid] = record
        self._tracklet_to_gid[tracklet.tracklet_uid] = gid
        self._persist_lifecycle(record)
        self._record_audit(
            timestamp=timestamp,
            camera_id=tracklet.camera_id,
            local_track_id=tracklet.local_track_id,
            global_id=gid,
            decision=decision_label,
            combined_confidence=1.0,
            reason=f"created_new_identity:{decision_label.lower()}",
        )
        return MultiCameraRecoveryOutput(
            camera_id=tracklet.camera_id,
            local_track_id=tracklet.local_track_id,
            global_id=gid,
            bounding_box=tracklet.latest_sample.bbox,
            confidence=1.0,
            identity_state=IdentityState.ACTIVE,
            recovery_decision=decision_label,
            previous_camera_id=None,
            previous_observation_timestamp=None,
        )

    def _build_lost_candidates(
        self, tracklet_obs: Sequence[TrackletObservation]
    ) -> list[IdentityCandidate]:
        """Query shared FAISS memory for TEMPORARILY_LOST global identities only.

        Identities currently ACTIVE/RECOVERED in another camera or EXPIRED are
        excluded so simultaneous appearances never hijack an active identity.
        """
        from association.cost_matrix import IdentityCandidate

        if not tracklet_obs:
            return []

        best_sim_by_gid: dict[str, float] = {}
        for obs in tracklet_obs:
            hits = self.memory.search(obs.embedding, top_k=10)
            for hit in hits:
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
    ) -> list[MultiCameraRecoveryOutput]:
        """Evaluate active tracklets across all cameras and recover global IDs."""
        ts = float(time.time() if current_timestamp is None else current_timestamp)
        self.update_lifecycles(ts, active_tracklets)

        outputs_by_uid: dict[str, MultiCameraRecoveryOutput] = {}
        unbound_tracklets: list[Tracklet] = []

        for tracklet in active_tracklets:
            gid = self._tracklet_to_gid.get(tracklet.tracklet_uid)
            if gid is not None:
                rec = self._identities[gid]
                rec.last_seen_at = ts
                rec.current_camera_id = tracklet.camera_id
                rec.active_tracks_by_camera[tracklet.camera_id] = (
                    tracklet.local_track_id,
                    tracklet.tracklet_uid,
                )

                latest_emb = tracklet.latest_embedding
                last_stored = rec.last_stored_ts_by_tracklet.get(tracklet.tracklet_uid)
                if (
                    latest_emb is not None
                    and tracklet.last_extracted_timestamp is not None
                    and (last_stored is None or tracklet.last_extracted_timestamp > last_stored)
                ):
                    self.memory.add_observation(
                        embedding=latest_emb,
                        identity_id=gid,
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        timestamp=tracklet.last_extracted_timestamp,
                    )
                    rec.last_stored_ts_by_tracklet[tracklet.tracklet_uid] = (
                        tracklet.last_extracted_timestamp
                    )

                outputs_by_uid[tracklet.tracklet_uid] = MultiCameraRecoveryOutput(
                    camera_id=tracklet.camera_id,
                    local_track_id=tracklet.local_track_id,
                    global_id=gid,
                    bounding_box=tracklet.latest_sample.bbox,
                    confidence=tracklet.latest_sample.confidence,
                    identity_state=rec.state,
                    recovery_decision="ACTIVE_TRACK",
                    previous_camera_id=None,
                    previous_observation_timestamp=None,
                )
            else:
                unbound_tracklets.append(tracklet)

        from association.cost_matrix import ConfidenceDecision, TrackletObservation

        ready_tracklets: list[Tracklet] = []
        obs_list: list[TrackletObservation] = []

        for idx, tracklet in enumerate(unbound_tracklets):
            emb = tracklet.latest_embedding
            if emb is None:
                outputs_by_uid[tracklet.tracklet_uid] = MultiCameraRecoveryOutput(
                    camera_id=tracklet.camera_id,
                    local_track_id=tracklet.local_track_id,
                    global_id=None,
                    bounding_box=tracklet.latest_sample.bbox,
                    confidence=0.0,
                    identity_state=IdentityState.UNKNOWN,
                    recovery_decision="AWAITING_EMBEDDING",
                )
                continue
            obs_idx = len(ready_tracklets)
            ready_tracklets.append(tracklet)
            # Use obs_idx as temporary assignment key so identical local_track_ids
            # across different cameras never collide during Hungarian matching.
            obs_list.append(
                TrackletObservation(
                    local_track_id=obs_idx,
                    camera_id=tracklet.camera_id,
                    timestamp=ts,
                    embedding=emb,
                    motion_state=tracklet.latest_sample.motion_state,
                )
            )

        if ready_tracklets:
            candidates = self._build_lost_candidates(obs_list)
            cand_by_id = {c.identity_id: c for c in candidates}
            assoc_result = self.associator.associate(obs_list, candidates=candidates)

            matched_by_uid: dict[str, AssociationMatch] = {}
            for match in assoc_result.matches:
                trk = ready_tracklets[match.local_track_id]
                matched_by_uid[trk.tracklet_uid] = match

            for tracklet in ready_tracklets:
                emb = tracklet.latest_embedding
                assert emb is not None
                match = matched_by_uid.get(tracklet.tracklet_uid)

                if match is None:
                    decision = "NEW_IDENTITY" if not candidates else "REJECT_NEW_IDENTITY"
                    outputs_by_uid[tracklet.tracklet_uid] = self._create_new_identity(
                        tracklet, emb, ts, decision
                    )
                    continue

                gid = match.identity_id
                conf = match.confidence
                decision = match.decision
                ev = match.evaluation
                cand = cand_by_id[gid]
                prev_obs = cand.latest_observation

                if decision == ConfidenceDecision.ACCEPT:
                    rec = self._identities[gid]
                    # Conflict check: prevent assigning a global ID that is already
                    # active in another camera or already bound in this step.
                    if rec.active_tracks_by_camera:
                        conflict_reason = (
                            f"simultaneous_active_conflict:active_in_"
                            f"{list(rec.active_tracks_by_camera.keys())}"
                        )
                        self._conflict_events.append(
                            ConflictEvent(
                                timestamp=ts,
                                camera_id=tracklet.camera_id,
                                local_track_id=tracklet.local_track_id,
                                conflicting_global_id=gid,
                                conflict_type="SIMULTANEOUS_ACTIVE_IDENTITY",
                                reason=conflict_reason,
                            )
                        )
                        self._record_audit(
                            timestamp=ts,
                            camera_id=tracklet.camera_id,
                            local_track_id=tracklet.local_track_id,
                            global_id=gid,
                            decision="CONFLICT_BLOCKED",
                            evaluation=ev,
                            reason=conflict_reason,
                        )
                        outputs_by_uid[tracklet.tracklet_uid] = self._create_new_identity(
                            tracklet, emb, ts, "CONFLICT_NEW_IDENTITY"
                        )
                        continue

                    rec.state = IdentityState.RECOVERED
                    rec.last_seen_at = ts
                    rec.current_camera_id = tracklet.camera_id
                    rec.active_tracks_by_camera[tracklet.camera_id] = (
                        tracklet.local_track_id,
                        tracklet.tracklet_uid,
                    )
                    rec.camera_track_history.setdefault(tracklet.camera_id, []).append(
                        tracklet.local_track_id
                    )
                    if not rec.camera_sequence or rec.camera_sequence[-1] != tracklet.camera_id:
                        rec.camera_sequence.append(tracklet.camera_id)
                    rec.last_stored_ts_by_tracklet[tracklet.tracklet_uid] = ts
                    self._tracklet_to_gid[tracklet.tracklet_uid] = gid
                    self._persist_lifecycle(rec)
                    self._record_audit(
                        timestamp=ts,
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=gid,
                        decision="RECOVERED_ACCEPT",
                        evaluation=ev,
                        reason=f"recovered_from_{prev_obs.camera_id}",
                    )

                    self.memory.add_observation(
                        embedding=emb,
                        identity_id=gid,
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        timestamp=ts,
                        deduplicate=False,
                    )
                    outputs_by_uid[tracklet.tracklet_uid] = MultiCameraRecoveryOutput(
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=gid,
                        bounding_box=tracklet.latest_sample.bbox,
                        confidence=conf,
                        identity_state=IdentityState.RECOVERED,
                        recovery_decision="RECOVERED_ACCEPT",
                        previous_camera_id=prev_obs.camera_id,
                        previous_observation_timestamp=prev_obs.timestamp,
                    )
                else:
                    # UNCERTAIN: keep unresolved for additional evidence or manual review.
                    reason = ev.gate_reason or "borderline_confidence"
                    self.memory.storage.upsert_unresolved(
                        UnresolvedObservationRecord(
                            tracklet_uid=tracklet.tracklet_uid,
                            camera_id=tracklet.camera_id,
                            local_track_id=tracklet.local_track_id,
                            first_seen_at=tracklet.start_timestamp,
                            last_seen_at=ts,
                            candidate_ids=[gid],
                            confidence=conf,
                            reason=reason,
                            status="PENDING",
                        )
                    )
                    self._record_audit(
                        timestamp=ts,
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=gid,
                        decision="UNCERTAIN",
                        evaluation=ev,
                        reason=reason,
                    )
                    outputs_by_uid[tracklet.tracklet_uid] = MultiCameraRecoveryOutput(
                        camera_id=tracklet.camera_id,
                        local_track_id=tracklet.local_track_id,
                        global_id=None,
                        bounding_box=tracklet.latest_sample.bbox,
                        confidence=conf,
                        identity_state=IdentityState.UNKNOWN,
                        recovery_decision="UNCERTAIN",
                        previous_camera_id=prev_obs.camera_id,
                        previous_observation_timestamp=prev_obs.timestamp,
                    )

        return [outputs_by_uid[t.tracklet_uid] for t in active_tracklets]

    def handle_camera_disconnect(
        self,
        camera_id: str,
        timestamp: Optional[float] = None,
    ) -> list[str]:
        """Handle interrupted camera stream cleanly without deleting identity history.

        Idempotent under repeated disconnections. Returns affected global IDs.
        """
        ts = float(time.time() if timestamp is None else timestamp)
        affected: list[str] = []
        for record in self._identities.values():
            if camera_id in record.active_tracks_by_camera:
                local_id, _ = record.active_tracks_by_camera.pop(camera_id)
                if not record.active_tracks_by_camera and record.state in (
                    IdentityState.ACTIVE,
                    IdentityState.RECOVERED,
                ):
                    record.state = IdentityState.TEMPORARILY_LOST
                self._persist_lifecycle(record)
                self._record_audit(
                    timestamp=ts,
                    camera_id=camera_id,
                    local_track_id=local_id,
                    global_id=record.global_id,
                    decision="CAMERA_DISCONNECTED",
                    reason=f"camera_stream_interrupted:{camera_id}",
                )
                affected.append(record.global_id)
        return affected

    def check_simultaneous_conflict(
        self,
        global_id: str,
        camera_id: str,
        local_track_id: int,
        timestamp: float,
    ) -> bool:
        """Return True and record a ConflictEvent if ``global_id`` is active elsewhere."""
        rec = self._identities.get(global_id)
        if rec is None:
            return False
        other_cams = [c for c in rec.active_tracks_by_camera if c != camera_id]
        if not other_cams:
            return False

        reason = f"simultaneous_active_in:{other_cams}"
        self._conflict_events.append(
            ConflictEvent(
                timestamp=float(timestamp),
                camera_id=camera_id,
                local_track_id=int(local_track_id),
                conflicting_global_id=global_id,
                conflict_type="SIMULTANEOUS_CROSS_CAMERA_CONFLICT",
                reason=reason,
            )
        )
        self._record_audit(
            timestamp=float(timestamp),
            camera_id=camera_id,
            local_track_id=local_track_id,
            global_id=global_id,
            decision="CONFLICT_BLOCKED",
            reason=reason,
        )
        return True

    def list_unresolved_observations(
        self, status: Optional[str] = "PENDING"
    ) -> list[UnresolvedObservationRecord]:
        """Return unresolved observations awaiting evidence or manual review."""
        return self.memory.storage.get_unresolved(status=status)

    def review_unresolved(
        self,
        tracklet_uid: str,
        decision: str,
        target_global_id: Optional[str] = None,
        reason: str = "manual_review",
        timestamp: Optional[float] = None,
    ) -> UnresolvedObservationRecord:
        """Manually resolve or dismiss an UNCERTAIN observation."""
        ts = float(time.time() if timestamp is None else timestamp)
        pending = {r.tracklet_uid: r for r in self.memory.storage.get_unresolved(status=None)}
        rec = pending.get(tracklet_uid)
        if rec is None:
            raise KeyError(f"Unresolved observation '{tracklet_uid}' not found.")

        decision_norm = decision.strip().upper()
        if decision_norm not in ("ACCEPT", "REJECT"):
            raise ValueError("Manual review decision must be 'ACCEPT' or 'REJECT'.")

        if decision_norm == "ACCEPT":
            if not target_global_id or target_global_id not in self._identities:
                raise ValueError(f"Unknown target_global_id: {target_global_id!r}")
            if self.check_simultaneous_conflict(
                target_global_id, rec.camera_id, rec.local_track_id, ts
            ):
                raise ValueError(
                    f"Cannot assign '{target_global_id}': simultaneous active conflict."
                )
            gid_rec = self._identities[target_global_id]
            gid_rec.state = IdentityState.RECOVERED
            gid_rec.last_seen_at = ts
            gid_rec.current_camera_id = rec.camera_id
            gid_rec.camera_track_history.setdefault(rec.camera_id, []).append(
                rec.local_track_id
            )
            self._tracklet_to_gid[tracklet_uid] = target_global_id
            self._persist_lifecycle(gid_rec)
            updated = UnresolvedObservationRecord(
                tracklet_uid=rec.tracklet_uid,
                camera_id=rec.camera_id,
                local_track_id=rec.local_track_id,
                first_seen_at=rec.first_seen_at,
                last_seen_at=ts,
                candidate_ids=rec.candidate_ids,
                confidence=rec.confidence,
                reason=reason,
                status="RESOLVED",
                resolved_global_id=target_global_id,
            )
        else:
            updated = UnresolvedObservationRecord(
                tracklet_uid=rec.tracklet_uid,
                camera_id=rec.camera_id,
                local_track_id=rec.local_track_id,
                first_seen_at=rec.first_seen_at,
                last_seen_at=ts,
                candidate_ids=rec.candidate_ids,
                confidence=rec.confidence,
                reason=reason,
                status="DISMISSED",
                resolved_global_id=None,
            )

        self.memory.storage.upsert_unresolved(updated)
        self._record_audit(
            timestamp=ts,
            camera_id=rec.camera_id,
            local_track_id=rec.local_track_id,
            global_id=updated.resolved_global_id,
            decision=f"MANUAL_REVIEW_{decision_norm}",
            combined_confidence=rec.confidence,
            reason=reason,
        )
        return updated

    def merge_identities(
        self,
        source_global_id: str,
        target_global_id: str,
        reason: str = "",
        operator_approved: bool = False,
        timestamp: Optional[float] = None,
    ) -> None:
        """Explicitly controlled identity merge; automatic merges are forbidden."""
        if not operator_approved:
            raise ValueError(
                "Automatic identity merging is forbidden; operator_approved=True "
                "and an explicit reason are required."
            )
        if not reason.strip():
            raise ValueError("A non-empty reason is required for controlled identity merge.")
        if source_global_id not in self._identities or target_global_id not in self._identities:
            raise KeyError("Both source_global_id and target_global_id must exist.")
        if source_global_id == target_global_id:
            raise ValueError("Cannot merge an identity with itself.")

        ts = float(time.time() if timestamp is None else timestamp)
        self._record_audit(
            timestamp=ts,
            camera_id=self._identities[target_global_id].current_camera_id or "unknown",
            local_track_id=-1,
            global_id=target_global_id,
            decision="MANUAL_MERGE",
            reason=f"merged_from_{source_global_id}:{reason}",
        )

    def get_audit_logs(
        self,
        global_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> list[AuditLogEntry]:
        """Return persisted audit log entries."""
        return self.memory.storage.get_audit_logs(
            global_id=global_id, camera_id=camera_id, decision=decision
        )

    def get_conflict_events(self) -> list[ConflictEvent]:
        """Return recorded conflict events."""
        return list(self._conflict_events)

    def save_state(self, index_path: Optional[Union[str, Path]] = None) -> Path:
        """Persist FAISS index and all lifecycle states for restart recovery."""
        for rec in self._identities.values():
            self._persist_lifecycle(rec)
        return self.memory.save_index(index_path=index_path)

    def close(self) -> None:
        """Close shared IdentityMemory if owned by this manager."""
        if self._owns_memory:
            self.memory.close()
