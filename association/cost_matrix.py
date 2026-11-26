"""Cost matrix construction and compatibility evaluators for ARGUS-X.

Combines appearance similarity (OSNet/FAISS), temporal compatibility,
camera transition rules, and optional calibrated Kalman motion estimates into
normalized, gated cost matrices for Hungarian assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from identity_memory.storage import ObservationRecord

logger = logging.getLogger(__name__)

# Large finite cost used to gate impossible candidate pairs in linear_sum_assignment.
GATING_COST: float = 1e6


class ConfidenceDecision(str, Enum):
    """Confidence classification for a candidate association."""

    ACCEPT = "ACCEPT"
    UNCERTAIN = "UNCERTAIN"
    REJECT = "REJECT"


@dataclass(frozen=True)
class CameraTransitionRule:
    """Directed transition constraint between two cameras."""

    source_camera: str
    target_camera: str
    min_travel_sec: float = 0.0
    max_travel_sec: Optional[float] = None
    allowed: bool = True

    def __post_init__(self) -> None:
        if not self.source_camera or not self.target_camera:
            raise ValueError("source_camera and target_camera must be non-empty.")
        if self.min_travel_sec < 0.0:
            raise ValueError(
                f"min_travel_sec must be >= 0, got {self.min_travel_sec}."
            )
        if self.max_travel_sec is not None and self.max_travel_sec < self.min_travel_sec:
            raise ValueError(
                f"max_travel_sec ({self.max_travel_sec}) must be >= "
                f"min_travel_sec ({self.min_travel_sec})."
            )


@dataclass(frozen=True)
class KalmanMotionEstimate:
    """Interface for Kalman filter position/velocity estimates.

    Raw pixel coordinates from unrelated cameras must never be compared directly;
    cross-camera motion comparison is only enabled when ``is_calibrated=True``
    and both states share the same ``coordinate_frame``.
    """

    position: Tuple[float, float]
    velocity: Optional[Tuple[float, float]] = None
    covariance_diag: Optional[Tuple[float, float]] = None
    coordinate_frame: str = "pixel"
    is_calibrated: bool = False


@dataclass(frozen=True)
class TrackletObservation:
    """Newly observed local BoT-SORT tracklet to evaluate against memory."""

    local_track_id: int
    camera_id: str
    timestamp: float
    embedding: np.ndarray
    motion_state: Optional[KalmanMotionEstimate] = None


@dataclass(frozen=True)
class IdentityCandidate:
    """Historical global identity candidate with one or more stored observations."""

    identity_id: str
    observations: Sequence[ObservationRecord]
    best_similarity: float
    motion_state: Optional[KalmanMotionEstimate] = None

    @property
    def latest_observation(self) -> ObservationRecord:
        """Return the most recent historical observation by timestamp."""
        if not self.observations:
            raise ValueError(f"Candidate '{self.identity_id}' has no observations.")
        return max(self.observations, key=lambda obs: obs.timestamp)


@dataclass(frozen=True)
class AssociationConfig:
    """Configurable weights, gates, and confidence thresholds."""

    appearance_weight: float = 0.50
    temporal_weight: float = 0.20
    camera_weight: float = 0.15
    motion_weight: float = 0.15

    max_time_gap_sec: float = 3600.0
    min_appearance_similarity: float = 0.35
    max_motion_distance: float = 100.0

    accept_threshold: float = 0.75
    uncertain_threshold: float = 0.45
    gating_cost: float = GATING_COST

    def __post_init__(self) -> None:
        weights = (
            self.appearance_weight,
            self.temporal_weight,
            self.camera_weight,
            self.motion_weight,
        )
        if any(w < 0.0 for w in weights):
            raise ValueError("All component weights must be non-negative.")
        if sum(weights) <= 0.0:
            raise ValueError("At least one component weight must be positive.")
        if self.max_time_gap_sec <= 0.0:
            raise ValueError("max_time_gap_sec must be positive.")
        if not (0.0 <= self.min_appearance_similarity <= 1.0):
            raise ValueError("min_appearance_similarity must be in [0.0, 1.0].")
        if self.max_motion_distance <= 0.0:
            raise ValueError("max_motion_distance must be positive.")
        if not (0.0 <= self.uncertain_threshold <= self.accept_threshold <= 1.0):
            raise ValueError(
                "Thresholds must satisfy 0 <= uncertain_threshold <= accept_threshold <= 1."
            )
        if self.gating_cost <= 1.0:
            raise ValueError("gating_cost must be > 1.0.")


@dataclass(frozen=True)
class PairEvaluation:
    """Detailed compatibility and confidence breakdown for a (tracklet, candidate) pair."""

    local_track_id: int
    identity_id: str
    total_cost: float
    confidence: float
    gated: bool
    gate_reason: Optional[str]
    appearance_score: float
    temporal_score: Optional[float]
    camera_score: Optional[float]
    motion_score: Optional[float]
    is_camera_uncertain: bool
    decision: ConfidenceDecision


def validate_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    """Validate that a cost matrix is 2D, finite, and non-negative."""
    if cost_matrix is None:
        raise ValueError("cost_matrix cannot be None.")

    arr = np.asarray(cost_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {arr.shape}.")
    if arr.size == 0:
        return arr
    if not np.all(np.isfinite(arr)):
        raise ValueError("cost_matrix contains NaN or infinite values.")
    if np.any(arr < 0.0):
        raise ValueError("cost_matrix cannot contain negative costs.")
    return arr


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Embedding must be non-empty and finite.")
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("Embedding cannot have zero norm.")
    return arr / norm


def evaluate_appearance_similarity(
    query_embedding: np.ndarray,
    candidate: IdentityCandidate,
    candidate_embeddings: Optional[Sequence[np.ndarray]] = None,
) -> float:
    """Compute normalized appearance similarity in [0.0, 1.0].

    Supports multiple historical embeddings when ``candidate_embeddings`` is
    provided, otherwise uses ``candidate.best_similarity`` retrieved from
    ``IdentityMemory``.
    """
    _normalize_vec(query_embedding)
    if candidate_embeddings:
        q = _normalize_vec(query_embedding)
        sims = [float(np.dot(q, _normalize_vec(emb))) for emb in candidate_embeddings]
        raw_sim = max(sims)
    else:
        raw_sim = float(candidate.best_similarity)

    return float(np.clip(raw_sim, 0.0, 1.0))


def evaluate_temporal_compatibility(
    query_timestamp: float,
    candidate_timestamp: float,
    max_time_gap_sec: float,
) -> tuple[Optional[float], bool, Optional[str]]:
    """Evaluate temporal compatibility between a historical observation and tracklet.

    Returns:
        ``(score_or_None, is_compatible, gate_reason)``
    """
    delta_t = float(query_timestamp) - float(candidate_timestamp)
    if delta_t < 0.0:
        return None, False, "negative_time_gap"
    if delta_t > max_time_gap_sec:
        return None, False, "exceeds_max_time_gap"

    score = 1.0 - (delta_t / max_time_gap_sec)
    return float(np.clip(score, 0.0, 1.0)), True, None


def evaluate_camera_transition(
    source_camera: str,
    target_camera: str,
    delta_t: float,
    transition_rules: Mapping[tuple[str, str], CameraTransitionRule],
) -> tuple[Optional[float], bool, bool, Optional[str]]:
    """Evaluate camera transition feasibility.

    Returns:
        ``(score_or_None, is_compatible, is_uncertain, gate_reason)``
    """
    if source_camera == target_camera:
        return 1.0, True, False, None

    rule = transition_rules.get((source_camera, target_camera))
    if rule is None:
        # Unknown transition: do not invent a travel-time score; mark uncertain.
        return None, True, True, "unknown_camera_transition"

    if not rule.allowed:
        return None, False, False, "forbidden_camera_transition"

    if delta_t < rule.min_travel_sec:
        return None, False, False, "below_min_travel_time"

    if rule.max_travel_sec is not None and delta_t > rule.max_travel_sec:
        return None, False, False, "exceeds_max_travel_time"

    if rule.max_travel_sec is None or rule.max_travel_sec == rule.min_travel_sec:
        return 1.0, True, False, None

    span = rule.max_travel_sec - rule.min_travel_sec
    normalized_offset = (delta_t - rule.min_travel_sec) / span
    score = 1.0 - 0.5 * normalized_offset
    return float(np.clip(score, 0.0, 1.0)), True, False, None


def evaluate_motion_compatibility(
    tracklet_motion: Optional[KalmanMotionEstimate],
    candidate_motion: Optional[KalmanMotionEstimate],
    delta_t: float,
    source_camera: str,
    target_camera: str,
    max_motion_distance: float,
) -> tuple[Optional[float], bool, Optional[str]]:
    """Evaluate spatial/Kalman motion compatibility when geometry is reliable.

    Never compares raw pixel coordinates across different cameras.
    Returns ``(score_or_None, is_compatible, gate_reason)``.
    """
    if tracklet_motion is None or candidate_motion is None:
        return None, True, None

    same_camera = source_camera == target_camera
    calibrated_shared_frame = (
        tracklet_motion.is_calibrated
        and candidate_motion.is_calibrated
        and tracklet_motion.coordinate_frame == candidate_motion.coordinate_frame
    )

    if not same_camera and not calibrated_shared_frame:
        logger.debug(
            "Skipping cross-camera motion comparison (%s -> %s): uncalibrated coordinates.",
            source_camera,
            target_camera,
        )
        return None, True, None

    if same_camera and tracklet_motion.coordinate_frame != candidate_motion.coordinate_frame:
        return None, True, None

    pred_x = float(candidate_motion.position[0])
    pred_y = float(candidate_motion.position[1])
    if candidate_motion.velocity is not None and delta_t >= 0.0:
        pred_x += float(candidate_motion.velocity[0]) * delta_t
        pred_y += float(candidate_motion.velocity[1]) * delta_t

    dx = float(tracklet_motion.position[0]) - pred_x
    dy = float(tracklet_motion.position[1]) - pred_y

    if (
        tracklet_motion.covariance_diag is not None
        and tracklet_motion.covariance_diag[0] > 0.0
        and tracklet_motion.covariance_diag[1] > 0.0
    ):
        var_x = float(tracklet_motion.covariance_diag[0])
        var_y = float(tracklet_motion.covariance_diag[1])
        distance = float(np.sqrt((dx * dx) / var_x + (dy * dy) / var_y))
    else:
        distance = float(np.hypot(dx, dy))

    if distance > max_motion_distance:
        return None, False, "exceeds_max_motion_distance"

    score = 1.0 - (distance / max_motion_distance)
    return float(np.clip(score, 0.0, 1.0)), True, None


class CostMatrixBuilder:
    """Builds normalized, gated cost matrices for Hungarian data association."""

    def __init__(
        self,
        config: Optional[AssociationConfig] = None,
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
    ) -> None:
        self.config = config or AssociationConfig()
        self.transition_rules: dict[tuple[str, str], CameraTransitionRule] = {
            (r.source_camera, r.target_camera): r for r in (transition_rules or [])
        }

    def evaluate_pair(
        self,
        tracklet: TrackletObservation,
        candidate: IdentityCandidate,
        candidate_embeddings: Optional[Sequence[np.ndarray]] = None,
    ) -> PairEvaluation:
        """Evaluate all compatibility signals for one (tracklet, candidate) pair."""
        latest = candidate.latest_observation
        delta_t = float(tracklet.timestamp) - float(latest.timestamp)

        app_score = evaluate_appearance_similarity(
            tracklet.embedding, candidate, candidate_embeddings
        )

        if app_score < self.config.min_appearance_similarity:
            return PairEvaluation(
                local_track_id=tracklet.local_track_id,
                identity_id=candidate.identity_id,
                total_cost=self.config.gating_cost,
                confidence=0.0,
                gated=True,
                gate_reason="low_appearance_similarity",
                appearance_score=app_score,
                temporal_score=None,
                camera_score=None,
                motion_score=None,
                is_camera_uncertain=False,
                decision=ConfidenceDecision.REJECT,
            )

        temp_score, temp_ok, temp_reason = evaluate_temporal_compatibility(
            tracklet.timestamp, latest.timestamp, self.config.max_time_gap_sec
        )
        if not temp_ok:
            return PairEvaluation(
                local_track_id=tracklet.local_track_id,
                identity_id=candidate.identity_id,
                total_cost=self.config.gating_cost,
                confidence=0.0,
                gated=True,
                gate_reason=temp_reason,
                appearance_score=app_score,
                temporal_score=temp_score,
                camera_score=None,
                motion_score=None,
                is_camera_uncertain=False,
                decision=ConfidenceDecision.REJECT,
            )

        cam_score, cam_ok, cam_uncertain, cam_reason = evaluate_camera_transition(
            latest.camera_id, tracklet.camera_id, delta_t, self.transition_rules
        )
        if not cam_ok:
            return PairEvaluation(
                local_track_id=tracklet.local_track_id,
                identity_id=candidate.identity_id,
                total_cost=self.config.gating_cost,
                confidence=0.0,
                gated=True,
                gate_reason=cam_reason,
                appearance_score=app_score,
                temporal_score=temp_score,
                camera_score=cam_score,
                motion_score=None,
                is_camera_uncertain=False,
                decision=ConfidenceDecision.REJECT,
            )

        motion_score, motion_ok, motion_reason = evaluate_motion_compatibility(
            tracklet.motion_state,
            candidate.motion_state,
            delta_t,
            latest.camera_id,
            tracklet.camera_id,
            self.config.max_motion_distance,
        )
        if not motion_ok:
            return PairEvaluation(
                local_track_id=tracklet.local_track_id,
                identity_id=candidate.identity_id,
                total_cost=self.config.gating_cost,
                confidence=0.0,
                gated=True,
                gate_reason=motion_reason,
                appearance_score=app_score,
                temporal_score=temp_score,
                camera_score=cam_score,
                motion_score=motion_score,
                is_camera_uncertain=cam_uncertain,
                decision=ConfidenceDecision.REJECT,
            )

        weighted_pairs: list[tuple[float, float]] = [
            (self.config.appearance_weight, app_score),
        ]
        if temp_score is not None:
            weighted_pairs.append((self.config.temporal_weight, temp_score))
        if cam_score is not None:
            weighted_pairs.append((self.config.camera_weight, cam_score))
        if motion_score is not None:
            weighted_pairs.append((self.config.motion_weight, motion_score))

        active_weight = sum(w for w, _ in weighted_pairs)
        confidence = (
            sum(w * s for w, s in weighted_pairs) / active_weight
            if active_weight > 0.0
            else 0.0
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        total_cost = float(np.clip(1.0 - confidence, 0.0, 1.0))

        # Appearance alone or unknown camera transitions must not be auto-accepted.
        non_appearance_signals = sum(
            1 for s in (temp_score, cam_score, motion_score) if s is not None
        )
        if confidence >= self.config.accept_threshold and not cam_uncertain and non_appearance_signals >= 1:
            decision = ConfidenceDecision.ACCEPT
        elif confidence >= self.config.uncertain_threshold:
            decision = ConfidenceDecision.UNCERTAIN
        else:
            decision = ConfidenceDecision.REJECT

        return PairEvaluation(
            local_track_id=tracklet.local_track_id,
            identity_id=candidate.identity_id,
            total_cost=total_cost,
            confidence=confidence,
            gated=False,
            gate_reason=cam_reason if cam_uncertain else None,
            appearance_score=app_score,
            temporal_score=temp_score,
            camera_score=cam_score,
            motion_score=motion_score,
            is_camera_uncertain=cam_uncertain,
            decision=decision,
        )

    def build(
        self,
        tracklets: Sequence[TrackletObservation],
        candidates: Sequence[IdentityCandidate],
    ) -> tuple[np.ndarray, list[list[PairEvaluation]]]:
        """Construct the (N_tracklets, M_candidates) cost matrix and evaluations."""
        n_rows = len(tracklets)
        n_cols = len(candidates)
        cost_matrix = np.zeros((n_rows, n_cols), dtype=np.float64)
        evaluations: list[list[PairEvaluation]] = []

        for i, tracklet in enumerate(tracklets):
            row_evals: list[PairEvaluation] = []
            for j, candidate in enumerate(candidates):
                ev = self.evaluate_pair(tracklet, candidate)
                cost_matrix[i, j] = ev.total_cost
                row_evals.append(ev)
            evaluations.append(row_evals)

        validate_cost_matrix(cost_matrix)
        return cost_matrix, evaluations
