"""Confidence-aware Hungarian data association module."""

from association.associator import AssociationMatch, AssociationResult, DataAssociator
from association.cost_matrix import (
    GATING_COST,
    AssociationConfig,
    CameraTransitionRule,
    ConfidenceDecision,
    CostMatrixBuilder,
    IdentityCandidate,
    KalmanMotionEstimate,
    PairEvaluation,
    TrackletObservation,
    evaluate_appearance_similarity,
    evaluate_camera_transition,
    evaluate_motion_compatibility,
    evaluate_temporal_compatibility,
    validate_cost_matrix,
)

__all__ = [
    "AssociationConfig",
    "AssociationMatch",
    "AssociationResult",
    "CameraTransitionRule",
    "ConfidenceDecision",
    "CostMatrixBuilder",
    "DataAssociator",
    "GATING_COST",
    "IdentityCandidate",
    "KalmanMotionEstimate",
    "PairEvaluation",
    "TrackletObservation",
    "evaluate_appearance_similarity",
    "evaluate_camera_transition",
    "evaluate_motion_compatibility",
    "evaluate_temporal_compatibility",
    "validate_cost_matrix",
]
