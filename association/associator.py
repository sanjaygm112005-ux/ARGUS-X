"""Confidence-aware Hungarian data association for ARGUS-X.

Evaluates whether newly observed BoT-SORT tracklets correspond to existing
global identities stored in ``IdentityMemory`` using ``scipy.optimize.linear_sum_assignment``.

Architectural invariants:
- Never merges global identities automatically.
- Keeps local BoT-SORT track IDs separate from global identity IDs.
- Keeps association decisions separate from ``IdentityMemory`` storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from association.cost_matrix import (
    AssociationConfig,
    CameraTransitionRule,
    ConfidenceDecision,
    CostMatrixBuilder,
    IdentityCandidate,
    PairEvaluation,
    TrackletObservation,
    validate_cost_matrix,
)
from identity_memory.memory import IdentityMemory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssociationMatch:
    """Structured result for a single Hungarian-assigned (tracklet, identity) pair."""

    local_track_id: int
    identity_id: str
    cost: float
    confidence: float
    decision: ConfidenceDecision
    evaluation: PairEvaluation


@dataclass(frozen=True)
class AssociationResult:
    """Complete output of a data association step."""

    matches: list[AssociationMatch]
    unmatched_tracklets: list[TrackletObservation]
    unmatched_candidates: list[IdentityCandidate]
    cost_matrix: np.ndarray


class DataAssociator:
    """Confidence-aware one-to-one tracklet-to-identity associator."""

    def __init__(
        self,
        memory: Optional[IdentityMemory] = None,
        config: Optional[AssociationConfig] = None,
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
    ) -> None:
        """Initialize the associator.

        Args:
            memory: Optional ``IdentityMemory`` instance for candidate retrieval.
            config: Configurable weights, gates, and confidence thresholds.
            transition_rules: Optional directed camera transition constraints.
        """
        self.memory = memory
        self.config = config or AssociationConfig()
        self.cost_builder = CostMatrixBuilder(
            config=self.config,
            transition_rules=transition_rules,
        )

    def retrieve_candidates(
        self,
        tracklets: Sequence[TrackletObservation],
        top_k_per_tracklet: int = 10,
    ) -> list[IdentityCandidate]:
        """Retrieve and group historical identity candidates from IdentityMemory."""
        if self.memory is None or not tracklets:
            return []

        if top_k_per_tracklet < 1:
            raise ValueError(f"top_k_per_tracklet must be >= 1, got {top_k_per_tracklet}.")

        best_sim_by_id: dict[str, float] = {}
        for tracklet in tracklets:
            hits = self.memory.search(tracklet.embedding, top_k=top_k_per_tracklet)
            for hit in hits:
                prev = best_sim_by_id.get(hit.identity_id, -1.0)
                if hit.similarity > prev:
                    best_sim_by_id[hit.identity_id] = hit.similarity

        candidates: list[IdentityCandidate] = []
        for identity_id, best_sim in best_sim_by_id.items():
            obs = self.memory.get_identity_observations(identity_id)
            if not obs:
                continue
            candidates.append(
                IdentityCandidate(
                    identity_id=identity_id,
                    observations=obs,
                    best_similarity=best_sim,
                )
            )
        return candidates

    def solve_assignment(
        self,
        cost_matrix: np.ndarray,
        gating_threshold: Optional[float] = None,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Run Hungarian one-to-one assignment and filter gated pairs.

        Args:
            cost_matrix: Validated 2D non-negative cost matrix ``(N, M)``.
            gating_threshold: Costs ``>= gating_threshold`` are rejected as
                unmatched (defaults to ``self.config.gating_cost``).

        Returns:
            ``(matched_pairs, unmatched_rows, unmatched_cols)``
        """
        matrix = validate_cost_matrix(cost_matrix)
        n_rows, n_cols = matrix.shape
        threshold = float(
            self.config.gating_cost if gating_threshold is None else gating_threshold
        )

        if n_rows == 0 or n_cols == 0:
            return [], list(range(n_rows)), list(range(n_cols))

        row_ind, col_ind = linear_sum_assignment(matrix)

        matched_pairs: list[tuple[int, int]] = []
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if matrix[r, c] >= threshold:
                continue
            matched_pairs.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

        unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
        unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
        return matched_pairs, unmatched_rows, unmatched_cols

    def associate(
        self,
        tracklets: Sequence[TrackletObservation],
        candidates: Optional[Sequence[IdentityCandidate]] = None,
        top_k_per_tracklet: int = 10,
    ) -> AssociationResult:
        """Associate newly observed tracklets with historical identity candidates.

        If ``candidates`` is ``None``, candidates are queried from ``self.memory``.
        Does not modify ``self.memory`` or merge any global identities.
        """
        tracklet_list = list(tracklets)
        if not tracklet_list:
            return AssociationResult(
                matches=[],
                unmatched_tracklets=[],
                unmatched_candidates=list(candidates or []),
                cost_matrix=np.empty((0, len(candidates or [])), dtype=np.float64),
            )

        candidate_list = (
            list(candidates)
            if candidates is not None
            else self.retrieve_candidates(tracklet_list, top_k_per_tracklet=top_k_per_tracklet)
        )

        if not candidate_list:
            return AssociationResult(
                matches=[],
                unmatched_tracklets=tracklet_list,
                unmatched_candidates=[],
                cost_matrix=np.empty((len(tracklet_list), 0), dtype=np.float64),
            )

        cost_matrix, evaluations = self.cost_builder.build(tracklet_list, candidate_list)
        matched_pairs, unmatched_rows, unmatched_cols = self.solve_assignment(cost_matrix)

        matches: list[AssociationMatch] = []
        for r, c in matched_pairs:
            ev = evaluations[r][c]
            if ev.decision == ConfidenceDecision.REJECT:
                unmatched_rows.append(r)
                unmatched_cols.append(c)
                continue
            matches.append(
                AssociationMatch(
                    local_track_id=ev.local_track_id,
                    identity_id=ev.identity_id,
                    cost=ev.total_cost,
                    confidence=ev.confidence,
                    decision=ev.decision,
                    evaluation=ev,
                )
            )

        unmatched_tracklets = [tracklet_list[r] for r in sorted(set(unmatched_rows))]
        unmatched_candidates = [candidate_list[c] for c in sorted(set(unmatched_cols))]

        logger.info(
            "Association complete: %d tracklets, %d candidates -> %d matches, %d unmatched tracklets",
            len(tracklet_list),
            len(candidate_list),
            len(matches),
            len(unmatched_tracklets),
        )
        return AssociationResult(
            matches=matches,
            unmatched_tracklets=unmatched_tracklets,
            unmatched_candidates=unmatched_candidates,
            cost_matrix=cost_matrix,
        )
