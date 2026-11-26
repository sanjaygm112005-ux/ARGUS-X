"""Ground-truth evaluation metrics for multi-camera identity recovery.

Explicitly separates:
1. Frame-level metrics (accuracy, precision, recall, F1, ID switches across every
   annotated frame observation).
2. Recovery-event metrics (evaluating discrete reappearance events after
   occlusion or camera transitions, independent of track duration).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union


@dataclass(frozen=True)
class GroundTruthAnnotation:
    """Single ground-truth person observation at a camera and timestamp."""

    camera_id: str
    timestamp: float
    gt_identity_id: str
    bbox: Optional[tuple[float, float, float, float]] = None
    scenario_tag: Optional[str] = None


@dataclass(frozen=True)
class PredictedObservation:
    """Predicted recovery output aligned with a frame timestamp."""

    camera_id: str
    timestamp: float
    predicted_global_id: Optional[str]
    local_track_id: int
    bbox: tuple[float, float, float, float]
    recovery_decision: str


@dataclass(frozen=True)
class RecoveryEventMetrics:
    """Event-level metrics for discrete reappearance / cross-camera transitions."""

    total_recovery_events: int
    correct_recoveries: int
    false_recoveries: int
    missed_recoveries: int
    recovery_event_accuracy: float
    event_precision: float
    event_recall: float
    event_f1_score: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Frame-level and event-level identity-recovery metrics."""

    total_ground_truth: int
    matched_predictions: int
    true_positives: int
    false_identity_matches: int
    missed_identity_recoveries: int
    id_switches: int
    identity_recovery_accuracy: float
    precision: float
    recall: float
    f1_score: float
    event_metrics: Optional[RecoveryEventMetrics] = None
    scenario_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


def _bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_ground_truth(path: Union[str, Path]) -> list[GroundTruthAnnotation]:
    """Load and validate ground-truth annotations from a JSON file."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {target}")

    raw = json.loads(target.read_text(encoding="utf-8"))
    items = raw.get("annotations", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError("Ground-truth JSON must contain a list or {'annotations': [...]}.")

    annotations: list[GroundTruthAnnotation] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each annotation entry must be a JSON object.")
        bbox_raw = item.get("bbox")
        bbox = (
            tuple(float(v) for v in bbox_raw)  # type: ignore[assignment]
            if bbox_raw is not None
            else None
        )
        if bbox is not None and len(bbox) != 4:
            raise ValueError(f"Annotation bbox must have 4 values, got {bbox}.")
        annotations.append(
            GroundTruthAnnotation(
                camera_id=str(item["camera_id"]),
                timestamp=float(item["timestamp"]),
                gt_identity_id=str(item["gt_identity_id"]),
                bbox=bbox,  # type: ignore[arg-type]
                scenario_tag=(
                    str(item["scenario_tag"])
                    if item.get("scenario_tag") is not None
                    else None
                ),
            )
        )
    return annotations


def save_ground_truth_template(path: Union[str, Path]) -> Path:
    """Write an empty ground-truth annotation file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    template = {
        "description": "ARGUS-X ground-truth annotations",
        "annotations": [],
    }
    target.write_text(json.dumps(template, indent=2), encoding="utf-8")
    return target


def _compute_recovery_event_metrics(
    gt_to_pred: Sequence[tuple[GroundTruthAnnotation, Optional[PredictedObservation]]],
    canonical_pred_for_gt: dict[str, str],
    reappearance_gap_sec: float,
) -> RecoveryEventMetrics:
    """Evaluate discrete reappearance events (camera change or gap > threshold)."""
    by_gt: dict[str, list[tuple[GroundTruthAnnotation, Optional[PredictedObservation]]]] = defaultdict(list)
    for gt, pred in gt_to_pred:
        by_gt[gt.gt_identity_id].append((gt, pred))

    total_events = 0
    correct = 0
    false_rec = 0
    missed = 0

    for gt_id, timeline in by_gt.items():
        if len(timeline) < 2:
            continue
        expected_gid = canonical_pred_for_gt.get(gt_id)
        prev_gt = timeline[0][0]
        for curr_gt, curr_pred in timeline[1:]:
            is_cam_switch = curr_gt.camera_id != prev_gt.camera_id
            is_time_gap = (curr_gt.timestamp - prev_gt.timestamp) >= reappearance_gap_sec
            if is_cam_switch or is_time_gap:
                total_events += 1
                if curr_pred is None or curr_pred.predicted_global_id is None:
                    missed += 1
                elif expected_gid is not None and curr_pred.predicted_global_id == expected_gid:
                    correct += 1
                else:
                    false_rec += 1
            prev_gt = curr_gt

    e_prec = correct / (correct + false_rec) if (correct + false_rec) > 0 else 0.0
    e_rec = correct / total_events if total_events > 0 else 0.0
    e_f1 = (
        2.0 * e_prec * e_rec / (e_prec + e_rec) if (e_prec + e_rec) > 0.0 else 0.0
    )
    e_acc = correct / total_events if total_events > 0 else 0.0
    return RecoveryEventMetrics(
        total_recovery_events=total_events,
        correct_recoveries=correct,
        false_recoveries=false_rec,
        missed_recoveries=missed,
        recovery_event_accuracy=e_acc,
        event_precision=e_prec,
        event_recall=e_rec,
        event_f1_score=e_f1,
    )


def evaluate_predictions(
    ground_truth: Sequence[GroundTruthAnnotation],
    predictions: Sequence[PredictedObservation],
    timestamp_tolerance_sec: float = 0.25,
    iou_threshold: float = 0.3,
    reappearance_gap_sec: float = 2.0,
) -> EvaluationMetrics:
    """Compare predicted global identities against ground-truth annotations.

    Frame-level recall is ``TP / total_ground_truth`` so both missed detections
    and wrong-ID matches reduce recall without inflating metrics.
    """
    if timestamp_tolerance_sec < 0.0:
        raise ValueError("timestamp_tolerance_sec must be >= 0.")
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0.0, 1.0].")
    if reappearance_gap_sec <= 0.0:
        raise ValueError("reappearance_gap_sec must be positive.")

    if not ground_truth:
        return EvaluationMetrics(
            total_ground_truth=0,
            matched_predictions=0,
            true_positives=0,
            false_identity_matches=0,
            missed_identity_recoveries=0,
            id_switches=0,
            identity_recovery_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            event_metrics=RecoveryEventMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0),
            scenario_breakdown={},
        )

    sorted_gt = sorted(ground_truth, key=lambda g: g.timestamp)
    used_pred_indices: set[int] = set()
    gt_to_pred: list[tuple[GroundTruthAnnotation, Optional[PredictedObservation]]] = []

    for gt in sorted_gt:
        best_idx: Optional[int] = None
        best_score: float = -1.0
        for idx, pred in enumerate(predictions):
            if idx in used_pred_indices:
                continue
            if pred.camera_id != gt.camera_id:
                continue
            dt = abs(pred.timestamp - gt.timestamp)
            if dt > timestamp_tolerance_sec:
                continue
            if gt.bbox is not None:
                iou = _bbox_iou(gt.bbox, pred.bbox)
                if iou < iou_threshold:
                    continue
                score = iou - 0.01 * dt
            else:
                score = 1.0 / (1.0 + dt)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            used_pred_indices.add(best_idx)
            gt_to_pred.append((gt, predictions[best_idx]))
        else:
            gt_to_pred.append((gt, None))

    votes_by_gt: dict[str, Counter[str]] = defaultdict(Counter)
    for gt, pred in gt_to_pred:
        if pred is not None and pred.predicted_global_id is not None:
            votes_by_gt[gt.gt_identity_id][pred.predicted_global_id] += 1

    canonical_pred_for_gt: dict[str, str] = {}
    claimed_pred_ids: set[str] = set()
    vote_pairs = [
        (cnt, gt_id, pred_id)
        for gt_id, counter in votes_by_gt.items()
        for pred_id, cnt in counter.items()
    ]
    for _, gt_id, pred_id in sorted(vote_pairs, reverse=True):
        if gt_id not in canonical_pred_for_gt and pred_id not in claimed_pred_ids:
            canonical_pred_for_gt[gt_id] = pred_id
            claimed_pred_ids.add(pred_id)

    tp = 0
    fp = 0
    fn = 0
    id_switches = 0
    matched_count = 0
    last_pred_by_gt: dict[str, str] = {}
    scenario_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tp": 0, "fp": 0, "fn": 0, "id_switches": 0}
    )

    for gt, pred in gt_to_pred:
        tag = gt.scenario_tag
        if tag:
            scenario_counts[tag]["total"] += 1

        if pred is None or pred.predicted_global_id is None:
            fn += 1
            if tag:
                scenario_counts[tag]["fn"] += 1
            continue

        matched_count += 1
        pred_gid = pred.predicted_global_id
        prev_gid = last_pred_by_gt.get(gt.gt_identity_id)
        if prev_gid is not None and prev_gid != pred_gid:
            id_switches += 1
            if tag:
                scenario_counts[tag]["id_switches"] += 1
        last_pred_by_gt[gt.gt_identity_id] = pred_gid

        expected_gid = canonical_pred_for_gt.get(gt.gt_identity_id)
        if expected_gid is not None and pred_gid == expected_gid:
            tp += 1
            if tag:
                scenario_counts[tag]["tp"] += 1
        else:
            fp += 1
            if tag:
                scenario_counts[tag]["fp"] += 1

    total = len(sorted_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total if total > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = tp / total if total > 0 else 0.0

    event_metrics = _compute_recovery_event_metrics(
        gt_to_pred, canonical_pred_for_gt, reappearance_gap_sec
    )

    breakdown: dict[str, dict[str, float]] = {}
    for tag, c in scenario_counts.items():
        s_tp, s_fp, s_tot = c["tp"], c["fp"], c["total"]
        s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_rec = s_tp / s_tot if s_tot > 0 else 0.0
        s_f1 = (
            2.0 * s_prec * s_rec / (s_prec + s_rec) if (s_prec + s_rec) > 0.0 else 0.0
        )
        breakdown[tag] = {
            "total": float(s_tot),
            "accuracy": s_tp / s_tot if s_tot > 0 else 0.0,
            "precision": s_prec,
            "recall": s_rec,
            "f1_score": s_f1,
            "id_switches": float(c["id_switches"]),
        }

    return EvaluationMetrics(
        total_ground_truth=total,
        matched_predictions=matched_count,
        true_positives=tp,
        false_identity_matches=fp,
        missed_identity_recoveries=fn,
        id_switches=id_switches,
        identity_recovery_accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        event_metrics=event_metrics,
        scenario_breakdown=breakdown,
    )
