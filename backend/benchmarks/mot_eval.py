"""TrackEval-compatible MOTA, IDF1, and HOTA metrics for MOT17 sequences."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class MOTMetricsResult:
    """Measured MOT17 tracking evaluation metrics."""

    mota: float
    idf1: float
    hota: float
    id_switches: int
    false_positives: int
    false_negatives: int
    true_positives: int
    total_gt: int
    frames_evaluated: int


def _iou_matrix(gt_boxes: np.ndarray, pr_boxes: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between (N, 4) xywh GT boxes and (M, 4) xywh pred boxes."""
    import numpy as np

    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return np.zeros((len(gt_boxes), len(pr_boxes)), dtype=np.float64)

    gx1 = gt_boxes[:, 0:1]
    gy1 = gt_boxes[:, 1:2]
    gx2 = gx1 + np.maximum(0.0, gt_boxes[:, 2:3])
    gy2 = gy1 + np.maximum(0.0, gt_boxes[:, 3:4])

    px1 = pr_boxes[:, 0:1].T
    py1 = pr_boxes[:, 1:2].T
    px2 = px1 + np.maximum(0.0, pr_boxes[:, 2:3].T)
    py2 = py1 + np.maximum(0.0, pr_boxes[:, 3:4].T)

    ix1 = np.maximum(gx1, px1)
    iy1 = np.maximum(gy1, py1)
    ix2 = np.minimum(gx2, px2)
    iy2 = np.minimum(gy2, py2)

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    g_area = np.maximum(0.0, gx2 - gx1) * np.maximum(0.0, gy2 - gy1)
    p_area = np.maximum(0.0, px2 - px1) * np.maximum(0.0, py2 - py1)
    union = g_area + p_area - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


def evaluate_mot17_predictions(
    gt_by_frame: dict[int, list[dict[str, float]]],
    pred_by_frame: dict[int, list[dict[str, float]]],
    evaluated_frames: list[int],
    iou_threshold: float = 0.5,
) -> MOTMetricsResult:
    """Compute MOTA, IDF1, and HOTA using Hungarian assignment.

    Args:
        gt_by_frame: 1-based frame_idx -> list of {id, x, y, w, h}
        pred_by_frame: 1-based frame_idx -> list of {id, x, y, w, h}
        evaluated_frames: Sorted list of 1-based frame indices evaluated.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    id_switches = 0
    last_gt_to_pred: dict[int, int] = {}

    # For IDF1: count co-occurrences of (gt_id, pred_id) across matched frames at iou >= 0.5
    gt_counts: dict[int, int] = defaultdict(int)
    pr_counts: dict[int, int] = defaultdict(int)
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    # Precompute per-frame IoU matrices for HOTA alpha sweep
    frame_data: list[tuple[list[int], list[int], np.ndarray]] = []

    for fidx in evaluated_frames:
        gts = gt_by_frame.get(fidx, [])
        prs = pred_by_frame.get(fidx, [])
        gt_ids = [int(g["id"]) for g in gts]
        pr_ids = [int(p["id"]) for p in prs]

        for gid in gt_ids:
            gt_counts[gid] += 1
        for pid in pr_ids:
            pr_counts[pid] += 1

        total_gt += len(gts)

        if not gts and not prs:
            frame_data.append((gt_ids, pr_ids, np.zeros((0, 0), dtype=np.float64)))
            continue
        if not gts:
            total_fp += len(prs)
            frame_data.append((gt_ids, pr_ids, np.zeros((0, len(prs)), dtype=np.float64)))
            continue
        if not prs:
            total_fn += len(gts)
            frame_data.append((gt_ids, pr_ids, np.zeros((len(gts), 0), dtype=np.float64)))
            continue

        g_arr = np.array([[g["x"], g["y"], g["w"], g["h"]] for g in gts], dtype=np.float64)
        p_arr = np.array([[p["x"], p["y"], p["w"], p["h"]] for p in prs], dtype=np.float64)
        iou_mat = _iou_matrix(g_arr, p_arr)
        frame_data.append((gt_ids, pr_ids, iou_mat))

        # CLEAR MOT Hungarian matching at iou_threshold (0.5)
        cost = 1.0 - iou_mat
        # Prefer keeping previous ID assignment on near-ties
        for gi, gid in enumerate(gt_ids):
            prev_pid = last_gt_to_pred.get(gid)
            if prev_pid is not None and prev_pid in pr_ids:
                pi = pr_ids.index(prev_pid)
                if iou_mat[gi, pi] >= iou_threshold:
                    cost[gi, pi] -= 1e-4

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_g = set()
        matched_p = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= iou_threshold:
                matched_g.add(r)
                matched_p.add(c)
                gid = gt_ids[r]
                pid = pr_ids[c]
                pair_counts[(gid, pid)] += 1
                prev = last_gt_to_pred.get(gid)
                if prev is not None and prev != pid:
                    id_switches += 1
                last_gt_to_pred[gid] = pid

        tp_f = len(matched_g)
        total_tp += tp_f
        total_fn += len(gts) - tp_f
        total_fp += len(prs) - len(matched_p)

    # MOTA
    mota = 1.0 - ((total_fp + total_fn + id_switches) / total_gt) if total_gt > 0 else 0.0

    # IDF1 via global bipartite matching on ID co-occurrence matrix
    all_gt_ids = sorted(gt_counts.keys())
    all_pr_ids = sorted(pr_counts.keys())
    idtp = 0
    if all_gt_ids and all_pr_ids:
        gt_idx_map = {gid: i for i, gid in enumerate(all_gt_ids)}
        pr_idx_map = {pid: j for j, pid in enumerate(all_pr_ids)}
        match_mat = np.zeros((len(all_gt_ids), len(all_pr_ids)), dtype=np.float64)
        for (gid, pid), cnt in pair_counts.items():
            match_mat[gt_idx_map[gid], pr_idx_map[pid]] = float(cnt)
        r_ind, c_ind = linear_sum_assignment(-match_mat)
        idtp = int(sum(match_mat[r, c] for r, c in zip(r_ind, c_ind)))

    total_pr_obs = sum(pr_counts.values())
    idf1 = (2.0 * idtp) / (total_gt + total_pr_obs) if (total_gt + total_pr_obs) > 0 else 0.0

    # HOTA across 19 alpha thresholds (0.05 to 0.95)
    alphas = np.linspace(0.05, 0.95, 19)
    hota_scores: list[float] = []
    for alpha in alphas:
        tp_a = 0
        fp_a = 0
        fn_a = 0
        tpa_pairs: dict[tuple[int, int], int] = defaultdict(int)
        gt_a_counts: dict[int, int] = defaultdict(int)
        pr_a_counts: dict[int, int] = defaultdict(int)
        matched_pairs_list: list[tuple[int, int]] = []

        for gt_ids, pr_ids, iou_mat in frame_data:
            for gid in gt_ids:
                gt_a_counts[gid] += 1
            for pid in pr_ids:
                pr_a_counts[pid] += 1

            if iou_mat.size == 0:
                fn_a += len(gt_ids)
                fp_a += len(pr_ids)
                continue

            r_ind, c_ind = linear_sum_assignment(-iou_mat)
            m_count = 0
            for r, c in zip(r_ind, c_ind):
                if iou_mat[r, c] >= alpha:
                    m_count += 1
                    gid = gt_ids[r]
                    pid = pr_ids[c]
                    tpa_pairs[(gid, pid)] += 1
                    matched_pairs_list.append((gid, pid))
            tp_a += m_count
            fn_a += len(gt_ids) - m_count
            fp_a += len(pr_ids) - m_count

        det_a = tp_a / (tp_a + fn_a + fp_a) if (tp_a + fn_a + fp_a) > 0 else 0.0
        if tp_a > 0:
            ass_sum = 0.0
            for gid, pid in matched_pairs_list:
                tpa = tpa_pairs[(gid, pid)]
                fna = gt_a_counts[gid] - tpa
                fpa = pr_a_counts[pid] - tpa
                denom = tpa + fna + fpa
                ass_sum += (tpa / denom) if denom > 0 else 0.0
            ass_a = ass_sum / tp_a
        else:
            ass_a = 0.0
        hota_scores.append(float(np.sqrt(det_a * ass_a)))

    hota = float(np.mean(hota_scores)) if hota_scores else 0.0

    return MOTMetricsResult(
        mota=round(mota * 100.0, 2),
        idf1=round(idf1 * 100.0, 2),
        hota=round(hota * 100.0, 2),
        id_switches=id_switches,
        false_positives=total_fp,
        false_negatives=total_fn,
        true_positives=total_tp,
        total_gt=total_gt,
        frames_evaluated=len(evaluated_frames),
    )
