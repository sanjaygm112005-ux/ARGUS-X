"""Markdown evaluation report generator for ARGUS-X benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from evaluation.benchmark import BenchmarkResult

if TYPE_CHECKING:
    from evaluation.real_video_validation import BottleneckAnalysis


def generate_markdown_report(
    result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None,
    bottleneck: Optional["BottleneckAnalysis"] = None,
) -> str:
    """Format a BenchmarkResult into a clear, factual Markdown report."""
    hw = result.hardware
    validation_scope = (
        "Real Local Video Validation"
        if result.is_real_video
        else "Synthetic / Unit Test Run (NOT real-world CCTV validation)"
    )

    lines: list[str] = [
        "# ARGUS-X Evaluation & Validation Report",
        "",
        "## 1. Validation Scope & Provenance",
        f"- **Validation Category:** {validation_scope}",
        f"- **Video Sources:** {', '.join(result.video_sources) if result.video_sources else 'None'}",
        f"- **Ground-Truth Verified:** {'Yes' if result.metrics is not None else 'No (predictions unverified)'}",
        "",
        "## 2. Hardware & Inference Device",
        f"- **OS / Architecture:** {hw.os_name} (`{hw.machine_arch}`)",
        f"- **Python / PyTorch:** {hw.python_version} / {hw.pytorch_version}",
        f"- **Active Compute Device:** `{hw.active_device}` (MPS built={hw.mps_built}, available={hw.mps_available})",
        f"- **CPU Cores (logical) / RAM:** {hw.cpu_count_logical} / {hw.total_ram_gb:.2f} GB",
        "",
        "## 3. Measured Performance & Bottlenecks",
        f"- **Overall Detection FPS:** {result.overall_detection_fps:.2f}",
        f"- **Overall Tracking FPS:** {result.overall_tracking_fps:.2f}",
        f"- **Avg Re-ID Extraction Time:** {result.avg_reid_extraction_ms:.2f} ms/frame",
        f"- **Avg Association Time:** {result.avg_association_ms:.2f} ms/frame",
        f"- **Avg End-to-End Latency:** {result.avg_e2e_latency_ms:.2f} ms/frame",
        f"- **Process CPU / Peak RSS Memory:** {result.cpu_percent:.1f}% / {result.peak_memory_mb:.2f} MB",
    ]
    if bottleneck is not None:
        lines.extend(
            [
                f"- **Primary Bottleneck:** {bottleneck.primary_bottleneck} "
                f"(Tracking/Detection={bottleneck.tracking_and_other_share_pct:.1f}%, "
                f"Re-ID={bottleneck.reid_share_pct:.1f}%, "
                f"Association={bottleneck.association_share_pct:.1f}%)",
            ]
        )

    lines.extend(
        [
            "",
            "### Camera-Wise Performance",
            "| Camera ID | Frames | Detection FPS | Tracking FPS | Re-ID (ms) | Assoc (ms) | E2E Latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cs in result.camera_stats:
        lines.append(
            f"| {cs.camera_id} | {cs.frames_processed} | {cs.detection_fps:.2f} | "
            f"{cs.tracking_fps:.2f} | {cs.avg_reid_ms:.2f} | "
            f"{cs.avg_association_ms:.2f} | {cs.avg_e2e_latency_ms:.2f} |"
        )

    lines.extend(["", "## 4. Ground-Truth Verified Metrics (Frame-Level vs. Event-Level)"])
    if result.metrics is None:
        lines.append(
            "- **Unverified Predictions Only:** No ground-truth annotations were supplied; accuracy, precision, recall, and F1 are not measured."
        )
    else:
        m = result.metrics
        lines.extend(
            [
                "### Frame-Level Identity Metrics",
                f"- **Total Ground-Truth Observations:** {m.total_ground_truth}",
                f"- **Frame-Level Accuracy (`TP / Total`):** {m.identity_recovery_accuracy:.4f}",
                f"- **Precision / Recall / F1-Score:** {m.precision:.4f} / {m.recall:.4f} / {m.f1_score:.4f}",
                f"- **True Positives / False Matches / Missed:** {m.true_positives} / {m.false_identity_matches} / {m.missed_identity_recoveries}",
                f"- **ID Switches:** {m.id_switches}",
            ]
        )
        if m.event_metrics is not None:
            em = m.event_metrics
            lines.extend(
                [
                    "",
                    "### Recovery-Event-Level Metrics",
                    f"- **Total Reappearance / Transition Events:** {em.total_recovery_events}",
                    f"- **Correct / False / Missed Recoveries:** {em.correct_recoveries} / {em.false_recoveries} / {em.missed_recoveries}",
                    f"- **Recovery Event Accuracy:** {em.recovery_event_accuracy:.4f}",
                    f"- **Event Precision / Recall / F1-Score:** {em.event_precision:.4f} / {em.event_recall:.4f} / {em.event_f1_score:.4f}",
                ]
            )
        if m.scenario_breakdown:
            lines.extend(
                [
                    "",
                    "### Scenario Breakdown",
                    "| Scenario | Total | Accuracy | Precision | Recall | F1 | ID Switches |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for tag, vals in sorted(m.scenario_breakdown.items()):
                lines.append(
                    f"| {tag} | {int(vals['total'])} | {vals['accuracy']:.4f} | "
                    f"{vals['precision']:.4f} | {vals['recall']:.4f} | "
                    f"{vals['f1_score']:.4f} | {int(vals['id_switches'])} |"
                )

    lines.extend(["", "## 5. Unresolved Predictions, Failure Cases & Limitations"])
    if result.failure_cases:
        for fc in result.failure_cases[:20]:
            lines.append(f"- {fc}")
    else:
        lines.append("- No UNCERTAIN failure cases logged in this run.")

    lines.extend(
        [
            "- **Known Limitations:** Appearance-only Re-ID degrades under severe clothing similarity, extreme lighting shifts, or heavy occlusion unless paired with calibrated multi-camera geometry and transition rules.",
        ]
    )

    report = "\n".join(lines) + "\n"
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
    return report
