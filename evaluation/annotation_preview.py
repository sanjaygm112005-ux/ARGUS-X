"""Generate contact sheets with YOLO11n person detections for manual annotation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from detection.detector import Detector
from evaluation.annotation_validator import read_video_metadata


def build_contact_sheet(
    tiles: list[np.ndarray],
    cols: int = 6,
    tile_width: int = 240,
) -> np.ndarray:
    """Arrange annotated frame tiles into a clean grid image."""
    if not tiles:
        raise ValueError("No frames provided for contact sheet.")

    h0, w0 = tiles[0].shape[:2]
    tile_height = max(1, int(round(tile_width * (h0 / float(w0)))))
    rows = math.ceil(len(tiles) / cols)
    canvas = np.full((rows * tile_height, cols * tile_width, 3), 24, dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        resized = cv2.resize(tile, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        y1, y2 = r * tile_height, (r + 1) * tile_height
        x1, x2 = c * tile_width, (c + 1) * tile_width
        canvas[y1:y2, x1:x2] = resized

    return canvas


def generate_annotation_preview(
    camera_videos: dict[str, Union[str, Path]],
    output_dir: Union[str, Path] = "evaluation/results/annotation_preview",
    interval_sec: float = 1.0,
) -> dict[str, object]:
    """Extract frames at regular intervals, run YOLO11n, and save contact sheets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    detector = Detector(target_classes=[0], device="auto")
    summary: dict[str, object] = {}

    for cam_id, raw_path in camera_videos.items():
        meta = read_video_metadata(cam_id, raw_path)
        cap = cv2.VideoCapture(str(meta.path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open {meta.path}")

        step_frames = max(1, int(round(interval_sec * meta.fps)))
        frame_indices = list(range(0, meta.frame_count, step_frames))
        if (meta.frame_count - 1) not in frame_indices:
            frame_indices.append(meta.frame_count - 1)

        tiles: list[np.ndarray] = []
        sampled_records: list[dict[str, object]] = []

        try:
            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                ts = round(f_idx / meta.fps, 3)
                dets = detector.detect(frame)
                annotated = frame.copy()

                det_entries: list[dict[str, object]] = []
                for det_num, det in enumerate(dets, start=1):
                    x1, y1, x2, y2 = [int(round(v)) for v in det.bbox]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 120), 2)
                    label = f"#{det_num} ({det.confidence:.2f})"
                    cv2.rectangle(
                        annotated,
                        (x1, max(0, y1 - 24)),
                        (min(meta.width, x1 + 115), y1),
                        (0, 220, 120),
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 3, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    det_entries.append(
                        {
                            "temp_id": det_num,
                            "bbox": [round(float(v), 1) for v in det.bbox],
                            "confidence": round(float(det.confidence), 4),
                        }
                    )

                # Header bar with camera ID, timestamp, frame index, and count
                cv2.rectangle(annotated, (0, 0), (meta.width, 34), (15, 15, 20), -1)
                header = f"{cam_id} | t={ts:.2f}s | f={f_idx} | people={len(dets)}"
                cv2.putText(
                    annotated,
                    header,
                    (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                tiles.append(annotated)
                sampled_records.append(
                    {
                        "frame_index": f_idx,
                        "timestamp": ts,
                        "person_count": len(dets),
                        "detections": det_entries,
                    }
                )
        finally:
            cap.release()

        sheet = build_contact_sheet(tiles, cols=6, tile_width=240)
        sheet_path = out_path / f"{cam_id}_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet)

        summary[cam_id] = {
            "video_path": str(meta.path),
            "resolution": [meta.width, meta.height],
            "fps": round(meta.fps, 3),
            "frame_count": meta.frame_count,
            "duration_sec": round(meta.duration_sec, 3),
            "contact_sheet": str(sheet_path),
            "sampled_frames": sampled_records,
        }

    index_path = out_path / "preview_index.json"
    index_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = generate_annotation_preview(
        {
            "camera1": "datasets/camera1.mp4",
            "camera2": "datasets/camera2.mp4",
        }
    )
    print(json.dumps(result, indent=2))
