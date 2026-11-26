"""Ground-truth annotation schema and geometry validator for ARGUS-X."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2

VALID_SCENARIO_TAGS = {
    "camera_transition",
    "occlusion",
    "lighting_change",
    "similar_appearance",
    "baseline",
}

PLACEHOLDER_IDENTITY_TOKENS = {
    "",
    "TODO",
    "TBD",
    "UNKNOWN",
    "REPLACE_ME",
    "PLACEHOLDER",
    "PERSON_?",
}


@dataclass(frozen=True)
class VideoMetadata:
    """Actual metadata extracted from a local video file."""

    camera_id: str
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def resolve_video_file(path: Union[str, Path]) -> Path:
    """Resolve a video file path, supporting directories containing a single .mp4."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        mp4s = sorted(
            p for p in candidate.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )
        if mp4s:
            return mp4s[0]
    raise FileNotFoundError(f"No video file found at: {path}")


def read_video_metadata(camera_id: str, path: Union[str, Path]) -> VideoMetadata:
    """Open a video file with OpenCV and read its actual dimensions and duration."""
    resolved = resolve_video_file(path)
    cap = cv2.VideoCapture(str(resolved))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video for {camera_id}: {resolved}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = (frame_count / fps) if fps > 0.0 else 0.0
    finally:
        cap.release()

    if width <= 0 or height <= 0 or frame_count <= 0 or fps <= 0.0:
        raise RuntimeError(
            f"Invalid video metadata for {camera_id} ({resolved}): "
            f"{width}x{height}, {frame_count} frames @ {fps} fps"
        )
    return VideoMetadata(
        camera_id=camera_id,
        path=resolved,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def validate_annotations_data(
    raw_data: Any,
    video_metadata: Optional[Mapping[str, VideoMetadata]] = None,
    require_non_empty: bool = False,
) -> list[str]:
    """Validate parsed JSON annotation data and return a list of error strings."""
    errors: list[str] = []

    if isinstance(raw_data, dict):
        items = raw_data.get("annotations")
    elif isinstance(raw_data, list):
        items = raw_data
    else:
        return ["Root JSON must be an object with 'annotations' or a JSON array."]

    if not isinstance(items, list):
        return ["'annotations' field must be a JSON array."]

    if require_non_empty and len(items) == 0:
        errors.append("Annotations list is empty; add ground-truth entries before running.")
        return errors

    for idx, item in enumerate(items):
        prefix = f"annotations[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be a JSON object.")
            continue

        # 1. camera_id
        cam_id = item.get("camera_id")
        if not isinstance(cam_id, str) or not cam_id.strip():
            errors.append(f"{prefix}.camera_id: must be a non-empty string.")
            cam_meta = None
        else:
            cam_meta = video_metadata.get(cam_id) if video_metadata else None
            if video_metadata is not None and cam_meta is None:
                errors.append(
                    f"{prefix}.camera_id: unknown camera '{cam_id}'. "
                    f"Expected one of {sorted(video_metadata.keys())}."
                )

        # 2. gt_identity_id
        gt_id = item.get("gt_identity_id")
        if not isinstance(gt_id, str) or gt_id.strip().upper() in PLACEHOLDER_IDENTITY_TOKENS:
            errors.append(
                f"{prefix}.gt_identity_id: invalid or placeholder identity '{gt_id}'."
            )

        # 3. timestamp
        ts = item.get("timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(float(ts)):
            errors.append(f"{prefix}.timestamp: must be a finite number (seconds).")
        else:
            ts_val = float(ts)
            if ts_val < 0.0:
                errors.append(f"{prefix}.timestamp: {ts_val} cannot be negative.")
            elif cam_meta is not None and ts_val > cam_meta.duration_sec + 1e-3:
                errors.append(
                    f"{prefix}.timestamp: {ts_val:.3f}s exceeds {cam_id} duration "
                    f"({cam_meta.duration_sec:.3f}s)."
                )

        # 4. bbox
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"{prefix}.bbox: must be a 4-element list [x1, y1, x2, y2].")
        else:
            parsed_box: list[float] = []
            valid_nums = True
            for coord in bbox:
                if isinstance(coord, bool) or not isinstance(coord, (int, float)) or not math.isfinite(float(coord)):
                    valid_nums = False
                    break
                parsed_box.append(float(coord))

            if not valid_nums:
                errors.append(f"{prefix}.bbox: all 4 coordinates must be finite numbers.")
            else:
                x1, y1, x2, y2 = parsed_box
                if parsed_box == [0.0, 0.0, 0.0, 0.0]:
                    errors.append(f"{prefix}.bbox: placeholder [0, 0, 0, 0] is not allowed.")
                elif x2 <= x1 or y2 <= y1:
                    errors.append(
                        f"{prefix}.bbox: invalid box {parsed_box}; require x2 > x1 and y2 > y1."
                    )
                elif (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
                    errors.append(
                        f"{prefix}.bbox: box {parsed_box} is too small (< 2px), likely a placeholder."
                    )
                elif x1 < 0.0 or y1 < 0.0:
                    errors.append(f"{prefix}.bbox: coordinates cannot be negative ({parsed_box}).")
                elif cam_meta is not None and (x2 > cam_meta.width + 1.0 or y2 > cam_meta.height + 1.0):
                    errors.append(
                        f"{prefix}.bbox: {parsed_box} exceeds {cam_id} resolution "
                        f"({cam_meta.width}x{cam_meta.height})."
                    )

        # 5. scenario_tag (optional)
        tag = item.get("scenario_tag")
        if tag is not None and (not isinstance(tag, str) or not tag.strip()):
            errors.append(f"{prefix}.scenario_tag: if provided, must be a non-empty string.")

    return errors


def validate_annotations_file(
    json_path: Union[str, Path],
    camera_videos: Optional[Mapping[str, Union[str, Path]]] = None,
    require_non_empty: bool = False,
) -> tuple[bool, list[str]]:
    """Load a ground-truth JSON file and validate it against video metadata."""
    target = Path(json_path)
    if not target.exists():
        return False, [f"Annotation file not found: {target}"]

    try:
        raw_data = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, [f"Invalid JSON syntax in {target}: {exc}"]

    meta_map: Optional[dict[str, VideoMetadata]] = None
    if camera_videos is not None:
        meta_map = {
            cam_id: read_video_metadata(cam_id, vpath)
            for cam_id, vpath in camera_videos.items()
        }

    errors = validate_annotations_data(
        raw_data,
        video_metadata=meta_map,
        require_non_empty=require_non_empty,
    )
    return len(errors) == 0, errors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate ARGUS-X ground-truth annotations.")
    parser.add_argument(
        "--annotations",
        default="evaluation/ground_truth/real_video_annotations.json",
    )
    parser.add_argument(
        "--cam1",
        default="datasets/camera1.mp4",
    )
    parser.add_argument(
        "--cam2",
        default="datasets/camera2.mp4",
    )
    parser.add_argument("--require-non-empty", action="store_true")
    args = parser.parse_args()

    ok, errors = validate_annotations_file(
        args.annotations,
        camera_videos={"camera1": args.cam1, "camera2": args.cam2},
        require_non_empty=args.require_non_empty,
    )
    if ok:
        print(f"VALID: {args.annotations}")
    else:
        print(f"INVALID: {args.annotations}")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
