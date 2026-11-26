"""Interactive Streamlit ground-truth annotation tool for ARGUS-X real videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import streamlit as st

from detection.detector import Detector
from evaluation.annotation_validator import (
    VALID_SCENARIO_TAGS,
    read_video_metadata,
    validate_annotations_data,
)

DEFAULT_VIDEOS = {
    "camera1": "datasets/camera1.mp4",
    "camera2": "datasets/camera2.mp4",
}
DEFAULT_GT_PATH = Path("evaluation/ground_truth/real_video_annotations.json")


@st.cache_resource
def get_detector() -> Detector:
    return Detector(target_classes=[0], device="auto")


@st.cache_data
def load_video_info() -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for cam_id, raw_path in DEFAULT_VIDEOS.items():
        meta = read_video_metadata(cam_id, raw_path)
        info[cam_id] = {
            "meta": meta,
            "path": str(meta.path),
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_sec": meta.duration_sec,
        }
    return info


@st.cache_data
def detect_frame(video_path: str, frame_idx: int) -> tuple[Any, list[dict[str, Any]]]:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None, []

    dets = get_detector().detect(frame)
    det_list = []
    for idx, det in enumerate(dets, start=1):
        det_list.append(
            {
                "temp_id": idx,
                "bbox": [round(float(v), 1) for v in det.bbox],
                "confidence": round(float(det.confidence), 4),
            }
        )
    return frame, det_list


def load_existing_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("annotations"), list):
                return data
        except Exception:
            pass
    return {
        "description": "ARGUS-X real-video ground-truth annotations for multi-camera identity recovery evaluation.",
        "videos": DEFAULT_VIDEOS,
        "annotations": [],
    }


def main() -> None:
    st.set_page_config(page_title="ARGUS-X Ground-Truth Annotator", layout="wide")
    st.title("ARGUS-X Ground-Truth Annotation Tool")
    st.caption(
        "Navigate frames, inspect YOLO11n person detections, and assign consistent "
        "ground-truth identities (e.g. PERSON_A, PERSON_B) across both cameras."
    )

    video_info = load_video_info()
    meta_map = {cid: v["meta"] for cid, v in video_info.items()}

    if "gt_doc" not in st.session_state:
        st.session_state.gt_doc = load_existing_json(DEFAULT_GT_PATH)
    if "known_identities" not in st.session_state:
        existing_ids = sorted(
            {
                str(a.get("gt_identity_id"))
                for a in st.session_state.gt_doc.get("annotations", [])
                if a.get("gt_identity_id")
            }
        )
        defaults = ["PERSON_A", "PERSON_B", "PERSON_C", "PERSON_D"]
        st.session_state.known_identities = list(dict.fromkeys(existing_ids + defaults))

    col_ctrl, col_view = st.columns([1.1, 1.4])

    with col_ctrl:
        cam_id = st.selectbox("Select Camera", list(video_info.keys()))
        cam_data = video_info[cam_id]
        max_frame = cam_data["frame_count"] - 1

        step_mode = st.radio(
            "Navigation Step",
            ["1 frame", "15 frames (~0.5s)", "30 frames (~1.0s)"],
            horizontal=True,
            index=1,
        )
        step_val = {"1 frame": 1, "15 frames (~0.5s)": 15, "30 frames (~1.0s)": 30}[step_mode]

        frame_key = f"frame_idx_{cam_id}"
        if frame_key not in st.session_state:
            st.session_state[frame_key] = 0

        nav_prev, nav_next = st.columns(2)
        if nav_prev.button("Previous Frame", use_container_width=True):
            st.session_state[frame_key] = max(0, st.session_state[frame_key] - step_val)
        if nav_next.button("Next Frame", use_container_width=True):
            st.session_state[frame_key] = min(max_frame, st.session_state[frame_key] + step_val)

        frame_idx = st.slider(
            "Frame Index",
            min_value=0,
            max_value=max_frame,
            value=st.session_state[frame_key],
            key=f"slider_{cam_id}",
        )
        st.session_state[frame_key] = frame_idx
        timestamp = round(frame_idx / cam_data["fps"], 3)
        st.info(
            f"**{cam_id}** | Frame `{frame_idx}/{max_frame}` | "
            f"Timestamp: `{timestamp:.3f}s` / `{cam_data['duration_sec']:.2f}s` | "
            f"Resolution: `{cam_data['width']}x{cam_data['height']}`"
        )

        frame, detections = detect_frame(cam_data["path"], frame_idx)

        if detections:
            det_options = {
                f"#{d['temp_id']} | bbox={d['bbox']} | conf={d['confidence']:.2f}": d
                for d in detections
            }
            selected_label = st.selectbox("Select Detected Person", list(det_options.keys()))
            selected_det = det_options[selected_label]
        else:
            st.warning("No people detected by YOLO11n in this frame.")
            selected_det = None

        id_col1, id_col2 = st.columns([1.2, 1.0])
        chosen_id = id_col1.selectbox(
            "Ground-Truth Identity",
            st.session_state.known_identities,
        )
        custom_id = id_col2.text_input("Or Enter New Identity (e.g. PERSON_E)", value="")
        final_gt_id = custom_id.strip() if custom_id.strip() else chosen_id

        scenario_tag = st.selectbox(
            "Scenario Tag",
            sorted(VALID_SCENARIO_TAGS),
            index=sorted(VALID_SCENARIO_TAGS).index("camera_transition"),
        )

        if st.button("Add Annotation", type="primary", disabled=(selected_det is None), use_container_width=True):
            if final_gt_id not in st.session_state.known_identities:
                st.session_state.known_identities.append(final_gt_id)
            entry = {
                "camera_id": cam_id,
                "timestamp": timestamp,
                "gt_identity_id": final_gt_id,
                "bbox": selected_det["bbox"],
                "scenario_tag": scenario_tag,
            }
            st.session_state.gt_doc["annotations"].append(entry)
            st.success(f"Added {final_gt_id} on {cam_id} at t={timestamp:.3f}s")

    with col_view:
        if frame is not None:
            vis = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
                is_sel = selected_det is not None and d["temp_id"] == selected_det["temp_id"]
                color = (0, 180, 255) if is_sel else (0, 220, 120)
                thickness = 3 if is_sel else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    vis,
                    f"#{d['temp_id']} ({d['confidence']:.2f})",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption=f"{cam_id} @ {timestamp:.3f}s", use_container_width=False, width=420)

    st.divider()
    annotations = st.session_state.gt_doc.get("annotations", [])
    st.subheader(f"Current Annotations ({len(annotations)})")

    if annotations:
        st.dataframe(annotations, use_container_width=True)
        del_idx = st.number_input(
            "Delete Annotation Row Index",
            min_value=0,
            max_value=max(0, len(annotations) - 1),
            value=0,
            step=1,
        )
        if st.button("Remove Selected Row"):
            annotations.pop(int(del_idx))
            st.rerun()

    save_col, val_col = st.columns(2)
    if val_col.button("Validate Annotations", use_container_width=True):
        errs = validate_annotations_data(
            st.session_state.gt_doc,
            video_metadata=meta_map,
            require_non_empty=True,
        )
        if errs:
            for e in errs:
                st.error(e)
        else:
            st.success("All annotations passed schema, bounds, and duration checks!")

    if save_col.button("Save to real_video_annotations.json", type="primary", use_container_width=True):
        errs = validate_annotations_data(
            st.session_state.gt_doc,
            video_metadata=meta_map,
            require_non_empty=False,
        )
        if errs:
            for e in errs:
                st.error(e)
        else:
            DEFAULT_GT_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_GT_PATH.write_text(
                json.dumps(st.session_state.gt_doc, indent=2),
                encoding="utf-8",
            )
            st.success(f"Saved {len(annotations)} annotations to {DEFAULT_GT_PATH}")


if __name__ == "__main__":
    main()
