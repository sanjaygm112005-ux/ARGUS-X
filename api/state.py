"""Shared backend service state for ARGUS-X FastAPI server.

Holds a single CameraManager and MultiCameraIdentityManager instance so
models (YOLO11n / OSNet) and FAISS/SQLite state are never duplicated per request.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from association import AssociationConfig, CameraTransitionRule
from camera import CameraConfig, CameraManager
from identity_memory import (
    MultiCameraIdentityManager,
    MultiCameraRecoveryOutput,
)

logger = logging.getLogger(__name__)


_display_id_map: dict[str, int] = {}
_display_id_counter = 0


def _get_display_id(global_id: Optional[str]) -> tuple[str, str]:
    """Map global IDs (e.g. gid_...) to user-friendly consistent labels (e.g. PERSON 1)."""
    global _display_id_counter
    if not global_id:
        return "PERSON ?", ""
    if global_id not in _display_id_map:
        _display_id_counter += 1
        _display_id_map[global_id] = _display_id_counter
    return f"PERSON {_display_id_map[global_id]}", global_id


def _draw_overlays(
    frame: np.ndarray,
    outputs: Sequence[MultiCameraRecoveryOutput],
) -> np.ndarray:
    """Draw high-contrast bounding boxes, persistent person IDs, and recovery states."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    for out in outputs:
        x1, y1, x2, y2 = [int(round(v)) for v in out.bounding_box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        person_name, _ = _get_display_id(out.global_id)
        state_str = out.identity_state.value if hasattr(out.identity_state, "value") else str(out.identity_state)

        if state_str == "RECOVERED":
            color = (46, 204, 113)  # Bright green
            badge = "RECOVERED"
        elif state_str == "UNCERTAIN" or not out.global_id:
            color = (0, 165, 255)  # Amber
            badge = "UNCERTAIN"
        else:
            color = (255, 140, 0)  # Bright cyan/blue
            badge = "ACTIVE"

        # Bounding box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # High-visibility filled badge header
        label = f"{person_name} [{badge}]"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        box_y1 = max(0, y1 - lh - 8)
        box_y2 = y1
        cv2.rectangle(canvas, (x1, box_y1), (min(w - 1, x1 + lw + 8), box_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 4, box_y2 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


class ArgusServiceState:
    """Thread-safe container coordinating CameraManager and MultiCameraIdentityManager."""

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        index_path: Optional[Union[str, Path]] = None,
        transition_rules: Optional[Sequence[CameraTransitionRule]] = None,
        expire_timeout_sec: float = 3600.0,
    ) -> None:
        self._lock = threading.RLock()
        self.camera_manager = CameraManager(
            feature_extractor=None,
            inactive_timeout_sec=1.0,
            extraction_interval_sec=0.5,
        )
        self.identity_manager = MultiCameraIdentityManager(
            transition_rules=transition_rules,
            config=AssociationConfig(max_time_gap_sec=expire_timeout_sec),
            expire_timeout_sec=expire_timeout_sec,
            db_path=db_path,
            index_path=index_path,
        )
        self.latest_frames_jpeg: dict[str, bytes] = {}
        self.latest_outputs: dict[str, list[dict[str, object]]] = {}
        self.latest_timestamps: dict[str, float] = {}

    def register_camera(
        self,
        camera_id: str,
        source: Optional[Union[str, int]] = None,
        name: Optional[str] = None,
        fps: Optional[float] = None,
        enabled: bool = True,
        auto_start: bool = False,
    ) -> dict[str, object]:
        """Register a camera stream and optionally start it."""
        with self._lock:
            cfg = CameraConfig(
                camera_id=camera_id,
                source=source,
                name=name,
                fps=fps,
                enabled=enabled,
            )
            state = self.camera_manager.register_camera(cfg)
            if auto_start:
                self.camera_manager.start_camera(camera_id)
            return self._serialize_camera(camera_id, state)

    def disconnect_camera(self, camera_id: str) -> dict[str, object]:
        """Gracefully stop a camera stream and mark its active identities lost."""
        with self._lock:
            self.camera_manager.stop_camera(camera_id)
            affected = self.identity_manager.handle_camera_disconnect(
                camera_id, timestamp=time.time()
            )
            return {
                "camera_id": camera_id,
                "is_running": False,
                "affected_global_ids": affected,
            }

    def ingest_frame_outputs(
        self,
        camera_id: str,
        frame: Optional[np.ndarray],
        outputs: Sequence[MultiCameraRecoveryOutput],
        timestamp: float,
    ) -> None:
        """Store latest JPEG-encoded annotated frame and structured detections."""
        with self._lock:
            self.latest_outputs[camera_id] = [o.to_dict() for o in outputs]
            self.latest_timestamps[camera_id] = float(timestamp)
            try:
                cam = self.camera_manager.get_camera(camera_id)
                cam.is_running = True
                cam.frame_count += 1
                cam.last_timestamp = float(timestamp)
            except KeyError:
                pass
            if frame is not None and frame.size > 0:
                annotated = _draw_overlays(frame, outputs)
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    self.latest_frames_jpeg[camera_id] = buf.tobytes()

    def _serialize_camera(self, camera_id: str, state) -> dict[str, object]:
        return {
            "camera_id": camera_id,
            "name": state.config.name or camera_id,
            "source": str(state.config.source) if state.config.source is not None else None,
            "enabled": state.config.enabled,
            "is_running": state.is_running,
            "frame_count": state.frame_count,
            "last_timestamp": self.latest_timestamps.get(camera_id, state.last_timestamp),
            "active_detections": self.latest_outputs.get(camera_id, []),
        }

    def list_cameras(self) -> list[dict[str, object]]:
        """Return serialized status for all registered cameras."""
        with self._lock:
            return [
                self._serialize_camera(cid, self.camera_manager.get_camera(cid))
                for cid in self.camera_manager.list_camera_ids()
            ]

    def get_camera_stream_payload(self, camera_id: str) -> dict[str, object]:
        """Return latest frame (base64 JPEG) and detections for WebSocket streaming."""
        with self._lock:
            state = self.camera_manager.get_camera(camera_id)
            jpg = self.latest_frames_jpeg.get(camera_id)
            return {
                "camera_id": camera_id,
                "is_running": state.is_running,
                "timestamp": self.latest_timestamps.get(camera_id, state.last_timestamp),
                "detections": self.latest_outputs.get(camera_id, []),
                "frame_b64": base64.b64encode(jpg).decode("ascii") if jpg else None,
            }

    def get_statistics(self) -> dict[str, object]:
        """Compute overview KPIs for the dashboard."""
        with self._lock:
            cameras = self.list_cameras()
            total_cams = len(cameras)
            online_cams = sum(1 for c in cameras if c["is_running"])
            identities = self.identity_manager.list_identities()

            counts = {
                "ACTIVE": 0,
                "TEMPORARILY_LOST": 0,
                "RECOVERED": 0,
                "UNCERTAIN": 0,
                "EXPIRED": 0,
                "UNKNOWN": 0,
            }
            for rec in identities:
                counts[rec.state.value] = counts.get(rec.state.value, 0) + 1

            pending_unresolved = self.identity_manager.list_unresolved_observations(
                status="PENDING"
            )
            return {
                "total_cameras": total_cams,
                "online_cameras": online_cams,
                "offline_cameras": total_cams - online_cams,
                "total_identities": len(identities),
                "active_identities": counts["ACTIVE"],
                "temporarily_lost_identities": counts["TEMPORARILY_LOST"],
                "recovered_identities": counts["RECOVERED"],
                "expired_identities": counts["EXPIRED"],
                "uncertain_observations": len(pending_unresolved),
                "total_observations_in_memory": self.identity_manager.memory.size,
                "total_conflicts": len(self.identity_manager.get_conflict_events()),
                "total_audit_logs": len(self.identity_manager.get_audit_logs()),
            }

    def close(self) -> None:
        """Release camera and identity memory resources."""
        with self._lock:
            self.camera_manager.close()
            self.identity_manager.close()
