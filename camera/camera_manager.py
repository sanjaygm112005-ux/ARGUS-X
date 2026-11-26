"""Multi-camera stream and local-tracker management for ARGUS-X.

Each registered camera maintains its own independent OpenCV capture handle,
independent BoT-SORT ``ObjectTracker`` state, and independent ``TrackletManager``,
while sharing a single ``PersonFeatureExtractor`` to keep memory and compute
efficient on Apple Silicon.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from association.cost_matrix import KalmanMotionEstimate
from identity_memory.tracklet_manager import TrackletManager, TrackletUpdateResult
from reidentification.feature_extractor import PersonFeatureExtractor
from tracking.tracker import ObjectTracker, TrackedObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single camera stream."""

    camera_id: str
    source: Optional[Union[str, Path, int]] = None
    name: Optional[str] = None
    fps: Optional[float] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.camera_id, str) or not self.camera_id.strip():
            raise ValueError("camera_id must be a non-empty string.")
        if self.fps is not None and self.fps <= 0.0:
            raise ValueError(f"fps must be positive, got {self.fps}.")


@dataclass
class CameraStreamState:
    """Runtime state for a registered camera stream."""

    config: CameraConfig
    tracklet_manager: TrackletManager
    tracker: Optional[ObjectTracker] = None
    capture: Optional[cv2.VideoCapture] = None
    is_running: bool = False
    frame_count: int = 0
    last_timestamp: Optional[float] = None


class CameraManager:
    """Manages multiple independent camera streams and their local trackers."""

    def __init__(
        self,
        feature_extractor: Optional[PersonFeatureExtractor] = None,
        inactive_timeout_sec: float = 1.0,
        extraction_interval_sec: float = 0.5,
        tracker_factory: Optional[Callable[[CameraConfig], ObjectTracker]] = None,
    ) -> None:
        """Initialize CameraManager.

        Args:
            feature_extractor: Shared ``PersonFeatureExtractor`` reused across
                all cameras to avoid loading OSNet multiple times.
            inactive_timeout_sec: Tracklet inactivity timeout per camera.
            extraction_interval_sec: Minimum OSNet extraction interval per track.
            tracker_factory: Optional factory creating an independent
                ``ObjectTracker`` when a camera starts.
        """
        if inactive_timeout_sec < 0.0:
            raise ValueError("inactive_timeout_sec must be >= 0.")
        if extraction_interval_sec < 0.0:
            raise ValueError("extraction_interval_sec must be >= 0.")

        self.feature_extractor = feature_extractor
        self.inactive_timeout_sec = float(inactive_timeout_sec)
        self.extraction_interval_sec = float(extraction_interval_sec)
        self.tracker_factory = tracker_factory
        self._cameras: dict[str, CameraStreamState] = {}

    def register_camera(
        self,
        config: CameraConfig,
        tracker: Optional[ObjectTracker] = None,
    ) -> CameraStreamState:
        """Register a new camera with its own independent TrackletManager."""
        if config.camera_id in self._cameras:
            raise ValueError(f"Camera '{config.camera_id}' is already registered.")

        tm = TrackletManager(
            camera_id=config.camera_id,
            feature_extractor=self.feature_extractor,
            inactive_timeout_sec=self.inactive_timeout_sec,
            extraction_interval_sec=self.extraction_interval_sec,
        )
        state = CameraStreamState(
            config=config,
            tracklet_manager=tm,
            tracker=tracker,
        )
        self._cameras[config.camera_id] = state
        logger.info("Registered camera '%s' (source=%s)", config.camera_id, config.source)
        return state

    def unregister_camera(self, camera_id: str) -> None:
        """Stop and remove a registered camera."""
        if camera_id not in self._cameras:
            raise KeyError(f"Unknown camera_id '{camera_id}'.")
        self.stop_camera(camera_id)
        self._cameras.pop(camera_id)
        logger.info("Unregistered camera '%s'", camera_id)

    def get_camera(self, camera_id: str) -> CameraStreamState:
        """Return runtime state for ``camera_id``."""
        if camera_id not in self._cameras:
            raise KeyError(f"Unknown camera_id '{camera_id}'.")
        return self._cameras[camera_id]

    def list_camera_ids(self) -> list[str]:
        """Return all registered camera IDs in registration order."""
        return list(self._cameras.keys())

    def start_camera(self, camera_id: str) -> None:
        """Start a registered camera stream and ensure its local tracker is ready."""
        state = self.get_camera(camera_id)
        if state.is_running:
            return
        if not state.config.enabled:
            logger.info("Camera '%s' is disabled; skipping start.", camera_id)
            return

        if state.config.source is not None:
            src = state.config.source
            cap_arg = int(src) if isinstance(src, int) else str(src)
            cap = cv2.VideoCapture(cap_arg)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(
                    f"Failed to open video source for camera '{camera_id}': {src!r}"
                )
            state.capture = cap

        if state.tracker is None and self.tracker_factory is not None:
            state.tracker = self.tracker_factory(state.config)

        state.is_running = True
        logger.info("Started camera '%s'", camera_id)

    def start_all(self) -> None:
        """Start all enabled registered cameras."""
        for camera_id in self._cameras:
            self.start_camera(camera_id)

    def stop_camera(self, camera_id: str) -> None:
        """Stop a camera stream and release its OpenCV VideoCapture handle."""
        state = self.get_camera(camera_id)
        if state.capture is not None:
            state.capture.release()
            state.capture = None
        state.tracklet_manager.retire_all()
        state.is_running = False
        logger.info("Stopped camera '%s'", camera_id)

    def stop_all(self) -> None:
        """Stop all registered cameras and release all capture handles."""
        for camera_id in list(self._cameras.keys()):
            self.stop_camera(camera_id)

    def read_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Read the next frame from a started camera's VideoCapture."""
        state = self.get_camera(camera_id)
        if not state.is_running or state.capture is None:
            return None
        ok, frame = state.capture.read()
        if not ok or frame is None:
            return None
        state.frame_count += 1
        return frame

    def process_camera_detections(
        self,
        camera_id: str,
        tracked_objects: Sequence[TrackedObject],
        timestamp: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        precomputed_embeddings: Optional[Mapping[int, np.ndarray]] = None,
        motion_states: Optional[Mapping[int, KalmanMotionEstimate]] = None,
    ) -> TrackletUpdateResult:
        """Update a camera's independent TrackletManager with local detections."""
        state = self.get_camera(camera_id)
        ts = float(time.time() if timestamp is None else timestamp)
        state.last_timestamp = ts
        return state.tracklet_manager.update(
            tracked_objects=tracked_objects,
            timestamp=ts,
            frame=frame,
            precomputed_embeddings=precomputed_embeddings,
            motion_states=motion_states,
        )

    def process_camera_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> TrackletUpdateResult:
        """Run the camera's independent local tracker and update its tracklets."""
        state = self.get_camera(camera_id)
        if state.tracker is None:
            state.tracker = (
                self.tracker_factory(state.config)
                if self.tracker_factory is not None
                else ObjectTracker()
            )
        tracked = state.tracker.track_frame(frame)
        return self.process_camera_detections(
            camera_id=camera_id,
            tracked_objects=tracked,
            timestamp=timestamp,
            frame=frame,
        )

    def close(self) -> None:
        """Release all camera resources."""
        self.stop_all()

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
