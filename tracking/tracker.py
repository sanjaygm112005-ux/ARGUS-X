"""Modular YOLO11n + BoT-SORT multi-object tracker for ARGUS-X.

Maintains local, short-horizon track IDs across consecutive frames.
Note: BoT-SORT handles frame-to-frame motion/IoU association only and does not
provide long-horizon cross-camera or post-occlusion global identity recovery.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

from config import BASE_DIR, DetectionConfig, TrackingConfig, resolve_device

VideoSource = Union[str, Path, int]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackedObject:
    """Single tracked object with a local short-horizon track ID."""

    track_id: int
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class ObjectTracker:
    """Reusable YOLO11n + BoT-SORT tracker for video frames, files, and webcams."""

    def __init__(
        self,
        model_path: Union[str, Path] = DetectionConfig.model_name,
        tracker_config: Union[str, Path] = TrackingConfig.tracker_config,
        conf_threshold: float = DetectionConfig.conf_threshold,
        iou_threshold: float = DetectionConfig.iou_threshold,
        imgsz: int = DetectionConfig.imgsz,
        target_classes: Optional[Sequence[int]] = None,
        device: str = "auto",
    ) -> None:
        """Initialize YOLO11n weights and BoT-SORT configuration.

        Args:
            model_path: Path or filename of the YOLO11 weights file.
            tracker_config: BoT-SORT config name (e.g., "botsort.yaml") or path.
            conf_threshold: Detection confidence threshold in [0.0, 1.0].
            iou_threshold: NMS IoU threshold in [0.0, 1.0].
            imgsz: Inference image size in pixels.
            target_classes: Optional class IDs to track (None tracks all classes).
            device: Preferred compute device ("auto", "mps", or "cpu").
        """
        self._validate_threshold(conf_threshold, "conf_threshold")
        self._validate_threshold(iou_threshold, "iou_threshold")
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}.")

        self.model_path: Path = self._resolve_model_path(model_path)
        self.tracker_config: str = self._resolve_tracker_config(tracker_config)
        self.conf_threshold: float = float(conf_threshold)
        self.iou_threshold: float = float(iou_threshold)
        self.imgsz: int = int(imgsz)
        self.target_classes: Optional[list[int]] = (
            [int(c) for c in target_classes] if target_classes is not None else None
        )
        self.device: str = resolve_device(device)

        try:
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            logger.exception("Failed to load YOLO model from '%s'", self.model_path)
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

        logger.info(
            "ObjectTracker ready | model=%s | tracker=%s | device=%s",
            self.model_path.name,
            self.tracker_config,
            self.device,
        )

    @staticmethod
    def _validate_threshold(value: float, name: str) -> None:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")

    @staticmethod
    def _resolve_model_path(model_path: Union[str, Path]) -> Path:
        if not model_path or (isinstance(model_path, str) and not model_path.strip()):
            raise ValueError("model_path must be a non-empty string or Path.")

        candidate = Path(model_path)
        if candidate.is_file():
            return candidate.resolve()

        workspace_candidate = BASE_DIR / candidate
        if workspace_candidate.is_file():
            return workspace_candidate.resolve()

        raise FileNotFoundError(
            f"YOLO model file not found: '{model_path}'. "
            f"Place '{candidate.name}' in '{BASE_DIR}' or pass a valid file path."
        )

    @staticmethod
    def _resolve_tracker_config(tracker_config: Union[str, Path]) -> str:
        if not tracker_config or (
            isinstance(tracker_config, str) and not tracker_config.strip()
        ):
            raise ValueError("tracker_config must be a non-empty string or Path.")

        candidate = Path(tracker_config)
        if candidate.is_file():
            return str(candidate.resolve())

        workspace_candidate = BASE_DIR / candidate
        if workspace_candidate.is_file():
            return str(workspace_candidate.resolve())

        builtin_tracker = (
            Path(ultralytics.__file__).resolve().parent
            / "cfg"
            / "trackers"
            / candidate.name
        )
        if builtin_tracker.is_file() and candidate.parent == Path("."):
            return str(builtin_tracker)

        raise FileNotFoundError(
            f"Tracker configuration file not found: '{tracker_config}'."
        )

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Input video frame cannot be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(
                f"Expected numpy.ndarray video frame, got {type(frame).__name__}."
            )
        if frame.size == 0 or frame.ndim not in (2, 3):
            raise ValueError(
                f"Invalid frame shape {frame.shape}; expected non-empty 2D or 3D image."
            )
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame has zero width or height: {frame.shape}.")
        if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"Unsupported channel count {frame.shape[2]} in frame shape {frame.shape}."
            )
        return frame

    def reset(self) -> None:
        """Reset BoT-SORT local track state between independent video streams."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                if hasattr(tracker, "reset"):
                    tracker.reset()
        logger.debug("BoT-SORT local track state reset.")

    def track_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> list[TrackedObject]:
        """Track objects in a single consecutive BGR video frame.

        Args:
            frame: OpenCV video frame as a numpy.ndarray.
            conf_threshold: Optional per-call confidence override in [0.0, 1.0].

        Returns:
            List of TrackedObject instances with local BoT-SORT track IDs.
        """
        valid_frame = self._validate_frame(frame)
        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        self._validate_threshold(conf, "conf_threshold")

        try:
            results = self.model.track(
                source=valid_frame,
                persist=True,
                tracker=self.tracker_config,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                classes=self.target_classes,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Tracking inference failed on frame")
            raise RuntimeError(f"Tracking inference failed: {exc}") from exc

        tracked_objects: list[TrackedObject] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                if box.id is None:
                    logger.debug("Skipping unconfirmed detection without local track ID.")
                    continue

                track_id = int(box.id[0].item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = str(self.model.names.get(class_id, class_id))
                tracked_objects.append(
                    TrackedObject(
                        track_id=track_id,
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        return tracked_objects

    def _open_capture(self, source: VideoSource) -> cv2.VideoCapture:
        if source is None:
            raise ValueError("Video source cannot be None.")

        if isinstance(source, bool):
            raise TypeError("Video source cannot be a boolean.")

        if isinstance(source, int):
            if source < 0:
                raise ValueError(f"Webcam index must be non-negative, got {source}.")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"Unable to open webcam device index {source}.")
            return cap

        if isinstance(source, (str, Path)):
            source_str = str(source).strip()
            if not source_str:
                raise ValueError("Video source path cannot be empty.")

            is_stream_url = "://" in source_str
            if not is_stream_url:
                video_path = Path(source_str)
                if not video_path.is_file():
                    workspace_video = BASE_DIR / video_path
                    if workspace_video.is_file():
                        video_path = workspace_video
                    else:
                        raise FileNotFoundError(f"Video file not found: '{source}'.")
                source_str = str(video_path)

            cap = cv2.VideoCapture(source_str)
            if not cap.isOpened():
                cap.release()
                raise ValueError(f"Unable to open video source: '{source}'.")
            return cap

        raise TypeError(
            f"Unsupported video source type {type(source).__name__}; "
            "expected file path (str/Path), stream URL (str), or webcam index (int)."
        )

    def track_stream(
        self,
        source: VideoSource,
        conf_threshold: Optional[float] = None,
        max_frames: Optional[int] = None,
        reset_tracker: bool = True,
    ) -> Iterator[tuple[int, np.ndarray, list[TrackedObject]]]:
        """Yield tracked objects frame-by-frame from a video file, stream, or webcam.

        Args:
            source: Video file path, stream URL, or integer webcam index (e.g., 0).
            conf_threshold: Optional per-call confidence override in [0.0, 1.0].
            max_frames: Optional maximum number of frames to process.
            reset_tracker: Reset BoT-SORT state before starting the stream.

        Yields:
            Tuples of (frame_index, frame_bgr, tracked_objects).
        """
        if max_frames is not None and max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}.")

        if reset_tracker:
            self.reset()

        cap = self._open_capture(source)
        frame_index = 0
        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                tracks = self.track_frame(frame, conf_threshold=conf_threshold)
                yield frame_index, frame, tracks
                frame_index += 1
        finally:
            cap.release()
