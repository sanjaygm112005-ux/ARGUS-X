"""Modular YOLO11n object detector for ARGUS-X."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
from ultralytics import YOLO

from config import BASE_DIR, DetectionConfig, resolve_device

ImageSource = Union[str, Path, np.ndarray]


@dataclass(frozen=True)
class Detection:
    """Single object detection result."""

    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class Detector:
    """Reusable YOLO11n detector supporting images and OpenCV video frames."""

    def __init__(
        self,
        model_path: Union[str, Path] = DetectionConfig.model_name,
        conf_threshold: float = DetectionConfig.conf_threshold,
        iou_threshold: float = DetectionConfig.iou_threshold,
        imgsz: int = DetectionConfig.imgsz,
        target_classes: Optional[Sequence[int]] = None,
        device: str = "auto",
    ) -> None:
        """Initialize the detector and load the YOLO11n weights once.

        Args:
            model_path: Path or filename of the YOLO11 weights file.
            conf_threshold: Default confidence threshold in [0.0, 1.0].
            iou_threshold: NMS IoU threshold in [0.0, 1.0].
            imgsz: Inference image size in pixels.
            target_classes: Optional class IDs to keep (None keeps all classes).
            device: Preferred compute device ("auto", "mps", or "cpu").
        """
        self._validate_threshold(conf_threshold, "conf_threshold")
        self._validate_threshold(iou_threshold, "iou_threshold")
        if imgsz <= 0:
            raise ValueError(f"imgsz must be positive, got {imgsz}.")

        resolved_model = self._resolve_model_path(model_path)
        self.model_path: Path = resolved_model
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
            raise RuntimeError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

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
    def _load_frame(source: ImageSource) -> np.ndarray:
        if source is None:
            raise ValueError("Input image or video frame cannot be None.")

        if isinstance(source, np.ndarray):
            if source.size == 0 or source.ndim not in (2, 3):
                raise ValueError(
                    f"Invalid frame array with shape {source.shape}; expected non-empty 2D or 3D image."
                )
            if source.shape[0] == 0 or source.shape[1] == 0:
                raise ValueError(f"Frame has zero width or height: {source.shape}.")
            if source.ndim == 3 and source.shape[2] not in (1, 3, 4):
                raise ValueError(
                    f"Unsupported channel count {source.shape[2]} in frame shape {source.shape}."
                )
            return source

        if isinstance(source, (str, Path)):
            path_str = str(source).strip()
            if not path_str:
                raise ValueError("Image path cannot be empty.")

            image_path = Path(path_str)
            if not image_path.is_file():
                workspace_image = BASE_DIR / image_path
                if workspace_image.is_file():
                    image_path = workspace_image
                else:
                    raise FileNotFoundError(f"Image file not found: '{source}'.")

            frame = cv2.imread(str(image_path))
            if frame is None or frame.size == 0:
                raise ValueError(f"Failed to decode image file: '{image_path}'.")
            return frame

        raise TypeError(
            f"Unsupported input type {type(source).__name__}; expected str, Path, or numpy.ndarray."
        )

    def detect(
        self,
        source: ImageSource,
        conf_threshold: Optional[float] = None,
    ) -> list[Detection]:
        """Run object detection on an image file path or OpenCV BGR frame.

        Args:
            source: Image path (str or Path) or image/video frame (numpy.ndarray).
            conf_threshold: Optional per-call confidence override in [0.0, 1.0].

        Returns:
            List of Detection objects containing bbox, class_id, class_name, and confidence.
        """
        conf = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        self._validate_threshold(conf, "conf_threshold")

        frame = self._load_frame(source)
        results = self.model.predict(
            source=frame,
            conf=conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = str(self.model.names.get(class_id, class_id))
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                    )
                )

        return detections
