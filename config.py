"""Central configuration for ARGUS-X.

Handles project paths, Apple Silicon (MPS) device selection with CPU fallback,
and independent per-module configuration dataclasses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Allow PyTorch ops unsupported on Apple Silicon MPS to fall back to CPU cleanly.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

BASE_DIR: Path = Path(__file__).resolve().parent
DATASETS_DIR: Path = BASE_DIR / "datasets"
LOGS_DIR: Path = BASE_DIR / "logs"


def resolve_device(preferred: str = "auto") -> str:
    """Resolve compute device with Apple Silicon MPS priority and CPU fallback.

    Args:
        preferred: Requested device ("auto", "mps", or "cpu").

    Returns:
        Active device string ("mps" or "cpu").
    """
    requested = preferred.strip().lower()
    if requested == "cpu":
        return "cpu"

    try:
        import torch

        mps_available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
    except ImportError:
        return "cpu"

    if requested in ("auto", "mps") and mps_available:
        return "mps"

    return "cpu"


@dataclass(frozen=True)
class DetectionConfig:
    """Configuration for YOLO11 object detection."""

    model_name: str = "yolo11n.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    imgsz: int = 640
    target_classes: Optional[tuple[int, ...]] = (0,)  # 0 = person in COCO


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for BoT-SORT short-horizon tracking."""

    tracker_config: str = "botsort.yaml"
    track_buffer: int = 30
    match_threshold: float = 0.8


@dataclass(frozen=True)
class ReIDConfig:
    """Configuration for OSNet appearance embedding extraction."""

    model_name: str = "osnet_x0_25"
    embedding_dim: int = 512
    input_height: int = 256
    input_width: int = 128


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for FAISS + SQLite long-horizon identity memory."""

    db_path: Path = LOGS_DIR / "argus_memory.db"
    similarity_threshold: float = 0.65
    max_gallery_embeddings: int = 10


@dataclass(frozen=True)
class AppConfig:
    """Top-level immutable configuration container for ARGUS-X."""

    device: str = field(
        default_factory=lambda: resolve_device(os.getenv("ARGUS_DEVICE", "auto"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("ARGUS_LOG_LEVEL", "INFO").upper()
    )
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def get_config() -> AppConfig:
    """Return initialized application configuration."""
    return AppConfig()
