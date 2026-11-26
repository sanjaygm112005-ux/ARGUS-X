"""Person re-identification feature extractor using OSNet for ARGUS-X.

Extracts L2-normalized 512-dim appearance embeddings from person crops.
This module is intentionally independent from the tracker — embeddings are
produced but *not* matched or merged here (global identity recovery is a
separate milestone).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from config import BASE_DIR, ReIDConfig, resolve_device
from reidentification.osnet import osnet_x0_25, osnet_x0_5, osnet_x0_75, osnet_x1_0

logger = logging.getLogger(__name__)

# ImageNet normalization used by OSNet pretrained weights.
_PIXEL_MEAN = [0.485, 0.456, 0.406]
_PIXEL_STD = [0.229, 0.224, 0.225]

# Minimum crop dimensions (px); smaller crops produce unreliable embeddings.
_MIN_CROP_HEIGHT = 32
_MIN_CROP_WIDTH = 16

# OSNet factory lookup.
_MODEL_BUILDERS = {
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
}

# Default weight file name and HuggingFace download URL per architecture.
_WEIGHT_INFO: dict[str, dict[str, str]] = {
    "osnet_x0_25": {
        "filename": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    },
    "osnet_x0_5": {
        "filename": "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    },
    "osnet_x0_75": {
        "filename": "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    },
    "osnet_x1_0": {
        "filename": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    },
}


class PersonFeatureExtractor:
    """Extract L2-normalized appearance embeddings from person crops using OSNet.

    Loads pretrained OSNet weights once and supports single or batch extraction
    from bounding-box crops (numpy BGR frames) or PIL images.

    Note: This module produces raw appearance vectors. It does NOT perform
    identity matching, gallery management, or cross-camera identity recovery.
    """

    def __init__(
        self,
        model_name: str = ReIDConfig.model_name,
        weights_path: Optional[Union[str, Path]] = None,
        input_height: int = ReIDConfig.input_height,
        input_width: int = ReIDConfig.input_width,
        device: str = "auto",
    ) -> None:
        """Initialize OSNet and preprocessing pipeline.

        Args:
            model_name: OSNet variant name (e.g. "osnet_x0_25").
            weights_path: Optional explicit path to pretrained ``.pth`` file.
                          If ``None``, the standard weight file is searched in
                          the workspace ``weights/`` directory.
            input_height: Resize height for OSNet input (default 256).
            input_width: Resize width for OSNet input (default 128).
            device: Preferred compute device ("auto", "mps", or "cpu").
        """
        if model_name not in _MODEL_BUILDERS:
            raise ValueError(
                f"Unknown OSNet variant '{model_name}'. "
                f"Choose from: {list(_MODEL_BUILDERS.keys())}."
            )

        self.model_name = model_name
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.device_str = resolve_device(device)
        self.device = torch.device(self.device_str)

        resolved_weights = self._resolve_weights(model_name, weights_path)
        builder = _MODEL_BUILDERS[model_name]

        try:
            self.model = builder(num_classes=1, pretrained_path=str(resolved_weights))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build OSNet '{model_name}' with weights '{resolved_weights}': {exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim: int = self.model.feature_dim

        self.preprocess = T.Compose([
            T.Resize((self.input_height, self.input_width)),
            T.ToTensor(),
            T.Normalize(mean=_PIXEL_MEAN, std=_PIXEL_STD),
        ])

        logger.info(
            "PersonFeatureExtractor ready | model=%s | embedding_dim=%d | device=%s | weights=%s",
            model_name,
            self.embedding_dim,
            self.device_str,
            resolved_weights.name,
        )

    @staticmethod
    def _resolve_weights(model_name: str, weights_path: Optional[Union[str, Path]]) -> Path:
        if weights_path is not None:
            p = Path(weights_path)
            if p.is_file():
                return p.resolve()
            raise FileNotFoundError(f"Specified weight file not found: '{weights_path}'.")

        info = _WEIGHT_INFO.get(model_name)
        if info is None:
            raise FileNotFoundError(
                f"No default weight info for '{model_name}'. Pass weights_path explicitly."
            )
        filename = info["filename"]

        for search_dir in [BASE_DIR / "weights", BASE_DIR]:
            candidate = search_dir / filename
            if candidate.is_file():
                return candidate.resolve()

        raise FileNotFoundError(
            f"OSNet weight file '{filename}' not found in '{BASE_DIR}/weights/' or '{BASE_DIR}/'. "
            f"Download it from: {info['url']}"
        )

    @staticmethod
    def crop_person(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop a person region from a BGR frame using a bounding box.

        Args:
            frame: OpenCV BGR image (H, W, 3).
            bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            Cropped BGR image. Raises ValueError if the crop is too small.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a non-empty numpy.ndarray.")

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h < _MIN_CROP_HEIGHT or crop_w < _MIN_CROP_WIDTH:
            raise ValueError(
                f"Crop too small ({crop_w}x{crop_h}); minimum is {_MIN_CROP_WIDTH}x{_MIN_CROP_HEIGHT}."
            )

        return frame[y1:y2, x1:x2].copy()

    def _prepare_tensor(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Convert a list of BGR numpy crops to a preprocessed batch tensor."""
        images: list[torch.Tensor] = []
        for crop in crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            images.append(self.preprocess(pil_img))
        return torch.stack(images, dim=0).to(self.device)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """Extract a single L2-normalized embedding from a BGR person crop.

        Args:
            crop: BGR person crop as numpy.ndarray (H, W, 3).

        Returns:
            1-D numpy array of shape (embedding_dim,), L2-normalized.
        """
        if crop is None or not isinstance(crop, np.ndarray):
            raise ValueError("crop must be a non-empty numpy.ndarray.")
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"Invalid crop shape {crop.shape}; expected (H, W, 3) BGR image.")
        if crop.shape[0] < _MIN_CROP_HEIGHT or crop.shape[1] < _MIN_CROP_WIDTH:
            raise ValueError(
                f"Crop too small ({crop.shape[1]}x{crop.shape[0]}); "
                f"minimum is {_MIN_CROP_WIDTH}x{_MIN_CROP_HEIGHT}."
            )

        batch = self._prepare_tensor([crop])
        with torch.no_grad():
            features = self.model(batch)
        embedding = F.normalize(features, p=2, dim=1).squeeze(0).cpu().numpy()
        return embedding

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Extract L2-normalized embeddings from multiple BGR person crops.

        Args:
            crops: Sequence of BGR person crops, each as numpy.ndarray (H, W, 3).

        Returns:
            2-D numpy array of shape (N, embedding_dim), each row L2-normalized.
        """
        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        valid_crops: list[np.ndarray] = []
        for idx, crop in enumerate(crops):
            if crop is None or not isinstance(crop, np.ndarray):
                raise ValueError(f"Crop at index {idx} must be a non-empty numpy.ndarray.")
            if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                raise ValueError(f"Invalid crop shape {crop.shape} at index {idx}; expected (H, W, 3).")
            if crop.shape[0] < _MIN_CROP_HEIGHT or crop.shape[1] < _MIN_CROP_WIDTH:
                raise ValueError(
                    f"Crop at index {idx} too small ({crop.shape[1]}x{crop.shape[0]}); "
                    f"minimum is {_MIN_CROP_WIDTH}x{_MIN_CROP_HEIGHT}."
                )
            valid_crops.append(crop)

        batch = self._prepare_tensor(valid_crops)
        with torch.no_grad():
            features = self.model(batch)
        embeddings = F.normalize(features, p=2, dim=1).cpu().numpy()
        return embeddings

    def extract_from_frame(
        self,
        frame: np.ndarray,
        bboxes: Sequence[tuple[float, float, float, float]],
    ) -> np.ndarray:
        """Crop persons from a full frame using bounding boxes and extract embeddings.

        Args:
            frame: Full BGR frame as numpy.ndarray (H, W, 3).
            bboxes: Sequence of bounding boxes, each as (x1, y1, x2, y2).

        Returns:
            2-D numpy array of shape (N, embedding_dim), each row L2-normalized.
            Crops that are too small are silently skipped (returned array may
            have fewer rows than input bboxes).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Frame must be a non-empty numpy.ndarray.")
        if not bboxes:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        crops: list[np.ndarray] = []
        for bbox in bboxes:
            try:
                crop = self.crop_person(frame, bbox)
                crops.append(crop)
            except ValueError:
                logger.debug("Skipping too-small crop for bbox %s", bbox)

        if not crops:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        return self.extract_batch(crops)
