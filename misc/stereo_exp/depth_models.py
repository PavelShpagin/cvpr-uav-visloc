"""Depth/height model registry for stereo experiments.

This module wraps a few off-the-shelf monocular depth estimators behind a
consistent interface so benchmarking scripts can iterate over them easily.

All models should expose:
    - ``name``: human readable identifier (unique)
    - ``metadata``: dict with model type, expected input size, etc.
    - ``predict(image: np.ndarray) -> np.ndarray`` returning depth in float32

The loaders are defensive: if optional dependencies are missing, they raise a
clear ``RuntimeError`` telling the caller how to install the requirement. This
keeps the benchmarking script honest (per user directive: never hallucinate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


try:
    import torch
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - torch should be available already
    raise RuntimeError(
        "PyTorch is required for stereo experiments. Install via pip install torch"
    ) from exc


try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - pillow is a hard dep elsewhere
    raise RuntimeError("Pillow is required (pip install pillow)") from exc


def _as_numpy(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=-1)
        return image
    raise TypeError(f"Unsupported image type {type(image)}")


class DepthModel:
    """Abstract base class for depth model wrappers."""

    name: str

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def predict(self, image: np.ndarray | Image.Image) -> np.ndarray:
        raise NotImplementedError

    @property
    def metadata(self) -> Dict[str, str]:
        return {}


class MiDaSModel(DepthModel):
    """Wrapper for MiDaS depth models via torch.hub."""

    def __init__(self, model_type: str, device: str = "cuda") -> None:
        super().__init__(device=device)
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type)
            self.model.to(self.device)
            self.model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        except Exception as exc:  # pragma: no cover - network errors
            raise RuntimeError(
                "Failed to load MiDaS model. Ensure internet access or pre-download "
                "weights."
            ) from exc

        if model_type in {"DPT_Large", "DPT_Hybrid"}:
            self._transform = transforms.dpt_transform
        else:
            self._transform = transforms.small_transform

        self.name = f"midas_{model_type.lower()}"
        self._model_type = model_type

    def predict(self, image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            pil_img = Image.fromarray(_as_numpy(image))

        arr = np.asarray(pil_img)
        input_batch = self._transform(arr).to(self.device)
        target_hw = arr.shape[:2]

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=target_hw,
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.detach().cpu().numpy().astype(np.float32)

    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "family": "MiDaS",
            "model_type": self._model_type,
        }


class ZoeDepthModel(DepthModel):
    """Wrapper for ZoeDepth models via torch.hub."""

    def __init__(self, variant: str = "zoedepth_nk", device: str = "cuda") -> None:
        super().__init__(device=device)

        try:
            self.model = torch.hub.load("isl-org/ZoeDepth", variant, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to load ZoeDepth model. Install git+https://github.com/isl-org/ZoeDepth "
                "or ensure torch>=1.10 with CUDA."
            ) from exc

        self.name = variant
        self._variant = variant

    def predict(self, image: np.ndarray | Image.Image) -> np.ndarray:
        rgb = _as_numpy(image)
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
        rgb_t = rgb_t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth = self.model.infer(rgb_t)

        depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
        return depth

    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "family": "ZoeDepth",
            "variant": self._variant,
        }


class DepthAnythingModel(DepthModel):
    """Wrapper for Depth Anything V2 models if package is installed."""

    _MODEL_MAP = {
        "depth_anything_v2_small": "small",
        "depth_anything_v2_base": "base",
        "depth_anything_v2_large": "large",
    }

    def __init__(self, preset: str, device: str = "cuda") -> None:
        super().__init__(device=device)

        try:
            from depth_anything_v2 import DepthAnythingV2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "depth_anything_v2 package missing. Install via pip install depth-anything-v2"
            ) from exc

        if preset not in self._MODEL_MAP:
            raise ValueError(f"Unsupported DepthAnything preset: {preset}")

        model_type = self._MODEL_MAP[preset]
        self.model = DepthAnythingV2.from_pretrained(model_type)
        self.model.to(self.device)
        self.model.eval()
        self.name = preset
        self._preset = preset

    def predict(self, image: np.ndarray | Image.Image) -> np.ndarray:
        rgb = _as_numpy(image)
        rgb_t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        rgb_t = rgb_t.to(self.device) / 255.0

        with torch.no_grad():
            depth = self.model(rgb_t)

        depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
        return depth

    @property
    def metadata(self) -> Dict[str, str]:
        return {
            "family": "DepthAnythingV2",
            "preset": self._preset,
        }


def build_depth_model(identifier: str, device: str = "cuda") -> DepthModel:
    """Factory to create depth models by identifier."""

    identifier = identifier.lower()

    if identifier in {"midas_dpt_hybrid", "midas_dpt_large", "midas_small"}:
        model_map = {
            "midas_dpt_hybrid": "DPT_Hybrid",
            "midas_dpt_large": "DPT_Large",
            "midas_small": "MiDaS_small",
        }
        return MiDaSModel(model_map[identifier], device=device)

    if identifier.startswith("zoedepth"):
        return ZoeDepthModel(identifier, device=device)

    if identifier in DepthAnythingModel._MODEL_MAP:
        return DepthAnythingModel(identifier, device=device)

    raise ValueError(f"Unknown depth model identifier: {identifier}")


def available_models() -> Dict[str, Dict[str, str]]:
    """Return metadata for supported models (without instantiating)."""

    return {
        "midas_dpt_hybrid": {"family": "MiDaS", "display_name": "MiDaS DPT-Hybrid"},
        "midas_dpt_large": {"family": "MiDaS", "display_name": "MiDaS DPT-Large"},
        "midas_small": {"family": "MiDaS", "display_name": "MiDaS small"},
        "zoedepth_nk": {"family": "ZoeDepth", "display_name": "ZoeDepth NK"},
        "depth_anything_v2_small": {"family": "DepthAnythingV2", "display_name": "Depth Anything V2 Small"},
        "depth_anything_v2_base": {"family": "DepthAnythingV2", "display_name": "Depth Anything V2 Base"},
        "depth_anything_v2_large": {"family": "DepthAnythingV2", "display_name": "Depth Anything V2 Large"},
    }


def warmup_model(model: DepthModel, image: np.ndarray) -> float:
    """Run a single forward pass to allocate kernels and return warmup latency."""

    start = time.perf_counter()
    _ = model.predict(image)
    torch.cuda.synchronize(model.device) if model.device.type == "cuda" else None
    return time.perf_counter() - start


