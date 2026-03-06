from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np
from PIL import Image


FusionMethod = Literal["alpha_blend_colormap"]


@dataclass(frozen=True)
class FusionConfig:
    method: FusionMethod = "alpha_blend_colormap"
    alpha: float = 0.35  # how much thermal to blend in
    colormap: Literal[
        "inferno",
        "magma",
        "plasma",
        "jet",
        "turbo",
        "hot",
        "bone",
        "viridis",
    ] = "inferno"


_COLORMAP_MAP = {
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
    "plasma": cv2.COLORMAP_PLASMA,
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    # OpenCV doesn't ship viridis everywhere; fall back if missing.
    "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_TURBO),
}


def fuse_rgb_thermal(
    rgb: Image.Image,
    thermal_gray: Image.Image,
    cfg: FusionConfig = FusionConfig(),
) -> Tuple[Image.Image, Image.Image]:
    """
    Returns:
      fused_rgb: RGB image to feed YOLO (3-channel).
      thermal_color: RGB pseudocolor image (for UI/debug).
    """
    if cfg.method != "alpha_blend_colormap":
        raise ValueError(f"Unsupported fusion method: {cfg.method}")

    alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
    rgb_np = np.array(rgb, dtype=np.uint8)  # RGB
    th_np = np.array(thermal_gray, dtype=np.uint8)  # HxW

    # Pseudocolor in BGR (OpenCV default) then convert to RGB.
    cmap = _COLORMAP_MAP.get(cfg.colormap, cv2.COLORMAP_INFERNO)
    th_color_bgr = cv2.applyColorMap(th_np, cmap)
    th_color_rgb = cv2.cvtColor(th_color_bgr, cv2.COLOR_BGR2RGB)

    fused = (rgb_np.astype(np.float32) * (1.0 - alpha) + th_color_rgb.astype(np.float32) * alpha).clip(
        0, 255
    )
    fused = fused.astype(np.uint8)

    return Image.fromarray(fused, mode="RGB"), Image.fromarray(th_color_rgb, mode="RGB")

