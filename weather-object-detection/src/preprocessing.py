from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class PreprocessConfig:
    resize_to_match_rgb: bool = True
    thermal_normalize: bool = True
    thermal_autocontrast: bool = True
    rgb_autocontrast: bool = False
    max_side: Optional[int] = 1280  # keep UI snappy; preserves aspect ratio


def _limit_max_side(img: Image.Image, max_side: Optional[int]) -> Image.Image:
    if not max_side:
        return img
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.BILINEAR)


def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


def ensure_grayscale(img: Image.Image) -> Image.Image:
    if img.mode in ("L", "I;16", "I"):
        return img.convert("L")
    return img.convert("L")


def preprocess_pair(
    rgb: Image.Image,
    thermal: Image.Image,
    cfg: PreprocessConfig = PreprocessConfig(),
) -> Tuple[Image.Image, Image.Image]:
    rgb = ensure_rgb(rgb)
    thermal = ensure_grayscale(thermal)

    rgb = _limit_max_side(rgb, cfg.max_side)
    thermal = _limit_max_side(thermal, cfg.max_side)

    if cfg.rgb_autocontrast:
        rgb = ImageOps.autocontrast(rgb)

    if cfg.resize_to_match_rgb:
        thermal = thermal.resize(rgb.size, Image.Resampling.BILINEAR)

    if cfg.thermal_autocontrast:
        thermal = ImageOps.autocontrast(thermal)

    if cfg.thermal_normalize:
        thermal_np = np.array(thermal).astype(np.float32)
        lo = float(np.percentile(thermal_np, 1.0))
        hi = float(np.percentile(thermal_np, 99.0))
        if hi > lo:
            thermal_np = np.clip((thermal_np - lo) / (hi - lo), 0.0, 1.0)
        else:
            thermal_np = np.zeros_like(thermal_np)
        thermal = Image.fromarray((thermal_np * 255.0).astype(np.uint8), mode="L")

    return rgb, thermal

