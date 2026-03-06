from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO


@dataclass(frozen=True)
class YoloConfig:
    weights: str = "yolov8n.pt"  # downloaded automatically by ultralytics
    device: Optional[str] = None  # e.g. "cpu", "cuda:0"


def load_yolo(cfg: YoloConfig = YoloConfig()) -> YOLO:
    model = YOLO(cfg.weights)
    if cfg.device:
        model.to(cfg.device)
    return model

