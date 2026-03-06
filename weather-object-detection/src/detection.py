from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO


@dataclass(frozen=True)
class DetectConfig:
    conf: float = 0.25
    iou: float = 0.7
    imgsz: int = 640
    max_det: int = 300


def run_detection(
    model: YOLO,
    image: Image.Image,
    cfg: DetectConfig = DetectConfig(),
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Returns:
      annotated_image: PIL RGB image with boxes drawn
      detections: list of dicts with class, confidence, box coords
    """
    img_np = np.array(image.convert("RGB"))

    results = model.predict(
        source=img_np,
        conf=float(cfg.conf),
        iou=float(cfg.iou),
        imgsz=int(cfg.imgsz),
        max_det=int(cfg.max_det),
        verbose=False,
    )
    r0 = results[0]

    # Ultralytics plot() returns BGR uint8 array
    plotted_bgr = r0.plot()
    plotted_rgb = plotted_bgr[..., ::-1].copy()
    annotated = Image.fromarray(plotted_rgb, mode="RGB")

    dets: List[Dict[str, Any]] = []
    names = getattr(r0, "names", None) or getattr(model, "names", {})

    if r0.boxes is None or len(r0.boxes) == 0:
        return annotated, dets

    boxes = r0.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
        dets.append(
            {
                "class_id": int(k),
                "class_name": str(names.get(int(k), int(k))),
                "confidence": float(c),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }
        )

    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return annotated, dets

