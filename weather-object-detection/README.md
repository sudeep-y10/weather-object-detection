# RGB + Thermal Fusion Object Detection (YOLOv8 + Streamlit)

Streamlit web app for object detection using **YOLOv8** on a **fused RGB + thermal** image.

## What this project does
- Upload an RGB image and a thermal image
- Preprocess + resize them (thermal is normalized)
- Convert thermal to a pseudocolor map and alpha-blend into RGB (3-channel fused image)
- Run YOLOv8 object detection
- Show annotated image + a table of detections (class, confidence, box coords)

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

On first run, Ultralytics will automatically download the selected YOLO weights (e.g. `yolov8n.pt`).

## Notes
- This repo uses **fusion for inference with pretrained YOLO weights** (no retraining). For best thermal-aware performance you’d typically train a multimodal model; this implementation is meant to be a practical baseline.

