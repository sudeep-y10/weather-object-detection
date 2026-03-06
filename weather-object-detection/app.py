from __future__ import annotations

import streamlit as st
from PIL import Image

from src.detection import DetectConfig, run_detection
from src.fusion import FusionConfig, fuse_rgb_thermal
from src.preprocessing import PreprocessConfig, preprocess_pair
from src.yolo_loader import YoloConfig, load_yolo


st.set_page_config(page_title="RGB + Thermal YOLOv8 Object Detection", layout="wide")


@st.cache_resource
def _get_model(weights: str, device: str | None):
    return load_yolo(YoloConfig(weights=weights, device=device or None))


def main():
    st.title("Real-time Object Detection with YOLOv8 (RGB + Thermal Fusion)")
    st.caption("Upload an RGB image and a thermal image. The app fuses them and runs YOLOv8 detection.")

    with st.sidebar:
        st.subheader("Model")
        weights = st.selectbox("YOLO weights", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
        device = st.text_input("Device (optional)", value="", help='Examples: "cpu", "cuda:0"')

        st.subheader("Detection")
        conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU threshold", 0.0, 1.0, 0.70, 0.01)
        imgsz = st.select_slider("Image size (imgsz)", options=[320, 416, 512, 640, 768, 896, 1024], value=640)
        max_det = st.number_input("Max detections", min_value=1, max_value=1000, value=300, step=10)

        st.subheader("Fusion")
        alpha = st.slider("Thermal blend alpha", 0.0, 1.0, 0.35, 0.01)
        colormap = st.selectbox(
            "Thermal colormap",
            ["inferno", "magma", "plasma", "turbo", "jet", "hot", "bone", "viridis"],
            index=0,
        )

        st.subheader("Preprocessing")
        max_side = st.select_slider("Max side (resize for speed)", options=[640, 960, 1280, 1600, 2048], value=1280)
        thermal_autocontrast = st.checkbox("Thermal autocontrast", value=True)
        thermal_normalize = st.checkbox("Thermal robust normalize", value=True)

    col1, col2 = st.columns(2)
    with col1:
        rgb_file = st.file_uploader("Upload RGB image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    with col2:
        thermal_file = st.file_uploader("Upload Thermal image", type=["png", "jpg", "jpeg", "bmp", "webp"])

    if not rgb_file or not thermal_file:
        st.info("Upload both RGB and thermal images to run detection.")
        return

    try:
        rgb_img = Image.open(rgb_file)
        thermal_img = Image.open(thermal_file)
    except Exception as e:
        st.error(f"Failed to read images: {e}")
        return

    pp_cfg = PreprocessConfig(
        resize_to_match_rgb=True,
        thermal_normalize=thermal_normalize,
        thermal_autocontrast=thermal_autocontrast,
        rgb_autocontrast=False,
        max_side=int(max_side),
    )
    rgb_pp, th_pp = preprocess_pair(rgb_img, thermal_img, pp_cfg)

    fusion_cfg = FusionConfig(alpha=float(alpha), colormap=colormap)
    fused, thermal_color = fuse_rgb_thermal(rgb_pp, th_pp, fusion_cfg)

    preview_c1, preview_c2, preview_c3 = st.columns(3)
    with preview_c1:
        st.image(rgb_pp, caption="RGB (preprocessed)", use_container_width=True)
    with preview_c2:
        st.image(thermal_color, caption="Thermal (pseudocolor)", use_container_width=True)
    with preview_c3:
        st.image(fused, caption="Fused (fed to YOLO)", use_container_width=True)

    run = st.button("Run detection", type="primary", use_container_width=True)
    if not run:
        return

    with st.spinner("Loading model and running inference..."):
        model = _get_model(weights=weights, device=device.strip() or None)
        det_cfg = DetectConfig(conf=float(conf), iou=float(iou), imgsz=int(imgsz), max_det=int(max_det))
        annotated, dets = run_detection(model, fused, det_cfg)

    out_c1, out_c2 = st.columns([2, 1])
    with out_c1:
        st.image(annotated, caption="Detections", use_container_width=True)
    with out_c2:
        st.subheader("Results")
        st.write(f"Detections: **{len(dets)}**")
        if dets:
            st.dataframe(
                [
                    {
                        "class": d["class_name"],
                        "conf": round(d["confidence"], 3),
                        "x1": round(d["x1"], 1),
                        "y1": round(d["y1"], 1),
                        "x2": round(d["x2"], 1),
                        "y2": round(d["y2"], 1),
                    }
                    for d in dets
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No objects detected at the current thresholds.")


if __name__ == "__main__":
    main()

