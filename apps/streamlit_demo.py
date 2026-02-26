from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO


st.set_page_config(page_title="PPE Helmet & Vest Detection", layout="wide")
st.title("PPE Helmet & Vest Detection Demo")

weights_path = st.sidebar.text_input("Weights Path", "artifacts/best.pt")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.25, 0.05)

if not Path(weights_path).exists():
    st.warning(f"Weight file not found: {weights_path}")
    st.stop()

model = YOLO(weights_path)
upload = st.file_uploader("Upload image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if upload is not None:
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        temp_path = tmp.name

    if suffix in [".jpg", ".jpeg", ".png"]:
        result = model.predict(source=temp_path, conf=conf, verbose=False)[0]
        plotted = result.plot()
        st.image(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
    else:
        cap = cv2.VideoCapture(temp_path)
        out_file = Path("outputs") / "streamlit_demo_result.mp4"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        with st.spinner("Running video inference ..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = model.predict(source=frame, conf=conf, verbose=False)[0]
                writer.write(result.plot())

        cap.release()
        writer.release()
        st.video(str(out_file))
        st.success(f"Saved result: {out_file}")

