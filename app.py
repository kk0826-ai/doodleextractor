import io
import os
import tempfile
import zipfile
import cv2
import numpy as np
import streamlit as st
import vtracer

st.set_page_config(page_title="Alphabet SVG Slicer & Vectorizer", layout="wide")

st.title("🔤 Alphabet SVG Slicer & Vectorizer")
st.write("Upload an A–Z grid sheet to slice, smooth, normalize, and convert each letter into clean, downloadable SVGs.")

uploaded_file = st.file_uploader("Upload Grid Sheet (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read uploaded file as OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image Grid", use_container_width=True)

    with st.sidebar:
        st.header("Detection & Canvas Settings")
        threshold_val = st.slider("Threshold Value", 50, 255, 230)
        padding = st.slider("Padding around letters (px)", 0, 50, 10)
        min_size = st.slider("Minimum letter size (px)", 10, 200, 30)
        canvas_size = st.number_input("Output Canvas Size (px)", value=500, step=50)

    if st.button("Slice & Convert to SVG", type="primary"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= min_size and h >= min_size:
                bounding_boxes.append((x, y, w, h))

        # Sort contours top-to-bottom, then left-to-right to maintain A-Z ordering
        bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 120, b[0]))
        bounding_boxes = bounding_boxes[:26]

        if not bounding_boxes:
            st.error("No letter shapes detected. Try adjusting the threshold slider in the sidebar.")
        else:
            st.success(f"Successfully detected {len(bounding_boxes)} letters!")

            zip_buffer = io.BytesIO()

            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    cols = st.columns(6)

                    for i, (x, y, w, h) in enumerate(bounding_boxes):
                        letter_char = chr(65 + i) if i < 26 else f"letter_{i+1}"

                        # 1. Crop raw letter region with safety padding
                        pad_y1 = max(0, y - padding)
                        pad_y2 = min(img.shape[0], y + h + padding)
                        pad_x1 = max(0, x - padding)
                        pad_x2 = min(img.shape[1], x + w + padding)
                        crop = img[pad_y1:pad_y2, pad_x1:pad_x2]

                        # 2. Smooth pixel noise to prevent jagged vector paths
                        smoothed = cv2.bilateralFilter(crop, d=9, sigmaColor=75, sigmaSpace=75)

                        # 3. Normalize aspect ratio inside a square white canvas
                        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

                        crop_h, crop_w = crop.shape[:2]
                        scale = (canvas_size - 40) / max(crop_h, crop_w)
                        new_w, new_h = int(crop_w * scale), int(crop_h * scale)

                        # High-quality upscale & centering
                        resized = cv2.resize(smoothed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                        offset_x = (canvas_size - new_w) // 2
                        offset_y = (canvas_size - new_h) // 2
                        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

                        # 4. Save smoothed PNG & trace into SVG
                        png_path = os.path.join(tmp_dir, f"{letter_char}.png")
                        svg_path = os.path.join(tmp_dir, f"{letter_char}.svg")

                        cv2.imwrite(png_path, canvas)

                        # Convert raster to SVG with tuned curve-smoothing parameters
                        vtracer.convert_image_to_svg_py(
                            png_path, 
                            svg_path,
                            colormode="color",
                            corner_threshold=60,
                            filter_speckle=8,
                            color_precision=6,
                            layer_difference=16,
                            path_precision=8
                        )

                        with open(svg_path, "r", encoding="utf-8") as f:
                            svg_code = f.read()

                        zip_file.writestr(f"{letter_char}.svg", svg_code)

                        with cols[i % 6]:
                            st.write(f"**Letter {letter_char}**")
                            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                            st.download_button(
                                label=f"Download {letter_char}.svg",
                                data=svg_code,
                                file_name=f"{letter_char}.svg",
                                mime="image/svg+xml",
                                key=f"dl_{i}"
                            )

            st.divider()
            st.download_button(
                label="📦 Download All 26 SVGs as ZIP",
                data=zip_buffer.getvalue(),
                file_name="alphabet_svgs.zip",
                mime="application/zip",
                type="primary"
            )
