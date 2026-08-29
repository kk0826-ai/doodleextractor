import io
import os
import tempfile
import zipfile
import cv2
import numpy as np
import streamlit as st
import vtracer

st.set_page_config(page_title="Alphabet SVG Slicer", layout="wide")

st.title("🔤 Alphabet SVG Slicer & Vectorizer")
st.write("Upload an A–Z grid sheet to slice and convert each letter into downloadable SVGs.")

uploaded_file = st.file_uploader("Upload Grid Sheet (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image Grid", use_container_width=True)

    with st.sidebar:
        st.header("Detection Settings")
        threshold_val = st.slider("Threshold Value", 50, 255, 230)
        padding = st.slider("Padding around letters (px)", 0, 50, 10)
        min_size = st.slider("Minimum letter size (px)", 10, 200, 30)

    if st.button("Slice & Convert to SVG", type="primary"):
        with st.spinner("Processing letters and converting to SVG..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounding_boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= min_size and h >= min_size:
                    bounding_boxes.append((x, y, w, h))

            # Sort top-to-bottom, left-to-right
            bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 100, b[0]))[:26]

            if not bounding_boxes:
                st.error("No letter shapes detected. Try lowering the threshold slider.")
            else:
                zip_buffer = io.BytesIO()

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        cols = st.columns(6)

                        for i, (x, y, w, h) in enumerate(bounding_boxes):
                            letter_char = chr(65 + i) if i < 26 else f"letter_{i+1}"

                            # 1. Safe Crop
                            pad_y1, pad_y2 = max(0, y - padding), min(img.shape[0], y + h + padding)
                            pad_x1, pad_x2 = max(0, x - padding), min(img.shape[1], x + w + padding)
                            crop = img[pad_y1:pad_y2, pad_x1:pad_x2]

                            # 2. Fast smoothing (Prevents OOM/Timeout crashes)
                            smoothed = cv2.GaussianBlur(crop, (3, 3), 0)

                            # 3. Canvas normalization (400x400 px)
                            canvas_size = 400
                            canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
                            crop_h, crop_w = crop.shape[:2]
                            scale = (canvas_size - 40) / max(crop_h, crop_w)
                            new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))

                            resized = cv2.resize(smoothed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            offset_x, offset_y = (canvas_size - new_w) // 2, (canvas_size - new_h) // 2
                            canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

                            # 4. Save temporary PNG
                            png_path = os.path.join(tmp_dir, f"{letter_char}.png")
                            svg_path = os.path.join(tmp_dir, f"{letter_char}.svg")
                            cv2.imwrite(png_path, canvas)

                            # 5. Stable Vectorization Call
                            try:
                                vtracer.convert_image_to_svg_py(
                                    png_path,
                                    svg_path,
                                    colormode="color",
                                    filter_speckle=4,
                                    color_precision=6,
                                    layer_difference=16,
                                    corner_threshold=60
                                )
                                with open(svg_path, "r", encoding="utf-8") as f:
                                    svg_code = f.read()

                                zip_file.writestr(f"{letter_char}.svg", svg_code)

                                with cols[i % 6]:
                                    st.write(f"**Letter {letter_char}**")
                                    st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    st.download_button(
                                        label=f"{letter_char}.svg",
                                        data=svg_code,
                                        file_name=f"{letter_char}.svg",
                                        mime="image/svg+xml",
                                        key=f"dl_{i}"
                                    )
                            except Exception as e:
                                st.error(f"Failed to vectorize letter {letter_char}: {e}")

                st.divider()
                st.download_button(
                    label="📦 Download All 26 SVGs as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="alphabet_svgs.zip",
                    mime="application/zip",
                    type="primary"
                )
