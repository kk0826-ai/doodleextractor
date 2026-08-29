import io
import os
import tempfile
import zipfile
import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Alphabet PNG Slicer", layout="wide")

st.title("🔤 Alphabet PNG Slicer")
st.write("Upload an A–Z grid sheet to automatically slice, center, and extract each letter as an individual PNG.")

uploaded_file = st.file_uploader("Upload Grid Sheet (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Decode image file bytes to OpenCV array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image Grid", use_container_width=True)

    with st.sidebar:
        st.header("Detection & Canvas Settings")
        threshold_val = st.slider("Threshold Value", 50, 255, 230)
        padding = st.slider("Padding around letters (px)", 0, 50, 10)
        min_size = st.slider("Minimum letter size (px)", 10, 200, 30)
        canvas_size = st.number_input("Output Canvas Size (px)", value=500, step=50)

    if st.button("Slice & Extract PNGs", type="primary"):
        with st.spinner("Slicing and processing letters..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            bounding_boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if w >= min_size and h >= min_size:
                    bounding_boxes.append((x, y, w, h))

            # Sort top-to-bottom, left-to-right (A–Z)
            bounding_boxes = sorted(bounding_boxes, key=lambda b: (b[1] // 100, b[0]))[:26]

            if not bounding_boxes:
                st.error("No letter shapes detected. Try adjusting the threshold slider in the sidebar.")
            else:
                st.success(f"Successfully extracted {len(bounding_boxes)} letters!")
                zip_buffer = io.BytesIO()

                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        cols = st.columns(6)

                        for i, (x, y, w, h) in enumerate(bounding_boxes):
                            letter_char = chr(65 + i) if i < 26 else f"letter_{i+1}"

                            # 1. Crop region with safety margins
                            pad_y1, pad_y2 = max(0, y - padding), min(img.shape[0], y + h + padding)
                            pad_x1, pad_x2 = max(0, x - padding), min(img.shape[1], x + w + padding)
                            crop = img[pad_y1:pad_y2, pad_x1:pad_x2]

                            # 2. Create a uniform square white canvas
                            canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

                            # 3. Calculate scaling to fit within canvas while maintaining aspect ratio
                            crop_h, crop_w = crop.shape[:2]
                            scale = (canvas_size - 40) / max(crop_h, crop_w)
                            new_w, new_h = max(1, int(crop_w * scale)), max(1, int(crop_h * scale))

                            # 4. Upscale/Downscale & Center letter on canvas
                            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            offset_x = (canvas_size - new_w) // 2
                            offset_y = (canvas_size - new_h) // 2
                            canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized

                            # 5. Encode PNG
                            png_filename = f"{letter_char}.png"
                            _, buffer = cv2.imencode(".png", canvas)
                            png_bytes = buffer.tobytes()

                            zip_file.writestr(png_filename, png_bytes)

                            # Display UI grid element
                            with cols[i % 6]:
                                st.write(f"**Letter {letter_char}**")
                                st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), use_container_width=True)
                                st.download_button(
                                    label=f"Download {letter_char}.png",
                                    data=png_bytes,
                                    file_name=png_filename,
                                    mime="image/png",
                                    key=f"dl_{i}"
                                )

                st.divider()
                st.download_button(
                    label="📦 Download All 26 PNGs as ZIP",
                    data=zip_buffer.getvalue(),
                    file_name="alphabet_pngs.zip",
                    mime="application/zip",
                    type="primary"
                )
