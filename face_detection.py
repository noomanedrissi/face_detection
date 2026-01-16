# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 07:41:38 2026

@author: noomane.drissi
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 1. Instructions for the user
st.title("Face Detection by NDRISSI using Improved Viola-Jones Face Detector")
st.markdown("""
### Instructions:
1. **Upload** an image (JPG, PNG, or JPEG).
2. **Adjust** the parameters in the sidebar to fine-tune detection.
3. **Choose** your preferred rectangle color.
4. **Download** the processed image using the button below the result.
""")

# Sidebar for parameters
st.sidebar.header("Detection Settings")

# 4 & 5. Adjust scaleFactor and minNeighbors
scale_factor = st.sidebar.slider("Scale Factor", 1.01, 2.0, 1.1, 0.05, help="Parameter specifying how much the image size is reduced at each image scale.")
min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, help="Parameter specifying how many neighbors each candidate rectangle should have to retain it.")

# 3. Choose rectangle color
rect_color_hex = st.sidebar.color_picker("Pick a rectangle color", "#00FF00")
# Convert hex to BGR (OpenCV format)
rect_color_bgr = tuple(int(rect_color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rect_color_bgr, 2)

    # Convert back to RGB for Streamlit display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Processed Image', use_column_width=True)
    st.success(f"Detected {len(faces)} face(s).")

    # 2. Save and Download feature
    # Use cv2.imencode to prepare the image for download without saving to disk first
    is_success, buffer = cv2.imencode(".jpg", image)
    if is_success:
        st.download_button(
            label="Download Image with Detections",
            data=buffer.tobytes(),
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )