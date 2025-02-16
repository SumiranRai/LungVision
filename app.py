import os
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("[server]\nmaxUploadSize = 25\n")

import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from model import load_model, predict

# Set device (CPU or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = "best-model.pt"
model, class_names = load_model(MODEL_PATH, device)

# Streamlit UI
st.title("Lungetivity: Tuberculosis & Pneumonia Detection")
st.write("Upload a chest X-ray image for classification and Grad-CAM visualization.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Prediction
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)

    # Convert Grad-CAM overlay for Streamlit display
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    # Create side-by-side columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", width=300)  # Decreased size
        st.write(f"**Prediction:** {predicted_label}")
        st.write(f"**Confidence:** {confidence_score:.2f}%")

    with col2:
        st.image(grad_cam_overlay, caption="Grad-CAM Heatmap", width=300)  # Decreased size
