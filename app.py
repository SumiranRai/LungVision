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
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed warning

    # Prediction
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)

    # Display Results
    st.write(f"**Prediction:** {predicted_label}")
    st.write(f"**Confidence:** {confidence_score:.2f}%")

    # Show Grad-CAM heatmap
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    st.image(grad_cam_overlay, caption="Grad-CAM Heatmap", use_container_width=True)  # Fixed warning
