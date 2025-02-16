import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from model import load_model, predict

# Ensure Streamlit config directory exists and set upload size limit
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("[server]\nmaxUploadSize = 25\n")

# Set device (CPU or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_cached_model():
    """Cache the model to avoid reloading on each run."""
    MODEL_PATH = "best-model.pt"
    return load_model(MODEL_PATH, device)

model, class_names = load_cached_model()

# Streamlit UI
st.title("Lungetivity: Tuberculosis & Pneumonia Detection")
st.write("Upload a chest X-ray image for classification and Grad-CAM visualization.")
st.write("⚠️ **Maximum file size allowed: 25 MB**")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

@st.cache_data
def process_image(image):
    """Cache the model's prediction and Grad-CAM processing for the same image."""
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)
    
    # Convert Grad-CAM overlay for Streamlit display
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    return predicted_label, confidence_score, grad_cam_overlay

if uploaded_file is not None:
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert bytes to MB
    if file_size_mb > 25:
        st.error(f"❌ File size exceeds 25 MB limit! (Your file: {file_size_mb:.2f} MB)")
    else:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Get prediction and Grad-CAM visualization (cached)
        predicted_label, confidence_score, grad_cam_overlay = process_image(image)

        # Create side-by-side columns
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", width=300)  # Decreased size
            st.write(f"**Prediction:** {predicted_label}")
            st.write(f"**Confidence:** {confidence_score:.2f}%")

        with col2:
            st.image(grad_cam_overlay, caption="Grad-CAM Heatmap", width=300)  # Decreased size
