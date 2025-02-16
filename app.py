import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
from model import load_model, predict

# Ensure Streamlit config directory exists
os.makedirs(".streamlit", exist_ok=True)
with open(".streamlit/config.toml", "w") as f:
    f.write("[server]\nmaxUploadSize = 25\n")

# Set device (CPU or CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache the model for better performance
@st.cache_resource
def load_cached_model():
    """Load model once and cache it."""
    MODEL_PATH = "best-model.pt"
    return load_model(MODEL_PATH, device)

model, class_names = load_cached_model()

# Get absolute path for sidebar image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sidebar_logo_path = os.path.join(BASE_DIR, "sidebar_logo.png")

# Sidebar with settings (if image exists)
with st.sidebar:
    if os.path.exists(sidebar_logo_path):
        st.image(sidebar_logo_path, width=120)

    st.subheader("⚙️ Settings")
    confidence_threshold = st.slider("📊 Confidence Threshold", 50, 100, 80)

# Title
st.title("Tuberculosis & Pneumonia Detection 🫁")
st.write("📂 Upload a **chest X-ray** for **AI-based diagnosis**.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload X-ray Image", type=["jpg", "png", "jpeg"])

# Cache the prediction function
@st.cache_data
def process_image(image):
    """Process the image and return predictions."""
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return predicted_label, confidence_score, grad_cam_overlay

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 25:
        st.error(f"❌ File too large! (Your file: {file_size_mb:.2f} MB)")
    else:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(image, caption="📷 Uploaded Image", width=300)

        # Run prediction (cached)
        with st.spinner("🔍 Analyzing..."):
            predicted_label, confidence_score, grad_cam_overlay = process_image(image)

        # Display results
        st.subheader("🔬 Prediction Results")
        st.write(f"**Prediction:** `{predicted_label}`")
        st.write(f"**Confidence:** `{confidence_score:.2f}%`")

        # Display Grad-CAM heatmap
        st.image(grad_cam_overlay, caption="🔥 Grad-CAM Heatmap", width=300)

        # Download Button for Diagnosis
        st.download_button("📥 Download Diagnosis", 
                           data=f"Prediction: {predicted_label}\nConfidence: {confidence_score:.2f}%", 
                           file_name="diagnosis.txt")
