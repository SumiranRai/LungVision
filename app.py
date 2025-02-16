import os
import io
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
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
    MODEL_PATH = "best-model.pt"  # Ensure correct path
    return load_model(MODEL_PATH, device)

model, class_names = load_cached_model()

# Title
st.title("Tuberculosis & Pneumonia Detection 🫁")

# File uploader
uploaded_file = st.file_uploader("📂 Upload a **chest X-ray** for **AI-based** diagnosis.", type=["jpg", "png", "jpeg"])

# Cache the prediction function
@st.cache_data
def process_image(image):
    """Process the image and return predictions."""
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)

    # Ensure Grad-CAM is in RGB format for Streamlit
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return predicted_label, confidence_score, grad_cam_overlay

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 25:
        st.error(f"❌ File too large! (Your file: {file_size_mb:.2f} MB)")
    else:
        # Load and process image
        image = Image.open(uploaded_file).convert("RGB")  # Ensure it's RGB

        # Run prediction
        with st.spinner("🔍 Analyzing..."):
            predicted_label, confidence_score, grad_cam_overlay = process_image(image)

        # Display uploaded image and Grad-CAM side by side
        # Display results in columns for better alignment
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Uploaded Image", width=300)

        with col2:
            st.image(grad_cam_overlay, caption="🔥 Grad-CAM Heatmap", width=300)

        # Display diagnosis results
        st.subheader("🔬 Prediction Results")
        st.write(f"**Prediction:** `{predicted_label}`")
        st.write(f"**Confidence:** `{confidence_score:.2f}%`")

        # Provide a brief explanation when "Other Findings" is detected
        if predicted_label == "OTHER FINDINGS":
            st.write("⚠️ **Diagnosis:** The model is not confident in this classification. The image has been labeled as 'Other Findings'.")

        # Convert Grad-CAM for downloading (optimized PNG compression)
        grad_cam_pil = Image.fromarray(grad_cam_overlay)
        img_byte_arr = io.BytesIO()
        grad_cam_pil.save(img_byte_arr, format='PNG', optimize=True)
        img_byte_arr = img_byte_arr.getvalue()

        # Download buttons
        st.download_button(
            "📥 Download Diagnosis", 
            data=f"Prediction: {predicted_label}\nConfidence: {confidence_score:.2f}%", 
            file_name="diagnosis.txt"
        )

        st.download_button(
            "📥 Download Grad-CAM", 
            data=img_byte_arr, 
            file_name="grad_cam.png", 
            mime="image/png"
        )
