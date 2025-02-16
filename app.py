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

# Caching the model to optimize performance
@st.cache_resource
def load_cached_model():
    """Cache the model to avoid reloading on each run."""
    MODEL_PATH = "best-model.pt"
    return load_model(MODEL_PATH, device)

model, class_names = load_cached_model()

# Custom CSS for better styling
st.markdown("""
    <style>
        .css-1d391kg {padding: 20px;} /* Padding around main container */
        .stApp {background-color: #f5f5f5;} /* Background color */
        h1 {text-align: center; color: #4CAF50;} /* Title Styling */
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for extra features
with st.sidebar:
    st.image("sidebar_logo.png", width=150)
    st.subheader("⚙️ App Settings")
    dark_mode = st.checkbox("🌙 Enable Dark Mode")
    confidence_threshold = st.slider("📊 Confidence Threshold", 50, 100, 80)

# Apply Dark Mode Styling
if dark_mode:
    st.markdown("""
        <style>
            .stApp {background-color: #333333; color: white;}
        </style>
    """, unsafe_allow_html=True)

# Main UI
st.image("logo.png", width=200)  # Add a logo at the top
st.title("Lungetivity: Tuberculosis & Pneumonia Detection")
st.write("🔍 Upload a chest X-ray image for classification and Grad-CAM visualization.")
st.write("⚠️ **Maximum file size allowed: 25 MB**")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

# Cache the image processing to save resources
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
        st.success("✅ Image uploaded successfully!")

        # Show progress bar
        progress_bar = st.progress(0)
        for percent in range(100):
            progress_bar.progress(percent + 1)

        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Get prediction and Grad-CAM visualization (cached)
        predicted_label, confidence_score, grad_cam_overlay = process_image(image)

        # Display results using columns for a better layout
        col1, col2 = st.columns([2, 3])

        with col1:
            st.image(image, caption="📷 Uploaded Image", width=300)
            st.write(f"**🔍 Prediction:** {predicted_label}")
            st.write(f"**📊 Confidence:** {confidence_score:.2f}%")

        with col2:
            st.image(grad_cam_overlay, caption="🔥 Grad-CAM Heatmap", width=300)

        # Add a "Download Results" button
        st.download_button("📥 Download Report", data=f"Prediction: {predicted_label}\nConfidence: {confidence_score:.2f}%", file_name="diagnosis.txt")

# Footer
st.markdown("---")
st.write("🚀 **Mini Project | Built with ❤️.**")
