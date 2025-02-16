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

# Set device (force CPU for Streamlit Cloud compatibility)
device = torch.device("cpu")

# Cache the model for better performance
@st.cache_resource
def load_cached_model():
    """Load model once and cache it."""
    return load_model("best-model.pt", device)

model, class_names = load_cached_model()

# Custom UI/UX with CSS
st.markdown("""
    <style>
        .stApp { background-color: #F5F7F9; }
        .stButton>button { background-color: #00796B !important; color: white !important; }
        .stButton>button:hover { background-color: #004D40 !important; }
        h1 { text-align: center; color: #00796B; }
        .sidebar { background-color: #CFD8DC; padding: 15px; border-radius: 10px; }
        .footer { text-align: center; font-size: 13px; color: #607D8B; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# Sidebar with settings
with st.sidebar:
    st.image("sidebar_logo.png", width=120)
    st.subheader("⚙️ Settings")
    dark_mode = st.checkbox("🌙 Dark Mode")
    confidence_threshold = st.slider("📊 Confidence Threshold", 50, 100, 80)

# Apply Dark Mode
if dark_mode:
    st.markdown("""
        <style>
            .stApp { background-color: #263238; color: white; }
        </style>
    """, unsafe_allow_html=True)

# Header with Logo
st.image("logo.png", width=150)
st.title("Lungetivity: AI-based Chest X-ray Diagnosis")
st.write("Upload a **chest X-ray** for **AI-based tuberculosis & pneumonia detection**.")
st.write("⚠️ **Max File Size: 25MB**")

uploaded_file = st.file_uploader("📂 Upload X-ray Image", type=["jpg", "png", "jpeg"])

# Cache the prediction function
@st.cache_data
def process_image(image):
    """Process the image and return predictions."""
    predicted_label, confidence_score, grad_cam_overlay = predict(model, image, class_names, device)

    # Convert Grad-CAM overlay to RGB format
    grad_cam_overlay = cv2.cvtColor((grad_cam_overlay * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return predicted_label, confidence_score, grad_cam_overlay

if uploaded_file:
    # File size validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > 25:
        st.error(f"❌ File too large! (Your file: {file_size_mb:.2f} MB)")
    else:
        st.success("✅ Image uploaded successfully!")

        # Progress Bar
        progress_bar = st.progress(0)
        for percent in range(100):
            progress_bar.progress(percent + 1)

        # Load & preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        image = image.resize((300, 300))  # Resize for faster inference

        # Run prediction (cached)
        predicted_label, confidence_score, grad_cam_overlay = process_image(image)

        # Display results in two columns
        col1, col2 = st.columns([2, 3])
        with col1:
            st.image(image, caption="📷 Uploaded Image", width=250)
            st.success(f"**Prediction:** `{predicted_label}`")
            st.info(f"**Confidence:** `{confidence_score:.2f}%`")

        with col2:
            st.image(grad_cam_overlay, caption="🔥 Grad-CAM Heatmap", width=250)

        # Diagnosis Download
        st.download_button("📥 Download Diagnosis", 
                           data=f"Prediction: {predicted_label}\nConfidence: {confidence_score:.2f}%", 
                           file_name="diagnosis.txt")

# Footer
st.markdown('<div class="footer">🚀 **Lungetivity | Mini Project | Built with ❤️**</div>', unsafe_allow_html=True)
