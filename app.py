import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from model import load_model, predict
from gradcam import generate_grad_cam, overlay_grad_cam

# Set device (Streamlit does not provide CUDA, so force CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model, class_names = load_model("best-model.pt", device)

# Streamlit UI
st.title("Tuberculosis Detection from X-ray Images 🫁")
st.write("Upload a chest X-ray to classify it as **Normal, Pneumonia, Tuberculosis, or Unknown.**")

# File uploader
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict
    with st.spinner("Analyzing..."):
        predicted_label, confidence, grad_cam_overlay = predict(model, image, class_names, device)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Display Grad-CAM
    st.image(grad_cam_overlay, caption="Grad-CAM Activation Map", use_column_width=True)
