import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import gdown
import os

# Streamlit page config
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Classifier")
st.write("Upload an image and see if it's a **cat** or **dog**!")

# ✅ Model download from Google Drive if not present
model_path = "model.h5"
gdrive_url = "https://drive.google.com/uc?id=1kIDT8hr8N62gGveHhA8ch_GuxVZT-Nww"

if not os.path.exists(model_path):
    with st.spinner("📥 Downloading model from Google Drive..."):
        gdown.download(gdrive_url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

# ✅ Load model using Streamlit cache
@st.cache_resource
def load_model():
    return keras.models.load_model(model_path)

model = load_model()

# ✅ Preprocessing function
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to model input size
    image_array = keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Class labels
class_names = ['Cat', 'Dog']

# ✅ Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("⏳ Predicting...")
    img_array = preprocess_image(image)

    prob = model.predict(img_array)[0][0]
    pred_class = 1 if prob > 0.5 else 0
    label = class_names[pred_class]

    st.success(f"Prediction: **{label}**")
    st.write(f"Confidence (probability): **{prob:.2f}**")
