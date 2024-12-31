import streamlit as st
import requests
from PIL import Image
import io

# FastAPI server URL
FASTAPI_URL = "http://127.0.0.1:8000/predict/"

# Streamlit UI
st.title("Fashion-MNIST Image Classifier")
st.write("Upload an image to classify it into one of the fashion categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to bytes and send to FastAPI backend for prediction
    img_bytes = uploaded_file.read()

    # Send the image to FastAPI for prediction
    response = requests.post(FASTAPI_URL, files={"file": img_bytes})

    # Parse the response and display prediction
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Predicted Class: {prediction['class']}")
        st.write(f"Prediction Confidence: {prediction['confidence'] * 100:.2f}%")
    else:
        st.write("Error with the prediction request!")
