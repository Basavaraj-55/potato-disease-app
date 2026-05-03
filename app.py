import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("potato_model.h5")

classes = ["Early Blight", "Late Blight", "Healthy"]

st.title("Potato Leaf Disease Detection 🌱")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((150,150))
    st.image(img)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")