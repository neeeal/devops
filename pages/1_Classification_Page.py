import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(page_title="Classification Page", page_icon="🔎")

@st.cache_data
def load_model():
    model = tf.keras.models.load_model("assets/car_bike_classifier.h5", compile=False)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model

def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_reshape = gray[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

st.markdown(
    """
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        height: 20vh; /* Increase the height for better centering */
    }
    .title {
        font-size: 48px; /* Increased font size for the title */
    }
    .emoji {
        font-size: 48px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="center"><span class="emoji">⏱️ → 📟</span><h1 class="title">Analog to Digital Time</h1></div>', unsafe_allow_html=True)

uploaded_images = st.file_uploader("Choose up to 5 analog clock photos from your device", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if not uploaded_images:
    st.text("Please upload image files")
else:
    model = load_model()
    class_names = ["Bike", "Car"]
    num_images = min(len(uploaded_images), 5)

    for i in range(num_images):
        image = Image.open(uploaded_images[i])
        st.image(image, use_column_width=True)
        prediction = import_and_predict(image, model)
        string = "OUTPUT : " + class_names[np.argmax(prediction)]
        st.success(string)
