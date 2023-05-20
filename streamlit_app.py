import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image,ImageOps

@st.cache_resource()

def load_model():
    model = tf.keras.models.load_model('car_bike_classifier.h5',compile=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['acc'])
    return model

def import_and_predict(image_data,model):
    size =  (75,75)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_reshape=gray[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction

st.write("""
# Car-Bike Detection """
         )

file=st.file_uploader("Choose Car or Bike photo from computer",
                      type=["jpg","jpeg","png"])

if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    model = load_model()
    prediction=import_and_predict(image,model)
    class_names=["Bike","Car"]
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)



