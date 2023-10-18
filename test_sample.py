import tensorflow as tf
import cv2
import numpy as np
from PIL import Image,ImageOps

def load_model():
    model = tf.keras.models.load_model("assets/best_model (1).h5", compile=False)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model

def import_and_predict(image_data, model):
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img = np.asarray(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_reshape = gray[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

def test_prediction():
    model = load_model()
    image = Image.open("assets/22.jpg")
    result = import_and_predict(image,model)
    print(result)
    assert np.argmax(result) != np.nan
    
