
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import streamlit as st

st.title("Disease Predictor")
class_=["Covid19","Normal","Pneumonia"]
ml=load_model("D:\demo-covid\Data\covid_pneu_model.h5")

# img=image.load_img("D:\\Covid19\\Data\\test\\PNEUMONIA\\PNEUMONIA(3420).jpg",target_size=(224,224))
u=st.file_uploader("Upload images")
sub=st.button("Predict")
if sub:
    img=image.load_img(u,target_size=(224,224))
    imag=image.img_to_array(img)
    imag=np.expand_dims(imag,axis=0)/255.0
    prd=ml.predict(imag)
    ind=np.argmax(prd[0])
    st.write(class_[ind])