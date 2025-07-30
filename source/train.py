import numpy as np
import os
from build import build_model 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image  import ImageDataGenerator

basedir=os.path.join(os.path.dirname(os.path.dirname(__file__)),"Data")

train_img=ImageDataGenerator(rescale=1./255,
           rotation_range=20,
           zoom_range=0.2,
           shear_range=0.2,
           horizontal_flip=True,
           vertical_flip=True)
test_img=ImageDataGenerator(rescale=1./255)

train_data=train_img.flow_from_directory(
    os.path.join(basedir,"train"),
    target_size=(224,224),
    class_mode="categorical",
    batch_size=32
    )


test_data=test_img.flow_from_directory(
    os.path.join(basedir,"test"),
    target_size=(224,224),
    class_mode="categorical",
    batch_size=32)

model=build_model()

model.fit(train_data,epochs=1)
model.save(os.path.join(basedir,"covid_pneu_model.h5"))
