import tensorflow as tf 
from tensorflow.keras import layers,models

def build_model():
    model=models.Sequential([
        layers.Input(shape=(224,224,3)),
        layers.Conv2D(32,(3,3),activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation="relu") ,
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3),activation="relu"),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128,activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3,activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['acc']
    )
    return model
    
