import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

X_train = np.load("processed_data/X_train.npy")
X_val = np.load("processed_data/X_val.npy")
y_train = np.load("processed_data/y_train.npy")
y_val = np.load("processed_data/y_val.npy")

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(43, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

os.makedirs("models", exist_ok=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

model.save("models/traffic_sign_model.keras")

print("Model training completed successfully!")
print("Model saved to models/traffic_sign_model.keras")