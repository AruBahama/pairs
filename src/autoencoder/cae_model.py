
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from ..config import WINDOW_LENGTH, LATENT_DIM

def build_cae(n_features: int):
    """Build a simple convolutional autoencoder using 1D convolutions."""
    inputs = layers.Input(shape=(WINDOW_LENGTH, n_features))

    # Encoder
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPool1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool1D(2)(x)
    shape = tf.keras.backend.int_shape(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(LATENT_DIM, name="latent")(x)

    # Decoder
    x = layers.Dense(np.prod(shape[1:]), activation="relu")(latent)
    x = layers.Reshape(shape[1:])(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1DTranspose(32, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling1D(2)(x)
    outputs = layers.Conv1DTranspose(n_features, 3, activation="linear", padding="same")(x)

    model = Model(inputs,outputs)
    encoder = Model(inputs,latent)
    model.compile(optimizer='adam',loss='mse')
    return model, encoder
