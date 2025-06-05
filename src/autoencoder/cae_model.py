
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from ..config import WINDOW_LENGTH, LATENT_DIM

def build_cae(n_features: int, window_length: int = WINDOW_LENGTH, latent_dim: int = LATENT_DIM):
    """Build a simple convolutional autoencoder."""
    inputs = layers.Input(shape=(window_length, n_features, 1))
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    shape = tf.keras.backend.int_shape(x)
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent')(x)

    # Decoder
    x = layers.Dense(np.prod(shape[1:]),activation='relu')(latent)
    x = layers.Reshape(shape[1:])(x)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2DTranspose(32,(3,3),activation='relu',padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    outputs = layers.Conv2DTranspose(1,(3,3),activation='linear',padding='same')(x)

    model = Model(inputs,outputs)
    encoder = Model(inputs,latent)
    model.compile(optimizer='adam',loss='mse')
    return model, encoder
