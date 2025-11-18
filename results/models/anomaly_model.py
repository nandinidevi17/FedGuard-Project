"""
Convolutional Autoencoder for Anomaly Detection.
Optimized architecture for 256x256 grayscale images.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

def create_autoencoder(input_shape=(256, 256, 1)):
    """
    Create a deep convolutional autoencoder.
    """
    encoder_input = layers.Input(shape=input_shape, name='encoder_input')

    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(encoder_input)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    x = layers.Conv2D(256, (3,3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2,2), padding='same')(x)

    encoded = x  # e.g., 16x16x256

    x = layers.Conv2DTranspose(256, (3,3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(128, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(64, (3,3), strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(32, (3,3), strides=2, activation='relu', padding='same')(x)

    decoded = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same', name='decoder_output')(x)

    autoencoder = models.Model(encoder_input, decoded, name='autoencoder')
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return autoencoder

if __name__ == "__main__":
    print("Creating autoencoder model...")
    model = create_autoencoder()
    model.summary()
    import numpy as np
    dummy_input = np.random.rand(1, 256, 256, 1).astype('float32')
    output = model.predict(dummy_input, verbose=0)
    print("Model input->output:", dummy_input.shape, "->", output.shape)
