
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Get the project root directory from the script's location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define paths using the project root
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UCSD_Anomaly_Dataset.v1p2", "UCSDped1", "Train", "Train001")
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")

IMG_SIZE = 128
NUM_IMAGES = 20

def create_dummy_data():
    """Creates a directory with dummy images."""
    print(f"Creating dummy data directory at: {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    for i in range(NUM_IMAGES):
        # Create a random grayscale image
        random_array = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        img = Image.fromarray(random_array, 'L')
        img.save(os.path.join(DATA_DIR, f"frame_{i:03d}.tif"))

    print(f"Successfully created {NUM_IMAGES} dummy images.")

def create_dummy_model():
    """Creates and saves a simple dummy autoencoder model."""
    print(f"Creating dummy model at: {MODEL_PATH}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Define a very simple autoencoder
    input_img = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = tf.keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')(x)

    x = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    # Save the untrained model
    autoencoder.save(MODEL_PATH)
    print("Successfully created and saved dummy model.")

if __name__ == "__main__":
    create_dummy_data()
    create_dummy_model()
    print("\n✅ Dummy data and model created successfully.")
