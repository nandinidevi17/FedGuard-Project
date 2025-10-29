import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

# This function defines the structure of our Autoencoder model.
def create_autoencoder(input_shape=(256, 256, 1)):
    """
    Creates a Convolutional Autoencoder model.
    An autoencoder has two parts: an Encoder and a Decoder.
    """
    # --- The Encoder ---
    # It takes a big image and compresses it into a small representation.
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x) # The compressed representation

    # --- The Decoder ---
    # It takes the small representation and tries to rebuild the original image.
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x) # The reconstructed image

    # This model maps an input image to its reconstruction
    autoencoder = Model(input_img, decoded)
    
    # We compile the model, telling it how to learn.
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Autoencoder model created successfully.")
    autoencoder.summary() # Prints a summary of the model structure
    
    return autoencoder

# This is a helper function to prepare our video frames for the model.
def preprocess_frame(frame):
    """
    Takes a video frame, converts it to grayscale, and resizes it.
    """
    # Resize the frame to the size our model expects.
    frame = cv2.resize(frame, (256, 256))
    # Convert the frame to grayscale because color isn't needed for this task.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize the pixel values to be between 0 and 1.
    frame = frame.astype('float32') / 255.0
    # Reshape it to include dimensions for batch size and channels.
    frame = np.reshape(frame, (1, 256, 256, 1))
    return frame


# --- This is where we will add code later to train and test the model ---

# For now, let's just create an instance of the model to see if it works.
if __name__ == '__main__':
    model = create_autoencoder()
    # (Keep all the code you already have in anomaly_model.py)
# ...

# This is the new code to add at the bottom
def load_training_data(folder_path="training_data"):
    """Loads all the training images from the specified folder."""
    train_images = []
    print(f"Loading images from {folder_path}...")
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Normalize pixel values to be between 0 and 1
                img = img.astype('float32') / 255.0
                train_images.append(img)
    print(f"Loaded {len(train_images)} images.")
    # Convert the list of images to a single NumPy array
    return np.array(train_images)

# This is where we will add code later to train and test the model ---

# For now, let's just create an instance of the model to see if it works.
if __name__ == '__main__':
    # 1. Create the model
    model = create_autoencoder()
    
    # 2. Load the training data
    # Reshape data to fit the model: (num_samples, height, width, channels)
    X_train = load_training_data()
    X_train = np.reshape(X_train, (len(X_train), 256, 256, 1))

    # 3. Train the model
    print("\nStarting model training...")
    # An epoch is one full pass through the entire training dataset.
    # We will train for 10 epochs.
    model.fit(X_train, X_train, # The model learns to make the output (y) the same as the input (x)
              epochs=10,
              batch_size=16,
              shuffle=True)
    
    # 4. Save the trained model for later use
    model.save('autoencoder.h5')
    print("\nTraining complete. Model saved as 'autoencoder.h5'.")