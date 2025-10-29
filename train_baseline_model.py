import os
import cv2
import numpy as np
import tensorflow as tf
from anomaly_model import create_autoencoder # Import our model structure

# --- CONFIGURATION ---
DATASET_PATH = 'C:\\Users\\Nandini Devi\\Downloads\\uropds\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train' # IMPORTANT: Set this to your 'Train' folder
IMG_SIZE = 256
# -------------------

def load_training_images(folder_path):
    """Loads all frames from all subfolders in the training path."""
    train_images = []
    print(f"Loading training images from: {folder_path}")
    
    # Loop through all subfolders (e.g., Train001, Train002)
    for train_folder in sorted(os.listdir(folder_path)):
        subfolder_path = os.path.join(folder_path, train_folder)
        if not os.path.isdir(subfolder_path):
            continue
            
        print(f"  - Loading from {train_folder}...")
        # Loop through all image frames in the subfolder
        for filename in sorted(os.listdir(subfolder_path)):
            if filename.endswith(".tif") or filename.endswith(".jpg"):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img.astype('float32') / 255.0
                    train_images.append(img)
                    
    print(f"Loaded {len(train_images)} total training images.")
    return np.array(train_images)

# --- Main script ---
if __name__ == "__main__":
    # 1. Create the model
    model = create_autoencoder(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # 2. Load the "normal" training data
    X_train = load_training_images(DATASET_PATH)
    X_train = np.reshape(X_train, (len(X_train), IMG_SIZE, IMG_SIZE, 1))
    
    # 3. Train the model
    print("\nStarting NEW baseline model training...")
    model.fit(X_train, X_train,
              epochs=10, # You can increase this for better results
              batch_size=16,
              shuffle=True)
              
    # 4. Save the new, clean baseline model
    model.save('ucsd_baseline.h5')
    print("\nTraining complete. New model saved as 'ucsd_baseline.h5'.")