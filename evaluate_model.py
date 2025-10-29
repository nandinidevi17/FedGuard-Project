import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# IMPORTANT: Point this to one of the TEST folders, which contains an anomaly
TEST_DATA_PATH = "C:\\Users\\Nandini Devi\\Downloads\\uropds\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Test\\Test001" 
IMG_SIZE = 256
# -------------------

def preprocess_frame(frame):
    """Prepares a single frame for the autoencoder."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    return normalized

def calculate_mse(img1, img2):
    """Calculates the Mean Squared Error (reconstruction error) between two images."""
    return np.mean((img1 - img2)**2)

def evaluate_model(model_path, test_data_path):
    """Loads a model and calculates the reconstruction error for every frame."""
    print(f"\nEvaluating model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    frame_files = [f for f in sorted(os.listdir(test_data_path)) if f.endswith('.tif') or f.endswith('.jpg')]
    errors = []
    
    for filename in frame_files:
        frame_path = os.path.join(test_data_path, filename)
        frame = cv2.imread(frame_path)
        
        processed_frame = preprocess_frame(frame)
        input_frame = np.reshape(processed_frame, (1, IMG_SIZE, IMG_SIZE, 1))
        
        # Get the model's reconstruction
        reconstructed_frame = model.predict(input_frame, verbose=0)
        
        # Calculate the error
        error = calculate_mse(input_frame, reconstructed_frame)
        errors.append(error)
        
    print("Evaluation complete.")
    return errors

# --- Main script ---
if __name__ == "__main__":
    # 1. Evaluate the SECURE model
    secure_errors = evaluate_model("global_model_secured.h5", TEST_DATA_PATH)
    
    # 2. Evaluate the ATTACKED model
    attacked_errors = evaluate_model("global_model_insecured.h5", TEST_DATA_PATH)
    
    # 3. Plot the comparison
    print("Plotting results...")
    plt.figure(figsize=(12, 7))
    plt.plot(secure_errors, label='FedGuard (Secure Model)', color='blue', linewidth=2)
    plt.plot(attacked_errors, label='Attacked Model (No Defense)', color='red', linestyle='--')
    plt.title('FedGuard vs. Attacked Model Performance')
    plt.xlabel('Video Frame Number')
    plt.ylabel('Anomaly Score (Reconstruction Error)')
    plt.legend()
    plt.show()