# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# # --- CONFIGURATION ---
# # IMPORTANT: Point this to one of the TEST folders, which contains an anomaly
# TEST_DATA_PATH = "C:\\Users\\Nandini Devi\\Downloads\\uropds\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Test\\Test001" 
# IMG_SIZE = 256
# # -------------------

# def preprocess_frame(frame):
#     """Prepares a single frame for the autoencoder."""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
#     normalized = resized.astype('float32') / 255.0
#     return normalized

# def calculate_mse(img1, img2):
#     """Calculates the Mean Squared Error (reconstruction error) between two images."""
#     return np.mean((img1 - img2)**2)

# def evaluate_model(model_path, test_data_path):
#     """Loads a model and calculates the reconstruction error for every frame."""
#     print(f"\nEvaluating model: {model_path}")
#     model = tf.keras.models.load_model(model_path)
    
#     frame_files = [f for f in sorted(os.listdir(test_data_path)) if f.endswith('.tif') or f.endswith('.jpg')]
#     errors = []
    
#     for filename in frame_files:
#         frame_path = os.path.join(test_data_path, filename)
#         frame = cv2.imread(frame_path)
        
#         processed_frame = preprocess_frame(frame)
#         input_frame = np.reshape(processed_frame, (1, IMG_SIZE, IMG_SIZE, 1))
        
#         # Get the model's reconstruction
#         reconstructed_frame = model.predict(input_frame, verbose=0)
        
#         # Calculate the error
#         error = calculate_mse(input_frame, reconstructed_frame)
#         errors.append(error)
        
#     print("Evaluation complete.")
#     return errors

# # --- Main script ---
# if __name__ == "__main__":
#     # 1. Evaluate the SECURE model
#     secure_errors = evaluate_model("global_model_secured.h5", TEST_DATA_PATH)
    
#     # 2. Evaluate the ATTACKED model
#     attacked_errors = evaluate_model("global_model_insecured.h5", TEST_DATA_PATH)
    
#     # 3. Plot the comparison
#     print("Plotting results...")
#     plt.figure(figsize=(12, 7))
#     plt.plot(secure_errors, label='FedGuard (Secure Model)', color='blue', linewidth=2)
#     plt.plot(attacked_errors, label='Attacked Model (No Defense)', color='red', linestyle='--')
#     plt.title('FedGuard vs. Attacked Model Performance')
#     plt.xlabel('Video Frame Number')
#     plt.ylabel('Anomaly Score (Reconstruction Error)')
#     plt.legend()
#     plt.show()
"""
Comprehensive Model Evaluation with Multiple Metrics.
Compares secure vs attacked models on test data.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import logging
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.data_loader import load_test_frames, get_ground_truth_labels
from utils.metrics import calculate_reconstruction_error, evaluate_anomaly_detection

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'evaluation.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_single_model(model_path, test_folder, model_name="Model"):
    """
    Evaluate a single model on test data.
    
    Returns:
        errors: List of reconstruction errors per frame
        metrics: Dictionary of evaluation metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test folder: {test_folder}")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None, None
    
    # Load test data
    logger.info("\nLoading test frames...")
    frames, frame_names = load_test_frames(test_folder, img_size=IMG_SIZE)
    logger.info(f"✓ Loaded {len(frames)} frames")
    
    # Load ground truth
    logger.info("Loading ground truth labels...")
    ground_truth = get_ground_truth_labels(test_folder)
    logger.info(f"✓ Ground truth loaded ({sum(ground_truth)} anomalies)")
    
    # Calculate reconstruction errors
    logger.info("\nCalculating reconstruction errors...")
    errors = []
    
    for i, frame in enumerate(frames):
        input_frame = np.reshape(frame, (1, IMG_SIZE, IMG_SIZE, 1))
        reconstructed = model.predict(input_frame, verbose=0)
        error = calculate_reconstruction_error(input_frame, reconstructed, method='mse')
        errors.append(error)
        
        if (i + 1) % 50 == 0:
            logger.info(f"  Processed {i+1}/{len(frames)} frames")
    
    logger.info("✓ Reconstruction errors calculated")
    
    # Evaluate anomaly detection
    logger.info("\nEvaluating anomaly detection performance...")
    metrics = evaluate_anomaly_detection(
        errors,
        ground_truth,
        percentile=ANOMALY_PERCENTILE
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"Threshold (MSE): {metrics['threshold']:.6f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Precision: {metrics['precision']:.2%}")
    logger.info(f"Recall: {metrics['recall']:.2%}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC: {metrics['auc']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Positives:  {metrics['true_positives']}")
    logger.info(f"  False Positives: {metrics['false_positives']}")
    logger.info(f"  True Negatives:  {metrics['true_negatives']}")
    logger.info(f"  False Negatives: {metrics['false_negatives']}")
    logger.info("="*60)
    
    return errors, metrics


def compare_models():
    """
    Main evaluation function: Compare secure vs insecure models.
    """
    logger.info("\n" + "="*60)
    logger.info("FEDGUARD MODEL COMPARISON")
    logger.info("="*60)
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test data path
    test_folder = os.path.join(UCSD_PED1_TEST, TEST_FOLDERS['ped1'])
    
    if not os.path.exists(test_folder):
        logger.error(f"Test folder not found: {test_folder}")
        return
    
    # Evaluate both models
    secure_