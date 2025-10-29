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

# Add project root to path (two levels up from experiments/evaluation/)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import *
from utils.data_loader import load_test_frames, get_ground_truth_labels
from utils.metrics import calculate_reconstruction_error, evaluate_anomaly_detection

# Setup logging with UTF-8 encoding
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'evaluation.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
# Set UTF-8 encoding for stdout to handle special characters
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
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
    secure_model_path = os.path.join(MODELS_DIR, 'global_model_secured.h5')
    insecure_model_path = os.path.join(MODELS_DIR, 'global_model_insecured.h5')
    
    # Check if models exist
    if not os.path.exists(secure_model_path):
        logger.error(f"Secure model not found: {secure_model_path}")
        logger.error("Run fedguard_server.py first!")
        return
    
    if not os.path.exists(insecure_model_path):
        logger.error(f"Insecure model not found: {insecure_model_path}")
        logger.error("Run insecure_server.py first!")
        return
    
    # Evaluate secure model
    secure_errors, secure_metrics = evaluate_single_model(
        secure_model_path,
        test_folder,
        model_name="FedGuard (Secure)"
    )
    
    if secure_errors is None:
        logger.error("Failed to evaluate secure model")
        return
    
    # Evaluate attacked model
    attacked_errors, attacked_metrics = evaluate_single_model(
        insecure_model_path,
        test_folder,
        model_name="Attacked (No Defense)"
    )
    
    if attacked_errors is None:
        logger.error("Failed to evaluate attacked model")
        return
    
    # Calculate improvement
    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)
    
    metrics_comparison = {
        'Accuracy': (secure_metrics['accuracy'], attacked_metrics['accuracy']),
        'Precision': (secure_metrics['precision'], attacked_metrics['precision']),
        'Recall': (secure_metrics['recall'], attacked_metrics['recall']),
        'F1-Score': (secure_metrics['f1_score'], attacked_metrics['f1_score']),
        'AUC-ROC': (secure_metrics['auc'], attacked_metrics['auc'])
    }
    
    for metric_name, (secure_val, attacked_val) in metrics_comparison.items():
        improvement = ((secure_val - attacked_val) / (attacked_val + 1e-8)) * 100
        logger.info(f"\n{metric_name}:")
        logger.info(f"  Secure:   {secure_val:.4f}")
        logger.info(f"  Attacked: {attacked_val:.4f}")
        logger.info(f"  Improvement: {improvement:+.2f}%")
    
    # Save comprehensive results
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'test_folder': test_folder,
        'secure_model': {
            'path': secure_model_path,
            'errors': [float(e) for e in secure_errors],
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in secure_metrics.items() if k not in ['fpr', 'tpr']}
        },
        'attacked_model': {
            'path': insecure_model_path,
            'errors': [float(e) for e in attacked_errors],
            'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in attacked_metrics.items() if k not in ['fpr', 'tpr']}
        },
        'comparison': {
            metric: {
                'secure': float(vals[0]),
                'attacked': float(vals[1]),
                'improvement_percent': float(((vals[0] - vals[1]) / (vals[1] + 1e-8)) * 100)
            }
            for metric, vals in metrics_comparison.items()
        }
    }
    
    results_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Detailed results saved to: {results_file}")
    logger.info("="*60)
    
    # Return for visualization
    return {
        'secure_errors': secure_errors,
        'attacked_errors': attacked_errors,
        'secure_metrics': secure_metrics,
        'attacked_metrics': attacked_metrics,
        'ground_truth': get_ground_truth_labels(test_folder)
    }


if __name__ == "__main__":
    results = compare_models()
    
    if results:
        print("\n✅ Evaluation complete!")
        print(f"Results saved in: {METRICS_DIR}")
        print("\nNext step: Run visualize_results.py to generate graphs")