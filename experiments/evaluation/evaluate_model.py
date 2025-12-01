# evaluate_model.py
"""
Comprehensive Model Evaluation with Multiple Metrics (robust + model loading fixes).
- Handles Keras version compatibility issues
- Loads two models: secure (global_model_secured.h5) and insecure (global_model_insecured.h5)
- Runs reconstruction on test frames, computes MSE per-frame
- Computes threshold by percentile, classification metrics, AUC when possible
- Saves results to METRICS_DIR/evaluation_results.json
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
import tensorflow as tf

# Try to import sklearn metrics if available (for AUC)
try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Add project root to path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import project config and loaders
try:
    from config import *
except Exception as e:
    raise

try:
    from utils.data_loader import load_test_frames, get_ground_truth_labels
except Exception as e:
    raise ImportError(f"Failed to import data loader utilities: {e}")

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT if 'LOG_FORMAT' in globals() else "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'evaluation.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GPU memory growth (if GPUs present)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        logger.warning("Could not set GPU memory growth")


def load_model_safe(model_path: str):
    """
    Safely load a Keras model with multiple fallback strategies.
    Handles version compatibility issues.
    """
    # Strategy 1: Direct load
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("✓ Model loaded successfully (direct)")
        return model
    except Exception as e1:
        logger.warning(f"Direct load failed: {e1}")
    
    # Strategy 2: Load with compile=False
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
        logger.info("✓ Model loaded successfully (compile=False)")
        return model
    except Exception as e2:
        logger.warning(f"Load with compile=False failed: {e2}")
    
    # Strategy 3: Custom object handling
    try:
        from tensorflow.keras.layers import InputLayer
        custom_objects = {'InputLayer': InputLayer}
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info("✓ Model loaded successfully (custom objects)")
        return model
    except Exception as e3:
        logger.warning(f"Load with custom objects failed: {e3}")
    
    # Strategy 4: Load weights only (requires architecture recreation)
    try:
        logger.info("Attempting to recreate model architecture and load weights...")
        from results.models.anomaly_model import create_autoencoder
        model = create_autoencoder(input_shape=(IMG_SIZE, IMG_SIZE, 1))
        model.load_weights(model_path)
        logger.info("✓ Model loaded successfully (weights only)")
        return model
    except Exception as e4:
        logger.warning(f"Load weights failed: {e4}")
    
    # All strategies failed
    logger.error(f"All model loading strategies failed for {model_path}")
    return None


def calculate_reconstruction_error(frames: np.ndarray, reconstructed: np.ndarray) -> List[float]:
    """
    Compute per-frame MSE between original frames and reconstructed frames.
    """
    if frames is None or reconstructed is None or len(frames) == 0:
        return []
    n = min(len(frames), len(reconstructed))
    diffs = (frames[:n].astype(np.float32) - reconstructed[:n].astype(np.float32)) ** 2
    per_frame_mse = np.mean(diffs.reshape(diffs.shape[0], -1), axis=1)
    return per_frame_mse.tolist()


def evaluate_anomaly_detection(errors: List[float], ground_truth: List[int], percentile: float = 99.0) -> Dict[str, Any]:
    """
    Evaluate anomaly detection performance.
    """
    results = {
        'threshold': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'auc': None,
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }

    if not errors or len(errors) == 0:
        return results

    errors_arr = np.array(errors, dtype=np.float32)
    try:
        thr = float(np.percentile(errors_arr, percentile))
    except Exception:
        thr = float(np.max(errors_arr))
    results['threshold'] = thr

    # Prepare ground truth
    if ground_truth is None or len(ground_truth) == 0:
        y_true = np.zeros_like(errors_arr, dtype=int)
    else:
        y_true = np.array(ground_truth, dtype=int)
        if len(y_true) < len(errors_arr):
            y_true = np.pad(y_true, (0, len(errors_arr) - len(y_true)), 'constant')
        elif len(y_true) > len(errors_arr):
            y_true = y_true[:len(errors_arr)]

    # Predictions
    y_pred = (errors_arr >= thr).astype(int)

    # Confusion matrix
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    results['true_positives'] = tp
    results['true_negatives'] = tn
    results['false_positives'] = fp
    results['false_negatives'] = fn

    # Metrics
    try:
        if SKLEARN_AVAILABLE:
            results['accuracy'] = float(accuracy_score(y_true, y_pred))
            results['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
            results['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
            results['f1_score'] = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            total = tp + tn + fp + fn
            results['accuracy'] = float((tp + tn) / total) if total > 0 else 0.0
            results['precision'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            results['recall'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            prec = results['precision']
            rec = results['recall']
            results['f1_score'] = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")

    # AUC
    try:
        if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
            results['auc'] = float(roc_auc_score(y_true, errors_arr))
    except Exception:
        results['auc'] = None

    return results


def evaluate_single_model(model_path: str, test_folder: str, model_name: str = "Model") -> Tuple[List[float], Dict[str, Any]]:
    """Evaluate a single model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test folder: {test_folder}")

    # Load model with fallback strategies
    model = load_model_safe(model_path)
    if model is None:
        logger.error(f"Failed to load model: {model_path}")
        return None, None

    # Load frames
    frames, frame_names = load_test_frames(test_folder, img_size=IMG_SIZE)
    logger.info(f"✓ Loaded {len(frames)} frames")

    ground_truth = get_ground_truth_labels(test_folder)
    if ground_truth is None:
        ground_truth = []
    logger.info(f"✓ Ground truth loaded ({int(np.sum(ground_truth)) if len(ground_truth)>0 else 0} anomalies)")

    if len(frames) == 0:
        logger.warning("No frames found for evaluation")
        return [], {}

    logger.info("Calculating reconstruction errors...")
    
    # Predict in batches
    try:
        CHUNK = 128
        reconstructed_all = []
        for start in range(0, len(frames), CHUNK):
            end = min(len(frames), start + CHUNK)
            batch = frames[start:end]
            preds = model.predict(batch, batch_size=min(32, len(batch)), verbose=0)
            reconstructed_all.append(preds)
        reconstructed_all = np.concatenate(reconstructed_all, axis=0)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        logger.info("Falling back to per-frame prediction...")
        reconstructed_all = []
        for i, frame in enumerate(frames):
            try:
                inp = np.reshape(frame, (1, IMG_SIZE, IMG_SIZE, 1))
                r = model.predict(inp, verbose=0)
                reconstructed_all.append(r[0])
            except Exception:
                logger.error(f"Predict failed on frame {i}")
                reconstructed_all.append(np.zeros_like(frame))
        reconstructed_all = np.array(reconstructed_all)

    # Calculate errors
    errors = calculate_reconstruction_error(frames, reconstructed_all)
    logger.info("✓ Reconstruction errors calculated")

    # Evaluate
    metrics = evaluate_anomaly_detection(errors, ground_truth, percentile=ANOMALY_PERCENTILE)

    # Print results
    logger.info("\nRESULTS")
    logger.info(f"Threshold (MSE): {metrics.get('threshold', 0.0):.6f}")
    logger.info(f"Accuracy: {metrics.get('accuracy', 0.0):.2%}")
    logger.info(f"Precision: {metrics.get('precision', 0.0):.2%}")
    logger.info(f"Recall: {metrics.get('recall', 0.0):.2%}")
    logger.info(f"F1-Score: {metrics.get('f1_score', 0.0):.4f}")
    auc_val = metrics.get('auc', None)
    logger.info(f"AUC-ROC: {auc_val:.4f}" if auc_val is not None else "AUC-ROC: N/A")
    
    # Confusion matrix
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {metrics.get('true_positives', 0)}, TN: {metrics.get('true_negatives', 0)}")
    logger.info(f"  FP: {metrics.get('false_positives', 0)}, FN: {metrics.get('false_negatives', 0)}")

    return [float(e) for e in errors], metrics


def compare_models():
    """Compare secure vs insecure models."""
    logger.info("\nFEDGUARD MODEL COMPARISON")
    test_folder = os.path.join(UCSD_PED1_TEST, TEST_FOLDERS.get('ped1', 'Test001'))

    if not os.path.exists(test_folder):
        logger.error(f"Test folder not found: {test_folder}")
        return

    secure_model_path = os.path.join(MODELS_DIR, 'global_model_secured.h5')
    insecure_model_path = os.path.join(MODELS_DIR, 'global_model_insecured.h5')

    if not os.path.exists(secure_model_path):
        logger.error(f"Secure model not found: {secure_model_path}")
        return
    if not os.path.exists(insecure_model_path):
        logger.error(f"Insecure model not found: {insecure_model_path}")
        return

    # Evaluate both models
    secure_errors, secure_metrics = evaluate_single_model(
        secure_model_path, test_folder, model_name="FedGuard (Secure)"
    )
    if secure_errors is None:
        logger.error("Failed to evaluate secure model")
        return

    attacked_errors, attacked_metrics = evaluate_single_model(
        insecure_model_path, test_folder, model_name="Attacked (No Defense)"
    )
    if attacked_errors is None:
        logger.error("Failed to evaluate attacked model")
        return

    # Compare metrics
    def safe_val(mdict, key):
        v = mdict.get(key)
        return float(v) if (v is not None) else 0.0

    metrics_comparison = {
        'Accuracy': (safe_val(secure_metrics, 'accuracy'), safe_val(attacked_metrics, 'accuracy')),
        'Precision': (safe_val(secure_metrics, 'precision'), safe_val(attacked_metrics, 'precision')),
        'Recall': (safe_val(secure_metrics, 'recall'), safe_val(attacked_metrics, 'recall')),
        'F1-Score': (safe_val(secure_metrics, 'f1_score'), safe_val(attacked_metrics, 'f1_score')),
        'AUC-ROC': (safe_val(secure_metrics, 'auc'), safe_val(attacked_metrics, 'auc')),
    }

    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    
    for metric_name, (secure_val, attacked_val) in metrics_comparison.items():
        if attacked_val == 0:
            improvement = float('inf') if secure_val > 0 else 0.0
            improvement_str = "inf" if improvement == float('inf') else f"{improvement:+.2f}%"
        else:
            improvement = ((secure_val - attacked_val) / (attacked_val + 1e-12)) * 100.0
            improvement_str = f"{improvement:+.2f}%"
        logger.info(f"{metric_name}: Secure={secure_val:.4f}, Attacked={attacked_val:.4f}, Improvement={improvement_str}")

    # Save results
    results = {
        'evaluation_date': datetime.now().isoformat(),
        'test_folder': test_folder,
        'secure_model': {
            'path': secure_model_path,
            'errors': secure_errors,
            'metrics': secure_metrics
        },
        'attacked_model': {
            'path': insecure_model_path,
            'errors': attacked_errors,
            'metrics': attacked_metrics
        },
        'comparison': {
            metric: {
                'secure': float(vals[0]),
                'attacked': float(vals[1]),
                'improvement_percent': (
                    float(((vals[0] - vals[1]) / (vals[1] + 1e-12)) * 100.0) 
                    if vals[1] != 0 else None
                )
            } for metric, vals in metrics_comparison.items()
        }
    }

    results_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    res = compare_models()
    if res:
        print("\n✅ Evaluation complete!")
        print(f"Results saved in: {METRICS_DIR}")