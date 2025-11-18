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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from config import *
from utils.data_loader import load_test_frames, get_ground_truth_labels
from utils.metrics import calculate_reconstruction_error, evaluate_anomaly_detection

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'evaluation.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
logger = logging.getLogger(__name__)

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        logger.warning("Could not set GPU memory growth")

def evaluate_single_model(model_path, test_folder, model_name="Model"):
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Test folder: {test_folder}")

    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return None, None

    frames, frame_names = load_test_frames(test_folder, img_size=IMG_SIZE)
    logger.info(f"✓ Loaded {len(frames)} frames")

    ground_truth = get_ground_truth_labels(test_folder)
    logger.info(f"✓ Ground truth loaded ({int(np.sum(ground_truth))} anomalies)")

    logger.info("Calculating reconstruction errors...")
    errors = []
    n_frames = len(frames)
    if n_frames == 0:
        logger.warning("No frames found for evaluation")
        return [], {}

    try:
        CHUNK = 128
        reconstructed_all = []
        for start in range(0, n_frames, CHUNK):
            end = min(n_frames, start + CHUNK)
            batch = frames[start:end]
            preds = model.predict(batch, batch_size=min(32, len(batch)), verbose=0)
            reconstructed_all.append(preds)
            if (start // CHUNK + 1) % 10 == 0:
                logger.info(f"  Predicted {end}/{n_frames} frames")
        reconstructed_all = np.concatenate(reconstructed_all, axis=0)
    except Exception:
        logger.exception("Batch prediction failed, falling back to per-frame")
        reconstructed_all = []
        for i, frame in enumerate(frames):
            try:
                inp = np.reshape(frame, (1, IMG_SIZE, IMG_SIZE, 1))
                r = model.predict(inp, verbose=0)
                reconstructed_all.append(r[0])
            except Exception:
                logger.exception(f"Predict failed on frame {i}")
                reconstructed_all.append(np.zeros_like(frame))
        reconstructed_all = np.array(reconstructed_all)

    try:
        if reconstructed_all.shape[0] != n_frames:
            logger.warning("Predicted frames count mismatch; trimming to common length")
            n_common = min(reconstructed_all.shape[0], n_frames)
            frames = frames[:n_common]
            reconstructed_all = reconstructed_all[:n_common]
        diffs = (frames - reconstructed_all) ** 2
        errors = np.mean(diffs.reshape(diffs.shape[0], -1), axis=1).tolist()
    except Exception:
        logger.exception("Vectorized MSE failed, falling back to per-frame MSE")
        errors = []
        for i in range(min(len(frames), len(reconstructed_all))):
            errors.append(float(np.mean((frames[i] - reconstructed_all[i]) ** 2)))

    logger.info("✓ Reconstruction errors calculated")

    metrics = evaluate_anomaly_detection(errors, ground_truth, percentile=ANOMALY_PERCENTILE)
    logger.info("\nRESULTS")
    logger.info(f"Threshold (MSE): {metrics['threshold']:.6f}")
    logger.info(f"Accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Precision: {metrics['precision']:.2%}")
    logger.info(f"Recall: {metrics['recall']:.2%}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC: {metrics['auc']:.4f}")

    return errors, metrics

def compare_models():
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

    secure_errors, secure_metrics = evaluate_single_model(secure_model_path, test_folder, model_name="FedGuard (Secure)")
    if secure_errors is None:
        logger.error("Failed to evaluate secure model"); return

    attacked_errors, attacked_metrics = evaluate_single_model(insecure_model_path, test_folder, model_name="Attacked (No Defense)")
    if attacked_errors is None:
        logger.error("Failed to evaluate attacked model"); return

    metrics_comparison = {
        'Accuracy': (secure_metrics['accuracy'], attacked_metrics['accuracy']),
        'Precision': (secure_metrics['precision'], attacked_metrics['precision']),
        'Recall': (secure_metrics['recall'], attacked_metrics['recall']),
        'F1-Score': (secure_metrics['f1_score'], attacked_metrics['f1_score']),
        'AUC-ROC': (secure_metrics['auc'], attacked_metrics['auc'])
    }

    for metric_name, (secure_val, attacked_val) in metrics_comparison.items():
        improvement = ((secure_val - attacked_val) / (attacked_val + 1e-8)) * 100
        logger.info(f"\n{metric_name}: Secure={secure_val:.4f}, Attacked={attacked_val:.4f}, Improvement={improvement:+.2f}%")

    results = {
        'evaluation_date': datetime.now().isoformat(),
        'test_folder': test_folder,
        'secure_model': {'path': secure_model_path, 'errors': [float(e) for e in secure_errors], 'metrics': secure_metrics},
        'attacked_model': {'path': insecure_model_path, 'errors': [float(e) for e in attacked_errors], 'metrics': attacked_metrics},
        'comparison': {metric: {'secure': float(vals[0]), 'attacked': float(vals[1]), 'improvement_percent': float(((vals[0] - vals[1]) / (vals[1] + 1e-8)) * 100)} for metric, vals in metrics_comparison.items()}
    }

    results_file = os.path.join(METRICS_DIR, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to: {results_file}")
    return {'secure_errors': secure_errors, 'attacked_errors': attacked_errors, 'secure_metrics': secure_metrics, 'attacked_metrics': attacked_metrics}

if __name__ == "__main__":
    results = compare_models()
    if results:
        print("\n✅ Evaluation complete!")
        print(f"Results saved in: {METRICS_DIR}")
