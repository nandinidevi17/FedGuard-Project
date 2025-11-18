"""
Evaluation metrics for anomaly detection and federated learning.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)

def calculate_reconstruction_error(original, reconstructed, method='mse'):
    """
    Calculate reconstruction error between original and reconstructed images.
    """
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    if method == 'mse':
        return float(np.mean((original - reconstructed) ** 2))
    elif method == 'mae':
        return float(np.mean(np.abs(original - reconstructed)))
    elif method == 'ssim':
        c1, c2 = (0.01**2), (0.03**2)
        mu1, mu2 = np.mean(original), np.mean(reconstructed)
        sigma1, sigma2 = np.std(original), np.std(reconstructed)
        sigma12 = np.mean((original - mu1) * (reconstructed - mu2))
        denom1 = (mu1**2 + mu2**2 + c1)
        denom2 = (sigma1**2 + sigma2**2 + c2)
        if denom1 == 0 or denom2 == 0:
            return 1.0
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / (denom1 * denom2)
        if not np.isfinite(ssim):
            return 1.0
        return float(1.0 - ssim)
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluate_anomaly_detection(errors, ground_truth, percentile=95):
    errors = np.asarray(errors)
    ground_truth = np.asarray(ground_truth)
    if errors.size == 0 or ground_truth.size == 0:
        logger.warning("Empty errors or ground_truth in evaluate_anomaly_detection")
        return {
            'threshold': float('nan'),
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'auc': 0.0,
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'fpr': [0],
            'tpr': [0]
        }
    threshold = float(np.percentile(errors, percentile))
    predictions = (errors > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    try:
        auc = float(roc_auc_score(ground_truth, errors))
        fpr, tpr, _ = roc_curve(ground_truth, errors)
    except Exception:
        auc = 0.0
        fpr, tpr = [0], [0]
        logger.warning("Could not calculate ROC-AUC")
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }

def calculate_model_divergence(weights1, weights2):
    flat1 = np.concatenate([w.flatten() for w in weights1]) if weights1 else np.array([], dtype=np.float32)
    flat2 = np.concatenate([w.flatten() for w in weights2]) if weights2 else np.array([], dtype=np.float32)
    if flat1.size == 0 or flat2.size == 0:
        return {'l2_distance': float('nan'), 'cosine_similarity': float('nan'), 'relative_change': float('nan')}
    l2_distance = float(np.linalg.norm(flat1 - flat2))
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    cosine_sim = float(np.dot(flat1, flat2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else float('nan')
    relative_change = float(l2_distance / norm1) if norm1 > 0 else float('nan')
    if not np.isfinite(cosine_sim):
        cosine_sim = float('nan')
    return {'l2_distance': l2_distance, 'cosine_similarity': cosine_sim, 'relative_change': relative_change}
