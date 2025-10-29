"""
Evaluation metrics for anomaly detection and federated learning.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

def calculate_reconstruction_error(original, reconstructed, method='mse'):
    """
    Calculate reconstruction error between original and reconstructed images.
    
    Args:
        method: 'mse', 'mae', or 'ssim'
    """
    if method == 'mse':
        return np.mean((original - reconstructed) ** 2)
    elif method == 'mae':
        return np.mean(np.abs(original - reconstructed))
    elif method == 'ssim':
        # Simplified SSIM (for full SSIM, use skimage)
        c1, c2 = 0.01**2, 0.03**2
        mu1, mu2 = np.mean(original), np.mean(reconstructed)
        sigma1, sigma2 = np.std(original), np.std(reconstructed)
        sigma12 = np.mean((original - mu1) * (reconstructed - mu2))
        
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        return 1 - ssim  # Return as error (lower is better)
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_anomaly_detection(errors, ground_truth, percentile=95):
    """
    Evaluate anomaly detection performance.
    
    Args:
        errors: Reconstruction errors for each frame
        ground_truth: Binary labels (0=normal, 1=anomaly)
        percentile: Threshold percentile
    
    Returns:
        dict with metrics
    """
    threshold = np.percentile(errors, percentile)
    predictions = (np.array(errors) > threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # ROC-AUC
    try:
        auc = roc_auc_score(ground_truth, errors)
        fpr, tpr, _ = roc_curve(ground_truth, errors)
    except:
        auc = 0
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
    """
    Calculate how much two models have diverged.
    Useful for measuring attack impact.
    """
    flat1 = np.concatenate([w.flatten() for w in weights1])
    flat2 = np.concatenate([w.flatten() for w in weights2])
    
    # Multiple divergence metrics
    l2_distance = np.linalg.norm(flat1 - flat2)
    cosine_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2))
    
    return {
        'l2_distance': float(l2_distance),
        'cosine_similarity': float(cosine_sim),
        'relative_change': float(l2_distance / np.linalg.norm(flat1))
    }