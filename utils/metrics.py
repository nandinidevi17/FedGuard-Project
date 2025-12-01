"""
Comprehensive metrics for FedGuard evaluation.
Includes reconstruction errors, anomaly detection, and model divergence.
"""
import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def calculate_reconstruction_error(frames, reconstructed):
    """
    Calculate per-frame MSE between original and reconstructed frames.
    
    Args:
        frames: numpy array of original frames (N, H, W, C)
        reconstructed: numpy array of reconstructed frames (N, H, W, C)
    
    Returns:
        list of float: Per-frame MSE values
    """
    frames = np.asarray(frames, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    
    n = min(frames.shape[0], reconstructed.shape[0])
    if n == 0:
        return []
    
    frames = frames[:n]
    reconstructed = reconstructed[:n]
    
    try:
        diffs = (frames - reconstructed) ** 2
        errors = np.mean(diffs.reshape(diffs.shape[0], -1), axis=1)
        return [float(e) for e in errors]
    except Exception as e:
        logger.error(f"Error calculating reconstruction errors: {e}")
        out = []
        for i in range(n):
            try:
                out.append(float(np.mean((frames[i] - reconstructed[i]) ** 2)))
            except Exception:
                out.append(0.0)
        return out


def evaluate_anomaly_detection(errors, ground_truth, percentile=95):
    """
    Evaluate anomaly detection performance with comprehensive metrics.
    
    Args:
        errors: list or array of reconstruction errors
        ground_truth: list or array of binary labels (0=normal, 1=anomaly)
        percentile: threshold percentile for anomaly classification
    
    Returns:
        dict: Comprehensive evaluation metrics
    """
    errors = np.asarray(errors, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.int32)

    out = {
        'threshold': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'auc': None,
        'true_positives': 0,
        'true_negatives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'fpr': [],
        'tpr': [],
        'roc_thresholds': []
    }

    if errors.size == 0 or ground_truth.size == 0:
        logger.warning("Empty errors or ground_truth - returning empty metrics")
        return out

    # Calculate threshold
    try:
        thr = float(np.percentile(errors, percentile))
    except Exception:
        thr = float(np.max(errors)) if errors.size > 0 else 0.0
    out['threshold'] = thr

    # Make predictions
    preds = (errors > thr).astype(int)

    # Calculate confusion matrix values
    tp = int(np.sum((ground_truth == 1) & (preds == 1)))
    tn = int(np.sum((ground_truth == 0) & (preds == 0)))
    fp = int(np.sum((ground_truth == 0) & (preds == 1)))
    fn = int(np.sum((ground_truth == 1) & (preds == 0)))
    
    out['true_positives'] = tp
    out['true_negatives'] = tn
    out['false_positives'] = fp
    out['false_negatives'] = fn

    # Calculate metrics
    try:
        acc = float(accuracy_score(ground_truth, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, preds, average='binary', zero_division=0
        )
        out['accuracy'] = acc
        out['precision'] = float(precision)
        out['recall'] = float(recall)
        out['f1_score'] = float(f1)
    except Exception as e:
        logger.error(f"Error computing classification metrics: {e}")
        out['accuracy'] = out['precision'] = out['recall'] = out['f1_score'] = None

    # Check if we can compute ROC/AUC
    unique_labels = np.unique(ground_truth)
    if unique_labels.size < 2:
        logger.warning("Ground truth contains single class - cannot compute AUC/ROC")
        out['auc'] = None
        return out

    # Calculate ROC curve
    try:
        fpr, tpr, roc_th = roc_curve(ground_truth, errors)
        out['fpr'] = np.asarray(fpr).tolist()
        out['tpr'] = np.asarray(tpr).tolist()
        out['roc_thresholds'] = np.asarray(roc_th).tolist()
    except Exception as e:
        logger.error(f"Error computing ROC curve: {e}")

    # Calculate AUC
    try:
        auc_val = roc_auc_score(ground_truth, errors)
        out['auc'] = float(auc_val)
    except Exception as e:
        logger.error(f"Error computing AUC: {e}")
        out['auc'] = None

    return out


def calculate_model_divergence(weights_initial, weights_updated):
    """
    Calculate divergence metrics between two sets of model weights.
    Measures how much the model has changed during federated learning.
    
    Args:
        weights_initial: List of numpy arrays (initial model weights)
        weights_updated: List of numpy arrays (updated model weights)
    
    Returns:
        dict: {
            'l2_distance': float - Euclidean distance between weight vectors,
            'cosine_similarity': float - Cosine similarity (1=identical, -1=opposite)
        }
    """
    if weights_initial is None or weights_updated is None:
        logger.warning("Cannot calculate divergence - one or both weight sets are None")
        return {'l2_distance': None, 'cosine_similarity': None}
    
    if len(weights_initial) != len(weights_updated):
        logger.warning(f"Weight list length mismatch: {len(weights_initial)} vs {len(weights_updated)}")
        return {'l2_distance': None, 'cosine_similarity': None}
    
    try:
        # Flatten all weights into single vectors
        flat_initial = np.concatenate([w.flatten() for w in weights_initial])
        flat_updated = np.concatenate([w.flatten() for w in weights_updated])
        
        # Ensure same length
        if len(flat_initial) != len(flat_updated):
            logger.warning(f"Flattened weight length mismatch: {len(flat_initial)} vs {len(flat_updated)}")
            return {'l2_distance': None, 'cosine_similarity': None}
        
        # Calculate L2 distance (Euclidean norm of difference)
        l2_dist = float(np.linalg.norm(flat_initial - flat_updated))
        
        # Calculate cosine similarity
        dot_product = np.dot(flat_initial, flat_updated)
        norm_initial = np.linalg.norm(flat_initial)
        norm_updated = np.linalg.norm(flat_updated)
        
        if norm_initial > 0 and norm_updated > 0:
            cosine_sim = float(dot_product / (norm_initial * norm_updated))
        else:
            logger.warning("Zero norm encountered in cosine similarity calculation")
            cosine_sim = 0.0
        
        return {
            'l2_distance': l2_dist,
            'cosine_similarity': cosine_sim
        }
    
    except Exception as e:
        logger.error(f"Error calculating model divergence: {e}", exc_info=True)
        return {'l2_distance': None, 'cosine_similarity': None}


def calculate_weight_statistics(weights):
    """
    Calculate statistics about model weights.
    Useful for monitoring weight distributions during training.
    
    Args:
        weights: List of numpy arrays (model weights)
    
    Returns:
        dict: Statistics including mean, std, min, max, norm per layer
    """
    if weights is None or len(weights) == 0:
        return {}
    
    stats = {
        'num_layers': len(weights),
        'total_parameters': 0,
        'layer_stats': []
    }
    
    try:
        for i, w in enumerate(weights):
            w_flat = w.flatten()
            layer_stat = {
                'layer_index': i,
                'shape': list(w.shape),
                'num_params': int(np.prod(w.shape)),
                'mean': float(np.mean(w_flat)),
                'std': float(np.std(w_flat)),
                'min': float(np.min(w_flat)),
                'max': float(np.max(w_flat)),
                'l2_norm': float(np.linalg.norm(w_flat))
            }
            stats['layer_stats'].append(layer_stat)
            stats['total_parameters'] += layer_stat['num_params']
        
        return stats
    
    except Exception as e:
        logger.error(f"Error calculating weight statistics: {e}")
        return stats