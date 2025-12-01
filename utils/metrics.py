import numpy as np
import logging
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)

def calculate_reconstruction_error(frames, reconstructed):
    frames = np.asarray(frames, dtype=np.float32)
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    if frames.ndim == 3 and reconstructed.ndim == 3:
        pass
    elif frames.ndim == 4 and reconstructed.ndim == 4:
        pass
    else:
        try:
            frames = frames.reshape((frames.shape[0],) + frames.shape[1:])
            reconstructed = reconstructed.reshape((reconstructed.shape[0],) + reconstructed.shape[1:])
        except Exception:
            frames = np.asarray(frames)
            reconstructed = np.asarray(reconstructed)
    n = min(frames.shape[0], reconstructed.shape[0])
    if n == 0:
        return []
    frames = frames[:n]
    reconstructed = reconstructed[:n]
    try:
        diffs = (frames - reconstructed) ** 2
        errors = np.mean(diffs.reshape(diffs.shape[0], -1), axis=1)
        return [float(e) for e in errors]
    except Exception:
        out = []
        for i in range(n):
            try:
                out.append(float(np.mean((frames[i] - reconstructed[i]) ** 2)))
            except Exception:
                out.append(float(np.mean(np.zeros_like(frames[i]))))
        return out

def evaluate_anomaly_detection(errors, ground_truth, percentile=95):
    errors = np.asarray(errors, dtype=np.float32)
    ground_truth = np.asarray(ground_truth, dtype=np.int32)

    out = {
        'threshold': None,
        'accuracy': None,
        'precision': None,
        'recall': None,
        'f1_score': None,
        'auc': None,
        'fpr': [],
        'tpr': [],
        'roc_thresholds': []
    }

    if errors.size == 0 or ground_truth.size == 0:
        logger.warning("Empty errors or ground_truth passed to evaluate_anomaly_detection")
        return out

    try:
        thr = float(np.percentile(errors, percentile))
    except Exception:
        thr = float(np.max(errors)) if errors.size else 0.0
    out['threshold'] = thr

    preds = (errors > thr).astype(int)

    try:
        acc = float(accuracy_score(ground_truth, preds))
        precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, preds, average='binary', zero_division=0)
        out['accuracy'] = acc
        out['precision'] = float(precision)
        out['recall'] = float(recall)
        out['f1_score'] = float(f1)
    except Exception:
        logger.exception("Failed to compute accuracy/PRF metrics")
        out['accuracy'] = out['precision'] = out['recall'] = out['f1_score'] = None

    unique_labels = np.unique(ground_truth)
    if unique_labels.size < 2:
        logger.warning("Could not calculate ROC-AUC (ground truth contains a single class). Returning auc=None.")
        out['auc'] = None
        out['fpr'] = []
        out['tpr'] = []
        out['roc_thresholds'] = []
        return out

    try:
        fpr, tpr, roc_th = roc_curve(ground_truth, errors)
        out['fpr'] = np.asarray(fpr).tolist()
        out['tpr'] = np.asarray(tpr).tolist()
        out['roc_thresholds'] = np.asarray(roc_th).tolist()
    except Exception:
        logger.exception("roc_curve failed")
        out['fpr'] = []
        out['tpr'] = []
        out['roc_thresholds'] = []

    try:
        auc_val = roc_auc_score(ground_truth, errors)
        out['auc'] = float(auc_val)
    except Exception:
        logger.exception("roc_auc_score failed")
        out['auc'] = None

    return out
