# """
# Advanced security functions for FedGuard.
# Implements multiple Byzantine-robust aggregation methods.
# """
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import logging

# logger = logging.getLogger(__name__)

# def _to_float_array(x):
#     arr = np.asarray(x)
#     if arr.size == 0:
#         return np.array([], dtype=np.float32)
#     return arr.astype(np.float32).ravel()

# def flatten_weights(weights):
#     """
#     Flattens a list of numpy arrays into a single 1D float32 array.
#     Defensive: handles empty layers and NaNs/Infs.
#     """
#     if weights is None:
#         return np.array([], dtype=np.float32)
#     parts = []
#     for w in weights:
#         aw = np.asarray(w)
#         if aw.size == 0:
#             continue
#         aw = np.nan_to_num(aw, nan=0.0, posinf=1e9, neginf=-1e9)
#         parts.append(aw.flatten().astype(np.float32))
#     if not parts:
#         return np.array([], dtype=np.float32)
#     return np.concatenate(parts)

# def calculate_pairwise_similarity(weights_list):
#     """
#     Calculates pairwise cosine similarity between all client updates.
#     Returns: similarity matrix (NxN)
#     """
#     flat_weights = [flatten_weights(w) for w in weights_list]
#     if len(flat_weights) == 0:
#         return np.zeros((0,0))
#     maxlen = max((fw.size for fw in flat_weights), default=0)
#     mat = np.array([np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size < maxlen else fw for fw in flat_weights])
#     if mat.shape[1] == 0:
#         return np.zeros((len(flat_weights), len(flat_weights)))
#     sim = cosine_similarity(mat)
#     return sim

# def detect_outliers_cosine(weights_list, threshold=0.5):
#     flat_weights = [flatten_weights(w) for w in weights_list]
#     if len(flat_weights) == 0:
#         return [], []
#     nonzero = [fw for fw in flat_weights if fw.size>0]
#     if len(nonzero) == 0:
#         return list(range(len(flat_weights))), [0.0]*len(flat_weights)
#     maxlen = max(fw.size for fw in nonzero)
#     padded = [np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size<maxlen else fw for fw in flat_weights]
#     avg = np.mean([p for p in padded if p.size==maxlen], axis=0)
#     similarities = []
#     for p in padded:
#         denom = (np.linalg.norm(p) * np.linalg.norm(avg))
#         sim = float(np.dot(p, avg) / denom) if denom > 0 else 0.0
#         similarities.append(sim)
#     honest_indices = [i for i, s in enumerate(similarities) if s >= threshold]
#     return honest_indices, similarities

# def detect_outliers_mad(weights_list, threshold=3.5):
#     n = len(weights_list)
#     if n == 0:
#         return [], []
#     if n == 1:
#         return [0], [0.0]

#     flat_weights = [flatten_weights(w) for w in weights_list]
#     avg_similarities = []
#     for i in range(n):
#         sims = []
#         for j in range(n):
#             if i == j:
#                 continue
#             a = flat_weights[i]
#             b = flat_weights[j]
#             if a.size==0 or b.size==0:
#                 sims.append(0.0)
#                 continue
#             if a.size != b.size:
#                 L = max(a.size, b.size)
#                 a_p = np.pad(a, (0, L - a.size), 'constant')
#                 b_p = np.pad(b, (0, L - b.size), 'constant')
#             else:
#                 a_p, b_p = a, b
#             denom = (np.linalg.norm(a_p) * np.linalg.norm(b_p))
#             sim = float(np.dot(a_p, b_p) / denom) if denom > 0 else 0.0
#             sims.append(sim)
#         avg_similarities.append(float(np.mean(sims)) if len(sims)>0 else 0.0)

#     median = np.median(avg_similarities)
#     mad = np.median(np.abs(np.array(avg_similarities) - median))

#     if not np.isfinite(mad) or mad == 0:
#         std = np.std(avg_similarities)
#         eps = 1e-8
#         if std <= 0:
#             mad_scores = [0.0 for _ in avg_similarities]
#         else:
#             mad_scores = [(sim - np.mean(avg_similarities)) / (std + eps) for sim in avg_similarities]
#     else:
#         mad_scores = [0.6745 * (sim - median) / mad for sim in avg_similarities]

#     honest_indices = [i for i, score in enumerate(mad_scores) if score > -threshold]
#     return honest_indices, mad_scores

# def validate_weight_list(weights):
#     """
#     Validate a deserialized weight list. Returns (ok:bool, reason:str, norms:list)
#     """
#     if weights is None:
#         return False, "No weights", []
#     try:
#         norms = []
#         for i, w in enumerate(weights):
#             arr = np.asarray(w)
#             if arr.size == 0:
#                 return False, f"Empty layer {i}", []
#             if not np.all(np.isfinite(arr)):
#                 return False, f"NaN or Inf in layer {i}", []
#             max_abs = float(np.max(np.abs(arr)))
#             if max_abs > 1e9:
#                 return False, f"Layer {i} huge values (>{1e9})", []
#             norms.append(float(np.linalg.norm(arr)))
#         return True, "ok", norms
#     except Exception as e:
#         return False, f"exception: {e}", []

# def robust_mad(vals, eps=1e-8):
#     arr = np.asarray(vals, dtype=np.float64)
#     if arr.size == 0:
#         return np.nan
#     arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
#     arr = arr[~np.isnan(arr)]
#     if arr.size == 0:
#         return np.nan
#     med = np.median(arr)
#     mad = np.median(np.abs(arr - med))
#     if np.isfinite(mad) and mad > 0:
#         return mad
#     std = np.std(arr)
#     if std > 0:
#         return std
#     return eps

# def aggregate_median(weights_list):
#     aggregated = []
#     if not weights_list:
#         return []
#     num_layers = len(weights_list[0])
#     for i in range(num_layers):
#         layer_stack = np.stack([np.nan_to_num(client_weights[i], nan=0.0, posinf=1e9, neginf=-1e9) for client_weights in weights_list], axis=0)
#         median_layer = np.median(layer_stack, axis=0)
#         aggregated.append(median_layer)
#     return aggregated

# def aggregate_trimmed_mean(weights_list, trim_ratio=0.2):
#     aggregated = []
#     if not weights_list:
#         return []
#     num_layers = len(weights_list[0])
#     m = len(weights_list)
#     trim_count = int(m * trim_ratio)
#     for i in range(num_layers):
#         layer_stack = np.stack([np.nan_to_num(w[i], nan=0.0, posinf=1e9, neginf=-1e9) for w in weights_list], axis=0)
#         flat = layer_stack.reshape(m, -1)
#         if trim_count <= 0:
#             mean_flat = np.mean(flat, axis=0)
#         else:
#             sorted_vals = np.sort(flat, axis=0)
#             if 2*trim_count >= m:
#                 mean_flat = np.mean(sorted_vals, axis=0)
#             else:
#                 mean_flat = np.mean(sorted_vals[trim_count: m-trim_count, :], axis=0)
#         aggregated.append(mean_flat.reshape(layer_stack.shape[1:]))
#     return aggregated

# def multi_krum(weights_list, num_attackers=1, num_selected=None):
#     n = len(weights_list)
#     if n == 0:
#         return [], []
#     if num_selected is None:
#         num_selected = max(1, n - num_attackers)
#     flat_weights = [flatten_weights(w) for w in weights_list]
#     maxlen = max((fw.size for fw in flat_weights), default=0)
#     mats = [np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size < maxlen else fw for fw in flat_weights]
#     distances = np.zeros((n, n), dtype=np.float32)
#     for i in range(n):
#         for j in range(i+1, n):
#             d = float(np.linalg.norm(mats[i] - mats[j]))
#             distances[i, j] = d
#             distances[j, i] = d
#     k = max(1, n - num_attackers - 2)
#     scores = []
#     for i in range(n):
#         dists = np.sort(distances[i])
#         k_local = min(k, n-1)
#         score = float(np.sum(dists[1:1+k_local])) if k_local > 0 else float(np.sum(dists[1:]))
#         scores.append(score)
#     honest_indices = np.argsort(scores)[:num_selected].tolist()
#     return honest_indices, scores

# import numpy as np
# import tensorflow as tf
# from config import *

# def validate_update_quality(global_model, client_weights, validation_data, max_loss_threshold=10.0):
#     """
#     Validates an update by checking if it destroys model performance.
#     Returns: (is_valid, loss)
#     """
#     # 1. Create a temp model to test the weights
#     temp_model = tf.keras.models.clone_model(global_model)
#     temp_model.compile(optimizer='adam', loss='mse')
    
#     # 2. Apply the weights
#     try:
#         temp_model.set_weights(client_weights)
#     except Exception:
#         return False, 999.9

#     # 3. Evaluate on a small batch of validation data
#     # (Using a single batch is fast and sufficient to catch random noise)
#     loss = temp_model.evaluate(validation_data, verbose=0)
    
#     # 4. Check if loss is acceptable
#     # Random noise usually causes loss to jump to > 1000.0
#     if np.isnan(loss) or np.isinf(loss) or loss > max_loss_threshold:
#         return False, loss
    
#     return True, loss

"""
Advanced security functions for FedGuard.
Implements multiple Byzantine-robust aggregation methods.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# NEW: Performance Validation (Fix applied here)
# ---------------------------------------------------------
def validate_update_quality(global_model, client_weights, validation_data, max_loss_threshold=10.0):
    """
    Validates an update by checking if it destroys model performance.
    Returns: (is_valid, loss)
    """
    # 0. Check if validation data exists
    if validation_data is None or len(validation_data) == 0:
        logger.error("Validation data is empty! Cannot validate update quality.")
        return False, 999.9

    try:
        # 1. Create a temp model to test the weights
        temp_model = tf.keras.models.clone_model(global_model)
        temp_model.compile(optimizer='adam', loss='mse')
        
        # 2. Apply the weights
        temp_model.set_weights(client_weights)

        # 3. Evaluate on validation data
        # For Autoencoder: Input (x) is also the Target (y)
        results = temp_model.evaluate(validation_data, validation_data, verbose=0)
        
        # 4. Handle List vs Float return type (THE FIX)
        if isinstance(results, list):
            if len(results) > 0:
                loss = results[0] 
            else:
                loss = 999.9 # Empty list returned
        else:
            loss = results # It's already a float

        # 5. Check if loss is acceptable
        # Random noise usually causes loss to jump to > 0.5 (for normalized 0-1 data)
        if np.isnan(loss) or np.isinf(loss) or loss > max_loss_threshold:
            return False, float(loss)
        
        return True, float(loss)

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        # If evaluation crashes (e.g., shape mismatch), reject the update
        return False, 999.9

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def _to_float_array(x):
    arr = np.asarray(x)
    if arr.size == 0:
        return np.array([], dtype=np.float32)
    return arr.astype(np.float32).ravel()

def flatten_weights(weights):
    """
    Flattens a list of numpy arrays into a single 1D float32 array.
    Defensive: handles empty layers and NaNs/Infs.
    """
    if weights is None:
        return np.array([], dtype=np.float32)
    parts = []
    for w in weights:
        aw = np.asarray(w)
        if aw.size == 0:
            continue
        aw = np.nan_to_num(aw, nan=0.0, posinf=1e9, neginf=-1e9)
        parts.append(aw.flatten().astype(np.float32))
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts)

def calculate_pairwise_similarity(weights_list):
    """
    Calculates pairwise cosine similarity between all client updates.
    Returns: similarity matrix (NxN)
    """
    flat_weights = [flatten_weights(w) for w in weights_list]
    if len(flat_weights) == 0:
        return np.zeros((0,0))
    maxlen = max((fw.size for fw in flat_weights), default=0)
    mat = np.array([np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size < maxlen else fw for fw in flat_weights])
    if mat.shape[1] == 0:
        return np.zeros((len(flat_weights), len(flat_weights)))
    sim = cosine_similarity(mat)
    return sim

# ---------------------------------------------------------
# Outlier Detection Algorithms
# ---------------------------------------------------------

def detect_outliers_cosine(weights_list, threshold=0.5):
    flat_weights = [flatten_weights(w) for w in weights_list]
    if len(flat_weights) == 0:
        return [], []
    nonzero = [fw for fw in flat_weights if fw.size>0]
    if len(nonzero) == 0:
        return list(range(len(flat_weights))), [0.0]*len(flat_weights)
    maxlen = max(fw.size for fw in nonzero)
    padded = [np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size<maxlen else fw for fw in flat_weights]
    avg = np.mean([p for p in padded if p.size==maxlen], axis=0)
    similarities = []
    for p in padded:
        denom = (np.linalg.norm(p) * np.linalg.norm(avg))
        sim = float(np.dot(p, avg) / denom) if denom > 0 else 0.0
        similarities.append(sim)
    honest_indices = [i for i, s in enumerate(similarities) if s >= threshold]
    return honest_indices, similarities

def detect_outliers_mad(weights_list, threshold=3.5):
    n = len(weights_list)
    if n == 0:
        return [], []
    if n == 1:
        return [0], [0.0]

    flat_weights = [flatten_weights(w) for w in weights_list]
    avg_similarities = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            a = flat_weights[i]
            b = flat_weights[j]
            if a.size==0 or b.size==0:
                sims.append(0.0)
                continue
            if a.size != b.size:
                L = max(a.size, b.size)
                a_p = np.pad(a, (0, L - a.size), 'constant')
                b_p = np.pad(b, (0, L - b.size), 'constant')
            else:
                a_p, b_p = a, b
            denom = (np.linalg.norm(a_p) * np.linalg.norm(b_p))
            sim = float(np.dot(a_p, b_p) / denom) if denom > 0 else 0.0
            sims.append(sim)
        avg_similarities.append(float(np.mean(sims)) if len(sims)>0 else 0.0)

    median = np.median(avg_similarities)
    mad = np.median(np.abs(np.array(avg_similarities) - median))

    if not np.isfinite(mad) or mad == 0:
        std = np.std(avg_similarities)
        eps = 1e-8
        if std <= 0:
            mad_scores = [0.0 for _ in avg_similarities]
        else:
            mad_scores = [(sim - np.mean(avg_similarities)) / (std + eps) for sim in avg_similarities]
    else:
        mad_scores = [0.6745 * (sim - median) / mad for sim in avg_similarities]

    honest_indices = [i for i, score in enumerate(mad_scores) if score > -threshold]
    return honest_indices, mad_scores

def validate_weight_list(weights):
    """
    Validate a deserialized weight list. Returns (ok:bool, reason:str, norms:list)
    """
    if weights is None:
        return False, "No weights", []
    try:
        norms = []
        for i, w in enumerate(weights):
            arr = np.asarray(w)
            if arr.size == 0:
                return False, f"Empty layer {i}", []
            if not np.all(np.isfinite(arr)):
                return False, f"NaN or Inf in layer {i}", []
            max_abs = float(np.max(np.abs(arr)))
            if max_abs > 1e9:
                return False, f"Layer {i} huge values (>{1e9})", []
            norms.append(float(np.linalg.norm(arr)))
        return True, "ok", norms
    except Exception as e:
        return False, f"exception: {e}", []

def robust_mad(vals, eps=1e-8):
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return np.nan
    arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if np.isfinite(mad) and mad > 0:
        return mad
    std = np.std(arr)
    if std > 0:
        return std
    return eps

def aggregate_median(weights_list):
    aggregated = []
    if not weights_list:
        return []
    num_layers = len(weights_list[0])
    for i in range(num_layers):
        layer_stack = np.stack([np.nan_to_num(client_weights[i], nan=0.0, posinf=1e9, neginf=-1e9) for client_weights in weights_list], axis=0)
        median_layer = np.median(layer_stack, axis=0)
        aggregated.append(median_layer)
    return aggregated

def aggregate_trimmed_mean(weights_list, trim_ratio=0.2):
    aggregated = []
    if not weights_list:
        return []
    num_layers = len(weights_list[0])
    m = len(weights_list)
    trim_count = int(m * trim_ratio)
    for i in range(num_layers):
        layer_stack = np.stack([np.nan_to_num(w[i], nan=0.0, posinf=1e9, neginf=-1e9) for w in weights_list], axis=0)
        flat = layer_stack.reshape(m, -1)
        if trim_count <= 0:
            mean_flat = np.mean(flat, axis=0)
        else:
            sorted_vals = np.sort(flat, axis=0)
            if 2*trim_count >= m:
                mean_flat = np.mean(sorted_vals, axis=0)
            else:
                mean_flat = np.mean(sorted_vals[trim_count: m-trim_count, :], axis=0)
        aggregated.append(mean_flat.reshape(layer_stack.shape[1:]))
    return aggregated

def multi_krum(weights_list, num_attackers=1, num_selected=None):
    n = len(weights_list)
    if n == 0:
        return [], []
    if num_selected is None:
        num_selected = max(1, n - num_attackers)
    flat_weights = [flatten_weights(w) for w in weights_list]
    maxlen = max((fw.size for fw in flat_weights), default=0)
    mats = [np.pad(fw, (0, maxlen-fw.size), 'constant') if fw.size < maxlen else fw for fw in flat_weights]
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i+1, n):
            d = float(np.linalg.norm(mats[i] - mats[j]))
            distances[i, j] = d
            distances[j, i] = d
    k = max(1, n - num_attackers - 2)
    scores = []
    for i in range(n):
        dists = np.sort(distances[i])
        k_local = min(k, n-1)
        score = float(np.sum(dists[1:1+k_local])) if k_local > 0 else float(np.sum(dists[1:]))
        scores.append(score)
    honest_indices = np.argsort(scores)[:num_selected].tolist()
    return honest_indices, scores