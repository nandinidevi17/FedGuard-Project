"""
Advanced security functions for FedGuard.
Implements multiple Byzantine-robust aggregation methods.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def flatten_weights(weights):
    """Converts a list of model weights into a single flat 1D array."""
    return np.concatenate([w.flatten() for w in weights])


def calculate_pairwise_similarity(weights_list):
    """
    Calculates pairwise cosine similarity between all client updates.
    Returns: similarity matrix (NxN)
    """
    flat_weights = [flatten_weights(w) for w in weights_list]
    similarity_matrix = cosine_similarity(flat_weights)
    return similarity_matrix


def detect_outliers_cosine(weights_list, threshold=0.5):
    """
    Original method: Detect outliers based on average cosine similarity.
    
    Returns:
        honest_indices: List of indices of honest clients
        similarities: Similarity scores for all clients
    """
    flat_weights = [flatten_weights(w) for w in weights_list]
    average_weights = np.mean(flat_weights, axis=0)
    
    similarities = []
    for fw in flat_weights:
        sim = cosine_similarity(fw.reshape(1, -1), average_weights.reshape(1, -1))
        similarities.append(sim[0][0])
    
    honest_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]
    
    return honest_indices, similarities


def detect_outliers_mad(weights_list, threshold=3.5):
    """
    IMPROVED METHOD: Median Absolute Deviation (MAD) for robust outlier detection.
    More resistant to multiple attackers.
    
    Args:
        weights_list: List of weight arrays from clients
        threshold: MAD threshold (3.5 is standard for outlier detection)
    
    Returns:
        honest_indices: List of indices of honest clients
        mad_scores: Modified Z-scores for all clients
    """
    # Calculate pairwise similarities
    flat_weights = [flatten_weights(w) for w in weights_list]
    n = len(flat_weights)
    
    # Calculate average pairwise similarity for each client
    avg_similarities = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i != j:
                sim = cosine_similarity(
                    flat_weights[i].reshape(1, -1),
                    flat_weights[j].reshape(1, -1)
                )[0][0]
                sims.append(sim)
        avg_similarities.append(np.mean(sims))
    
    # Calculate MAD-based scores
    median = np.median(avg_similarities)
    mad = np.median(np.abs(np.array(avg_similarities) - median))
    
    if mad == 0:
        # All values identical - use simpler method
        logger.warning("MAD is zero, falling back to mean-based detection")
        mean = np.mean(avg_similarities)
        std = np.std(avg_similarities)
        mad_scores = [(sim - mean) / (std + 1e-8) for sim in avg_similarities]
    else:
        # Modified Z-score (more robust than standard Z-score)
        mad_scores = [0.6745 * (sim - median) / mad for sim in avg_similarities]
    
    # Clients with very negative scores are outliers
    honest_indices = [i for i, score in enumerate(mad_scores) if score > -threshold]
    
    return honest_indices, mad_scores


def aggregate_median(weights_list):
    """
    Byzantine-robust aggregation using coordinate-wise median.
    More robust than mean, but slower.
    """
    aggregated = []
    num_layers = len(weights_list[0])
    
    for i in range(num_layers):
        layer_weights = np.array([client_weights[i] for client_weights in weights_list])
        median_layer = np.median(layer_weights, axis=0)
        aggregated.append(median_layer)
    
    return aggregated


def aggregate_trimmed_mean(weights_list, trim_ratio=0.2):
    """
    Trimmed mean: Remove top and bottom X% of values before averaging.
    Good balance between robustness and efficiency.
    """
    aggregated = []
    num_layers = len(weights_list[0])
    
    for i in range(num_layers):
        layer_weights = np.array([client_weights[i] for client_weights in weights_list])
        
        # Flatten, trim, average, reshape
        original_shape = layer_weights[0].shape
        flat_weights = layer_weights.reshape(len(weights_list), -1)
        
        # For each coordinate, remove outliers
        trimmed_mean = []
        for coord_idx in range(flat_weights.shape[1]):
            coord_values = flat_weights[:, coord_idx]
            sorted_vals = np.sort(coord_values)
            
            # Trim top and bottom
            trim_count = int(len(sorted_vals) * trim_ratio)
            if trim_count > 0:
                trimmed_vals = sorted_vals[trim_count:-trim_count]
            else:
                trimmed_vals = sorted_vals
            
            trimmed_mean.append(np.mean(trimmed_vals))
        
        trimmed_layer = np.array(trimmed_mean).reshape(original_shape)
        aggregated.append(trimmed_layer)
    
    return aggregated


def multi_krum(weights_list, num_attackers=1, num_selected=None):
    """
    Multi-Krum: Select clients with smallest sum of distances to k nearest neighbors.
    Highly robust but computationally expensive.
    
    Args:
        weights_list: List of client weight updates
        num_attackers: Expected number of Byzantine clients
        num_selected: Number of clients to select (default: n - num_attackers)
    """
    n = len(weights_list)
    if num_selected is None:
        num_selected = n - num_attackers
    
    flat_weights = [flatten_weights(w) for w in weights_list]
    
    # Calculate pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
            distances[i][j] = dist
            distances[j][i] = dist
    
    # For each client, sum distances to k nearest neighbors
    k = n - num_attackers - 2
    scores = []
    for i in range(n):
        sorted_distances = np.sort(distances[i])
        score = np.sum(sorted_distances[1:k+2])  # Exclude distance to self
        scores.append(score)
    
    # Select clients with smallest scores
    honest_indices = np.argsort(scores)[:num_selected]
    
    return honest_indices.tolist(), scores