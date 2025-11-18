"""
FedGuard Secure Server (updated).
- Reads raw bytes weight payloads from Kafka
- Extracts metadata from message headers (if present)
- Deserializes weights with numpy.load after zlib.decompress
- Security & aggregation code preserved (hooks in utils.security)
"""
import os
import sys
import json
import numpy as np
import logging
import time
import io
import zlib
from datetime import datetime
from kafka import KafkaConsumer

# Add parent dir to path to import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
# Import your existing utils (ensure these exist)
from utils.security import (
    detect_outliers_cosine,
    detect_outliers_mad,
    aggregate_median,
    aggregate_trimmed_mean,
    multi_krum,
    validate_weight_list,
    robust_mad
)
from utils.metrics import calculate_model_divergence

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'fedguard_server.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fedguard_server")

def deserialize_weights_from_bytes(b):
    """
    Inverse of serialize_weights_to_bytes in client.
    Returns: list of numpy arrays (weights) or raises ValueError.
    Supports:
      - zlib-compressed npz bytes (preferred)
      - raw npz bytes
      - JSON-encoded payloads with lists (fallback)
    """
    import json
    if b is None or len(b) == 0:
        raise ValueError("Empty payload")

    # Try zlib + numpy load (preferred)
    try:
        decompressed = zlib.decompress(b)
        buf = io.BytesIO(decompressed)
        buf.seek(0)
        npz = np.load(buf, allow_pickle=True)
        arrays = [npz[f] for f in npz.files]
        return arrays
    except zlib.error:
        # maybe uncompressed npz
        try:
            buf = io.BytesIO(b)
            buf.seek(0)
            npz = np.load(buf, allow_pickle=True)
            arrays = [npz[f] for f in npz.files]
            return arrays
        except Exception:
            pass
    except Exception:
        pass

    # JSON fallback
    try:
        payload = b.decode('utf-8')
        parsed = json.loads(payload)
        if isinstance(parsed, dict) and 'weights' in parsed:
            wlists = parsed['weights']
        elif isinstance(parsed, list):
            wlists = parsed
        else:
            raise ValueError("Unknown JSON payload structure for weights")
        arrays = [np.array(w, dtype=np.float32) for w in wlists]
        return arrays
    except Exception as e:
        raise ValueError(f"Failed to deserialize weights payload: {e}")

def extract_metadata_from_headers(headers):
    """
    kafka-python returns headers as list of tuples: [(k, v), ...]
    We expect a header 'metadata' with JSON bytes.
    """
    if not headers:
        return {}
    for k, v in headers:
        try:
            if k == 'metadata' and v is not None:
                return json.loads(v.decode('utf-8'))
        except Exception:
            continue
    return {}

def main():
    try:
        logger.info("="*60)
        logger.info("FEDGUARD SECURE SERVER")
        logger.info("="*60)
        logger.info(f"Defense method: {DEFENSE_METHOD}")
        logger.info(f"Aggregation method: {AGGREGATION_METHOD}")
        logger.info(f"Expected clients: {NUM_CLIENTS}")
        logger.info(f"Min honest clients: {MIN_HONEST_CLIENTS}")

        model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
        import tensorflow as tf
        global_model = None
        initial_weights = None
        if os.path.exists(model_path):
            logger.info(f"Loading initial model from {model_path}...")
            global_model = tf.keras.models.load_model(model_path)
            initial_weights = global_model.get_weights()
            logger.info("[OK] Model loaded successfully")
        else:
            logger.warning(f"Initial model not found at {model_path}. Server will still run but cannot compute divergence.")

        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        logger.info(f"Connecting to Kafka at {KAFKA_SERVER} (topic: {KAFKA_TOPIC})...")
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda m: m,   # raw bytes; metadata in headers
            auto_offset_reset='earliest',
            consumer_timeout_ms=KAFKA_TIMEOUT,
            fetch_max_bytes=KAFKA_FETCH_MAX_BYTES,
            max_partition_fetch_bytes=KAFKA_FETCH_MAX_BYTES,
            enable_auto_commit=True
        )
        logger.info("[OK] Kafka consumer initialized")
        logger.info("\nSERVER RUNNING - Waiting for client updates\n")

        client_updates = {}
        round_number = 0
        total_updates_received = 0
        total_attacks_blocked = 0

        updates_processed = False
        for message in consumer:
            updates_processed = True
            try:
                total_updates_received += 1
                headers = message.headers if hasattr(message, 'headers') else None
                metadata = extract_metadata_from_headers(headers)
                key = message.key.decode('utf-8') if message.key else f"unknown_{total_updates_received}"
                logger.info(f"\n[Round {round_number + 1}] Received message from key: {key}")

                if metadata:
                    logger.info(f"  Metadata: update #{metadata.get('update_number', 'N/A')}, loss: {metadata.get('training_loss', 'N/A')}")

                raw_bytes = message.value  # bytes
                if raw_bytes is None or len(raw_bytes) == 0:
                    logger.warning("  Empty payload received (maybe metadata-only). Storing metadata only.")
                    client_updates[key] = {
                        'weights': None,
                        'metadata': metadata
                    }
                else:
                    try:
                        received_weights = deserialize_weights_from_bytes(raw_bytes)
                        # Validate weights
                        valid, reason, norms = validate_weight_list(received_weights)
                        if not valid:
                            logger.warning(f"  Update from {key} failed validation: {reason}. Storing metadata only.")
                            client_updates[key] = {'weights': None, 'metadata': metadata, 'validation_reason': reason}
                        else:
                            client_updates[key] = {
                                'weights': received_weights,
                                'metadata': metadata,
                                'norms': norms
                            }
                            logger.info(f"  Received weights (layers: {len(received_weights)}) from {key}")
                    except Exception as e:
                        logger.error(f"  Failed to deserialize weights from {key}: {e}", exc_info=True)
                        client_updates[key] = {
                            'weights': None,
                            'metadata': metadata
                        }

                num_with_weights = len([1 for v in client_updates.values() if v and v.get('weights') is not None])
                logger.info(f"  Updates collected: {num_with_weights}/{NUM_CLIENTS}")

                # Only aggregate when we have NUM_CLIENTS entries in client_updates (weight or metadata)
                if len(client_updates) >= NUM_CLIENTS:
                    round_number += 1
                    logger.info("\n" + "="*60)
                    logger.info(f"ROUND {round_number} - AGGREGATION")
                    logger.info("="*60)

                    client_ids = list(client_updates.keys())
                    client_weights_list = [client_updates[cid]['weights'] for cid in client_ids]

                    weightful_indices = [i for i, w in enumerate(client_weights_list) if w is not None]
                    if not weightful_indices:
                        logger.error("No valid weight payloads to aggregate this round. Clearing updates and continuing.")
                        client_updates = {}
                        continue

                    selected_weights = [client_weights_list[i] for i in weightful_indices]

                    logger.info("\n[SECURITY] Running FedGuard security check...")
                    if DEFENSE_METHOD == "cosine":
                        honest_indices_rel, scores = detect_outliers_cosine(selected_weights, threshold=SIMILARITY_THRESHOLD)
                        score_name = "Cosine Similarity"
                    elif DEFENSE_METHOD == "mad":
                        honest_indices_rel, scores = detect_outliers_mad(selected_weights, threshold=MAD_THRESHOLD)
                        score_name = "MAD Score"
                    elif DEFENSE_METHOD == "krum":
                        num_attackers = max(0, NUM_CLIENTS - MIN_HONEST_CLIENTS)
                        honest_indices_rel, scores = multi_krum(selected_weights, num_attackers=num_attackers)
                        score_name = "Krum Score"
                    else:
                        honest_indices_rel = list(range(len(selected_weights)))
                        scores = [1.0] * len(selected_weights)
                        score_name = "Score"

                    # Map back to global indices
                    honest_indices = [weightful_indices[i] for i in honest_indices_rel]

                    logger.info("\nClient Security Scores:")
                    honest_clients = []
                    malicious_clients = []
                    for i, cid in enumerate(client_ids):
                        if i in weightful_indices:
                            j = weightful_indices.index(i)
                            score = scores[j] if j < len(scores) else None
                            is_honest = i in honest_indices
                            status = "[HONEST]" if is_honest else "[MALICIOUS]"
                            safe_score = float(score) if score is not None and np.isfinite(score) else None
                            logger.info(f"  {cid:30s} {score_name}: {safe_score} - {status}")
                            if is_honest:
                                honest_clients.append(cid)
                            else:
                                malicious_clients.append(cid)
                                total_attacks_blocked += 1
                        else:
                            logger.info(f"  {cid:30s} No weights received (skipped)")
                            # treat missing as suspicious but do not auto-block
                    logger.info(f"\nVerdict: {len(honest_clients)} honest, {len(malicious_clients)} malicious")
                    if malicious_clients:
                        logger.warning(f"[WARNING] Blocked malicious/invalid clients: {', '.join(malicious_clients)}")

                    if len(honest_clients) < MIN_HONEST_CLIENTS:
                        logger.error(f"[ERROR] Insufficient honest clients ({len(honest_clients)} < {MIN_HONEST_CLIENTS}) - skipping round")
                        client_updates = {}
                        continue

                    honest_weights = [client_updates[cid]['weights'] for cid in honest_clients if client_updates[cid]['weights'] is not None]

                    if AGGREGATION_METHOD == "mean":
                        averaged_weights = []
                        num_layers = len(honest_weights[0])
                        for li in range(num_layers):
                            layer_w = [cw[li] for cw in honest_weights]
                            averaged_layer = np.mean(layer_w, axis=0)
                            averaged_weights.append(averaged_layer)
                    elif AGGREGATION_METHOD == "median":
                        averaged_weights = aggregate_median(honest_weights)
                    elif AGGREGATION_METHOD == "trimmed_mean":
                        averaged_weights = aggregate_trimmed_mean(honest_weights, trim_ratio=0.2)
                    else:
                        raise ValueError(f"Unknown aggregation method: {AGGREGATION_METHOD}")

                    if global_model is not None and initial_weights is not None:
                        divergence = calculate_model_divergence(initial_weights, averaged_weights)
                        logger.info(f"\nModel divergence from initial:")
                        logger.info(f"  L2 distance: {divergence.get('l2_distance', float('nan')):.6f}")
                        logger.info(f"  Cosine similarity: {divergence.get('cosine_similarity', float('nan')):.6f}")
                        global_model.set_weights(averaged_weights)
                        save_path = os.path.join(MODELS_DIR, 'global_model_secured.h5')
                        global_model.save(save_path)
                        logger.info(f"\n[OK] Global model updated and saved to: {save_path}")
                    else:
                        logger.info("Global model not loaded; skipping model update and divergence calc.")

                    round_stats = {
                        'round': round_number,
                        'timestamp': datetime.now().isoformat(),
                        'clients_participated': client_ids,
                        'honest_clients': honest_clients,
                        'malicious_clients': malicious_clients,
                        'security_scores': {client_ids[i]: (float(scores[weightful_indices.index(i)]) if i in weightful_indices and weightful_indices.index(i) < len(scores) else None) for i in range(len(client_ids))},
                        'defense_method': DEFENSE_METHOD,
                        'aggregation_method': AGGREGATION_METHOD
                    }
                    stats_file = os.path.join(METRICS_DIR, f'round_{round_number:03d}_stats.json')
                    with open(stats_file, 'w') as f:
                        json.dump(round_stats, f, indent=2)
                    logger.info(f"Round statistics saved to: {stats_file}")
                    logger.info("="*60 + "\n")

                    client_updates = {}

                    if round_number >= NUM_FEDERATED_ROUNDS:
                        logger.info(f"\n[OK] Completed {NUM_FEDERATED_ROUNDS} rounds - stopping server")
                        break

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                continue

        if not updates_processed:
            logger.warning("Kafka consumer timed out - no messages received.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - stopping server")
    except Exception as e:
        logger.error(f"FATAL ERROR (server): {e}", exc_info=True)
    finally:
        logger.info("\n" + "="*60)
        logger.info("SERVER SHUTDOWN SUMMARY")
        logger.info("="*60)
        logger.info(f"Total rounds completed: {round_number}")
        logger.info(f"Total updates received: {total_updates_received}")
        logger.info(f"Total attacks blocked: {total_attacks_blocked}")
        if total_updates_received > 0:
            block_rate = (total_attacks_blocked / total_updates_received) * 100
            logger.info(f"Attack detection rate: {block_rate:.1f}%")
        logger.info("="*60)
        try:
            consumer.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
