"""
FedGuard Secure Server with Advanced Byzantine Defense.
Implements multiple defense mechanisms and comprehensive logging.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer
import logging
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.security import (
    detect_outliers_cosine,
    detect_outliers_mad,
    aggregate_median,
    aggregate_trimmed_mean,
    multi_krum
)
from utils.metrics import calculate_model_divergence

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'fedguard_server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
DEFENSE_METHOD = "mad"  # Options: "cosine", "mad", "krum", "median", "trimmed_mean"
AGGREGATION_METHOD = "mean"  # Options: "mean", "median", "trimmed_mean"
# -------------------

logger.info("="*60)
logger.info("FEDGUARD SECURE SERVER")
logger.info("="*60)
logger.info(f"Defense method: {DEFENSE_METHOD}")
logger.info(f"Aggregation method: {AGGREGATION_METHOD}")
logger.info(f"Expected clients: {NUM_CLIENTS}")
logger.info(f"Min honest clients: {MIN_HONEST_CLIENTS}")

try:
    # Load initial model
    model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
    logger.info(f"\nLoading initial model from {model_path}...")
    global_model = tf.keras.models.load_model(model_path)
    initial_weights = global_model.get_weights()
    logger.info("✓ Model loaded successfully")
    
    # Initialize Kafka consumer
    logger.info(f"\nConnecting to Kafka at {KAFKA_SERVER}...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        consumer_timeout_ms=KAFKA_TIMEOUT
    )
    logger.info("✓ Kafka consumer initialized")
    
    logger.info("\n" + "="*60)
    logger.info("SERVER RUNNING - Waiting for client updates")
    logger.info("="*60 + "\n")
    
    client_updates = {}
    round_number = 0
    total_updates_received = 0
    total_attacks_blocked = 0
    
    for message in consumer:
        try:
            update_data = message.value
            client_id = update_data['client_id']
            total_updates_received += 1
            
            logger.info(f"\n[Round {round_number + 1}] Received update from: {client_id}")
            
            # Log metadata if available
            if 'metadata' in update_data:
                meta = update_data['metadata']
                logger.info(f"  Update #{meta.get('update_number', 'N/A')}")
                if 'training_loss' in meta:
                    logger.info(f"  Training loss: {meta['training_loss']:.6f}")
            
            # Store update
            received_weights = [np.array(w) for w in update_data['weights']]
            client_updates[client_id] = {
                'weights': received_weights,
                'metadata': update_data.get('metadata', {})
            }
            
            logger.info(f"  Updates collected: {len(client_updates)}/{NUM_CLIENTS}")
            
            # When we have enough updates, aggregate
            if len(client_updates) >= NUM_CLIENTS:
                round_number += 1
                logger.info("\n" + "="*60)
                logger.info(f"ROUND {round_number} - AGGREGATION")
                logger.info("="*60)
                
                client_ids = list(client_updates.keys())
                client_weights_list = [client_updates[cid]['weights'] for cid in client_ids]
                
                # SECURITY CHECK
                logger.info("\n🛡️  Running FedGuard security check...")
                logger.info(f"Defense method: {DEFENSE_METHOD}")
                
                if DEFENSE_METHOD == "cosine":
                    honest_indices, scores = detect_outliers_cosine(
                        client_weights_list,
                        threshold=SIMILARITY_THRESHOLD
                    )
                    score_name = "Cosine Similarity"
                    
                elif DEFENSE_METHOD == "mad":
                    honest_indices, scores = detect_outliers_mad(
                        client_weights_list,
                        threshold=MAD_THRESHOLD
                    )
                    score_name = "MAD Score"
                    
                elif DEFENSE_METHOD == "krum":
                    num_attackers = NUM_CLIENTS - MIN_HONEST_CLIENTS
                    honest_indices, scores = multi_krum(
                        client_weights_list,
                        num_attackers=num_attackers
                    )
                    score_name = "Krum Score"
                    
                else:
                    # Default: accept all
                    honest_indices = list(range(len(client_ids)))
                    scores = [1.0] * len(client_ids)
                    score_name = "Score"
                
                # Display results
                logger.info("\nClient Security Scores:")
                honest_clients = []
                malicious_clients = []
                
                for i, client_id in enumerate(client_ids):
                    score = scores[i]
                    is_honest = i in honest_indices
                    
                    status = "✓ HONEST" if is_honest else "✗ MALICIOUS"
                    logger.info(f"  {client_id:30s} {score_name}: {score:8.4f} - {status}")
                    
                    if is_honest:
                        honest_clients.append(client_id)
                    else:
                        malicious_clients.append(client_id)
                        total_attacks_blocked += 1
                
                logger.info(f"\nVerdict: {len(honest_clients)} honest, {len(malicious_clients)} malicious")
                
                if malicious_clients:
                    logger.warning(f"⚠️  Blocked malicious clients: {', '.join(malicious_clients)}")
                
                # Check if we have enough honest clients
                if len(honest_clients) < MIN_HONEST_CLIENTS:
                    logger.error(f"❌ Insufficient honest clients ({len(honest_clients)} < {MIN_HONEST_CLIENTS})")
                    logger.error("Skipping this round - model not updated")
                    client_updates = {}
                    continue
                
                # AGGREGATION
                logger.info(f"\n📊 Aggregating {len(honest_clients)} honest updates...")
                logger.info(f"Aggregation method: {AGGREGATION_METHOD}")
                
                honest_weights = [client_weights_list[i] for i in honest_indices]
                
                if AGGREGATION_METHOD == "mean":
                    # Standard Federated Averaging
                    averaged_weights = []
                    num_layers = len(honest_weights[0])
                    for i in range(num_layers):
                        layer_weights = [cw[i] for cw in honest_weights]
                        averaged_layer = np.mean(layer_weights, axis=0)
                        averaged_weights.append(averaged_layer)
                        
                elif AGGREGATION_METHOD == "median":
                    averaged_weights = aggregate_median(honest_weights)
                    
                elif AGGREGATION_METHOD == "trimmed_mean":
                    averaged_weights = aggregate_trimmed_mean(honest_weights, trim_ratio=0.2)
                    
                else:
                    raise ValueError(f"Unknown aggregation method: {AGGREGATION_METHOD}")
                
                # Calculate model divergence
                divergence = calculate_model_divergence(initial_weights, averaged_weights)
                logger.info(f"\nModel divergence from initial:")
                logger.info(f"  L2 distance: {divergence['l2_distance']:.6f}")
                logger.info(f"  Cosine similarity: {divergence['cosine_similarity']:.6f}")
                
                # Update global model
                global_model.set_weights(averaged_weights)
                
                # Save model
                save_path = os.path.join(MODELS_DIR, 'global_model_secured.h5')
                global_model.save(save_path)
                logger.info(f"\n✓ Global model updated and saved to: {save_path}")
                
                # Save round statistics
                round_stats = {
                    'round': round_number,
                    'timestamp': datetime.now().isoformat(),
                    'clients_participated': client_ids,
                    'honest_clients': honest_clients,
                    'malicious_clients': malicious_clients,
                    'security_scores': {client_ids[i]: float(scores[i]) for i in range(len(client_ids))},
                    'model_divergence': divergence,
                    'defense_method': DEFENSE_METHOD,
                    'aggregation_method': AGGREGATION_METHOD
                }
                
                stats_file = os.path.join(METRICS_DIR, f'round_{round_number:03d}_stats.json')
                with open(stats_file, 'w') as f:
                    json.dump(round_stats, f, indent=2)
                
                logger.info(f"Round statistics saved to: {stats_file}")
                logger.info("="*60 + "\n")
                
                # Reset for next round
                client_updates = {}
                
                # Stop after specified number of rounds
                if round_number >= NUM_FEDERATED_ROUNDS:
                    logger.info(f"\n✓ Completed {NUM_FEDERATED_ROUNDS} rounds - stopping server")
                    break
        
        except Exception as e:
            logger.error(f"Error processing update: {str(e)}", exc_info=True)
            continue

except KeyboardInterrupt:
    logger.info("\n\nKeyboard interrupt - stopping server")
except Exception as e:
    logger.error(f"\n\nFATAL ERROR: {str(e)}", exc_info=True)
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
    
    if 'consumer' in locals():
        consumer.close()