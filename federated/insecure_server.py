"""
Insecure Server - NO DEFENSE MECHANISMS.
Accepts all updates including malicious ones.
Used for comparison with FedGuard.
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

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'insecure_server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.warning("="*60)
logger.warning("⚠️  INSECURE SERVER (NO DEFENSE)")
logger.warning("="*60)
logger.warning("This server accepts ALL updates without security checks")
logger.warning(f"Expected clients: {NUM_CLIENTS}")

try:
    # Load model
    model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
    logger.info(f"\nLoading model from {model_path}...")
    global_model = tf.keras.models.load_model(model_path)
    logger.info("✓ Model loaded")
    
    # Initialize Kafka
    logger.info(f"\nConnecting to Kafka at {KAFKA_SERVER}...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_SERVER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        consumer_timeout_ms=KAFKA_TIMEOUT
    )
    logger.info("✓ Kafka connected")
    
    logger.info("\n" + "="*60)
    logger.info("SERVER RUNNING")
    logger.info("="*60 + "\n")
    
    client_updates = {}
    round_number = 0
    
    for message in consumer:
        update_data = message.value
        client_id = update_data['client_id']
        
        logger.info(f"\n[Round {round_number + 1}] Received update from: {client_id}")
        
        received_weights = [np.array(w) for w in update_data['weights']]
        client_updates[client_id] = received_weights
        
        logger.info(f"  Updates collected: {len(client_updates)}/{NUM_CLIENTS}")
        
        if len(client_updates) >= NUM_CLIENTS:
            round_number += 1
            logger.warning(f"\n⚠️  ROUND {round_number} - Accepting ALL clients (no security check)")
            
            client_ids = list(client_updates.keys())
            all_weights = list(client_updates.values())
            
            logger.info(f"Averaging updates from: {', '.join(client_ids)}")
            
            # Simple averaging (no filtering)
            averaged_weights = []
            num_layers = len(all_weights[0])
            for i in range(num_layers):
                layer_weights = [cw[i] for cw in all_weights]
                averaged_layer = np.mean(layer_weights, axis=0)
                averaged_weights.append(averaged_layer)
            
            # Update model
            global_model.set_weights(averaged_weights)
            
            save_path = os.path.join(MODELS_DIR, 'global_model_insecured.h5')
            global_model.save(save_path)
            logger.info(f"✓ Model updated and saved to: {save_path}\n")
            
            client_updates = {}
            
            if round_number >= NUM_FEDERATED_ROUNDS:
                logger.info(f"Completed {NUM_FEDERATED_ROUNDS} rounds - stopping")
                break

except KeyboardInterrupt:
    logger.info("\nKeyboard interrupt - stopping server")
except Exception as e:
    logger.error(f"\nERROR: {str(e)}", exc_info=True)
finally:
    if 'consumer' in locals():
        consumer.close()
    logger.warning(f"\nInsecure server stopped after {round_number} rounds")