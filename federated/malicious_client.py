"""
Malicious Client - Sends poisoned model updates.
Used to test FedGuard security mechanisms.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from kafka import KafkaProducer
import time
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'malicious_client.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CLIENT_ID = "Malicious-Attacker-01"
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
UPDATE_INTERVAL = 10  # Send attack every 10 seconds

# Attack parameters
ATTACK_TYPE = "random_noise"  # Options: random_noise, gradient_ascent, label_flip
ATTACK_SCALE = 10.0  # Multiplier for attack strength
# -------------------

logger.warning("="*60)
logger.warning("⚠️  MALICIOUS CLIENT STARTING")
logger.warning("="*60)
logger.warning(f"Client ID: {CLIENT_ID}")
logger.warning(f"Attack type: {ATTACK_TYPE}")
logger.warning(f"Attack scale: {ATTACK_SCALE}")

try:
    # Load model to get weight structure
    logger.info(f"Loading model structure from {MODEL_PATH}...")
    base_model = tf.keras.models.load_model(MODEL_PATH)
    weight_shapes = [w.shape for w in base_model.get_weights()]
    logger.info(f"✓ Model structure loaded - {len(weight_shapes)} layers")
    
    # Initialize Kafka
    logger.info(f"Connecting to Kafka at {KAFKA_SERVER}...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        request_timeout_ms=KAFKA_TIMEOUT
    )
    logger.info("✓ Kafka connection established")
    
    logger.warning("\n" + "="*60)
    logger.warning("ATTACK IN PROGRESS")
    logger.warning(f"Sending poisoned updates every {UPDATE_INTERVAL} seconds")
    logger.warning("="*60 + "\n")
    
    attack_count = 0
    
    while True:
        attack_count += 1
        
        # Generate malicious weights based on attack type
        if ATTACK_TYPE == "random_noise":
            # Pure random noise
            garbage_weights = [
                (np.random.normal(0, ATTACK_SCALE, size=shape)).tolist()
                for shape in weight_shapes
            ]
            
        elif ATTACK_TYPE == "gradient_ascent":
            # Flip the gradients (opposite of learning)
            real_weights = base_model.get_weights()
            garbage_weights = [
                (-w * ATTACK_SCALE).tolist()
                for w in real_weights
            ]
            
        elif ATTACK_TYPE == "label_flip":
            # Subtle attack: small perturbations
            real_weights = base_model.get_weights()
            garbage_weights = [
                (w + np.random.normal(0, 0.1 * ATTACK_SCALE, size=w.shape)).tolist()
                for w in real_weights
            ]
        else:
            raise ValueError(f"Unknown attack type: {ATTACK_TYPE}")
        
        # Prepare message
        update_message = {
            'client_id': CLIENT_ID,
            'weights': garbage_weights,
            'metadata': {
                'update_number': attack_count,
                'attack_type': ATTACK_TYPE,
                'attack_scale': ATTACK_SCALE
            }
        }
        
        # Send attack
        logger.warning(f"\n[Attack #{attack_count}] Sending poisoned update...")
        producer.send(KAFKA_TOPIC, value=update_message)
        producer.flush()
        logger.warning(f"✓ Malicious update sent to topic '{KAFKA_TOPIC}'")
        
        # Wait before next attack
        time.sleep(UPDATE_INTERVAL)

except KeyboardInterrupt:
    logger.info("\n\nKeyboard interrupt - stopping attacks")
except Exception as e:
    logger.error(f"\n\nERROR: {str(e)}", exc_info=True)
finally:
    if 'producer' in locals():
        producer.close()
    logger.warning(f"\nMalicious client stopped after {attack_count} attacks")