"""
Malicious Client - Sends poisoned model updates.
Used to test FedGuard security mechanisms.
Now sends poisoned updates in same bytes format as honest clients.
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from kafka import KafkaProducer
import time
import logging
import io
import zlib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'malicious_client.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

CLIENT_ID = "Malicious-Attacker-01"
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
UPDATE_INTERVAL = 10  # seconds

ATTACK_TYPE = "random_noise"  # random_noise, gradient_ascent, label_flip
ATTACK_SCALE = 10.0

def serialize_weights_to_bytes(weights_list):
    buf = io.BytesIO()
    savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
    np.savez_compressed(buf, **savez_dict)
    buf.seek(0)
    raw = buf.read()
    return zlib.compress(raw, level=6)

try:
    logger.warning("="*60)
    logger.warning("[WARNING] MALICIOUS CLIENT STARTING")
    logger.warning("="*60)
    logger.warning(f"Client ID: {CLIENT_ID}")
    logger.warning(f"Attack type: {ATTACK_TYPE}")
    logger.warning(f"Attack scale: {ATTACK_SCALE}")

    logger.info(f"Loading model structure from {MODEL_PATH}...")
    base_model = tf.keras.models.load_model(MODEL_PATH)
    weight_shapes = [w.shape for w in base_model.get_weights()]
    logger.info(f"[OK] Model structure loaded - {len(weight_shapes)} layers")

    logger.info(f"Connecting to Kafka at {KAFKA_SERVER}...")
    producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=lambda v: v, request_timeout_ms=KAFKA_TIMEOUT)
    logger.info("[OK] Kafka connection established")

    logger.warning("\n" + "="*60)
    logger.warning("ATTACK IN PROGRESS")
    logger.warning(f"Sending poisoned updates every {UPDATE_INTERVAL} seconds")
    logger.warning("="*60 + "\n")

    attack_count = 0
    while True:
        attack_count += 1
        if ATTACK_TYPE == "random_noise":
            garbage_weights = [np.random.normal(0, ATTACK_SCALE, size=shape).astype(np.float32) for shape in weight_shapes]
        elif ATTACK_TYPE == "gradient_ascent":
            real_weights = base_model.get_weights()
            garbage_weights = [(-w * ATTACK_SCALE).astype(np.float32) for w in real_weights]
        elif ATTACK_TYPE == "label_flip":
            real_weights = base_model.get_weights()
            garbage_weights = [(w + np.random.normal(0, 0.1 * ATTACK_SCALE, size=w.shape)).astype(np.float32) for w in real_weights]
        else:
            raise ValueError(f"Unknown attack type: {ATTACK_TYPE}")

        metadata = {
            'client_id': CLIENT_ID,
            'update_number': attack_count,
            'attack_type': ATTACK_TYPE,
            'attack_scale': ATTACK_SCALE,
            'timestamp': time.time()
        }

        logger.warning(f"\n[Attack #{attack_count}] Sending poisoned update...")
        try:
            payload_bytes = serialize_weights_to_bytes(garbage_weights)
            producer.send(KAFKA_TOPIC, value=payload_bytes, key=CLIENT_ID.encode('utf-8'),
                          headers=[('metadata', json.dumps(metadata).encode('utf-8'))])
            producer.flush()
            logger.warning(f"[OK] Malicious update sent to topic '{KAFKA_TOPIC}'")
        except Exception as e:
            logger.error(f"Failed to send malicious update: {e}", exc_info=True)

        time.sleep(UPDATE_INTERVAL)

except KeyboardInterrupt:
    logger.info("\n\nKeyboard interrupt - stopping attacks")
except Exception as e:
    logger.error(f"\n\nERROR: {str(e)}", exc_info=True)
finally:
    try:
        producer.close()
    except Exception:
        pass
    logger.warning(f"\nMalicious client stopped after {attack_count} attacks")
