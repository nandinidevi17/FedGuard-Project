# """
# Malicious Client - Sends poisoned model updates.
# Used to test FedGuard security mechanisms.
# Now sends poisoned updates in same bytes format as honest clients.
# """
# import os
# import sys
# import json
# import numpy as np
# import tensorflow as tf
# from kafka import KafkaProducer
# import time
# import logging
# import io
# import zlib

# # Add parent directory to path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config import *

# os.makedirs(LOGS_DIR, exist_ok=True)
# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL),
#     format=LOG_FORMAT,
#     handlers=[
#         logging.FileHandler(os.path.join(LOGS_DIR, 'malicious_client.log'), encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# CLIENT_ID = "Malicious-Attacker-01"
# MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
# UPDATE_INTERVAL = 10  # seconds

# ATTACK_TYPE = "random_noise"  # random_noise, gradient_ascent, label_flip
# ATTACK_SCALE = 10.0

# def serialize_weights_to_bytes(weights_list):
#     buf = io.BytesIO()
#     savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
#     np.savez_compressed(buf, **savez_dict)
#     buf.seek(0)
#     raw = buf.read()
#     return zlib.compress(raw, level=6)

# try:
#     logger.warning("="*60)
#     logger.warning("[WARNING] MALICIOUS CLIENT STARTING")
#     logger.warning("="*60)
#     logger.warning(f"Client ID: {CLIENT_ID}")
#     logger.warning(f"Attack type: {ATTACK_TYPE}")
#     logger.warning(f"Attack scale: {ATTACK_SCALE}")

#     logger.info(f"Loading model structure from {MODEL_PATH}...")
#     base_model = tf.keras.models.load_model(MODEL_PATH)
#     weight_shapes = [w.shape for w in base_model.get_weights()]
#     logger.info(f"[OK] Model structure loaded - {len(weight_shapes)} layers")

#     logger.info(f"Connecting to Kafka at {KAFKA_SERVER}...")
#     producer = KafkaProducer(bootstrap_servers=KAFKA_SERVER, value_serializer=lambda v: v, request_timeout_ms=KAFKA_TIMEOUT)
#     logger.info("[OK] Kafka connection established")

#     logger.warning("\n" + "="*60)
#     logger.warning("ATTACK IN PROGRESS")
#     logger.warning(f"Sending poisoned updates every {UPDATE_INTERVAL} seconds")
#     logger.warning("="*60 + "\n")

#     attack_count = 0
#     while True:
#         attack_count += 1
#         if ATTACK_TYPE == "random_noise":
#             garbage_weights = [np.random.normal(0, ATTACK_SCALE, size=shape).astype(np.float32) for shape in weight_shapes]
#         elif ATTACK_TYPE == "gradient_ascent":
#             real_weights = base_model.get_weights()
#             garbage_weights = [(-w * ATTACK_SCALE).astype(np.float32) for w in real_weights]
#         elif ATTACK_TYPE == "label_flip":
#             real_weights = base_model.get_weights()
#             garbage_weights = [(w + np.random.normal(0, 0.1 * ATTACK_SCALE, size=w.shape)).astype(np.float32) for w in real_weights]
#         else:
#             raise ValueError(f"Unknown attack type: {ATTACK_TYPE}")

#         metadata = {
#             'client_id': CLIENT_ID,
#             'update_number': attack_count,
#             'attack_type': ATTACK_TYPE,
#             'attack_scale': ATTACK_SCALE,
#             'timestamp': time.time()
#         }

#         logger.warning(f"\n[Attack #{attack_count}] Sending poisoned update...")
#         try:
#             payload_bytes = serialize_weights_to_bytes(garbage_weights)
#             producer.send(KAFKA_TOPIC, value=payload_bytes, key=CLIENT_ID.encode('utf-8'),
#                           headers=[('metadata', json.dumps(metadata).encode('utf-8'))])
#             producer.flush()
#             logger.warning(f"[OK] Malicious update sent to topic '{KAFKA_TOPIC}'")
#         except Exception as e:
#             logger.error(f"Failed to send malicious update: {e}", exc_info=True)

#         time.sleep(UPDATE_INTERVAL)

# except KeyboardInterrupt:
#     logger.info("\n\nKeyboard interrupt - stopping attacks")
# except Exception as e:
#     logger.error(f"\n\nERROR: {str(e)}", exc_info=True)
# finally:
#     try:
#         producer.close()
#     except Exception:
#         pass
#     logger.warning(f"\nMalicious client stopped after {attack_count} attacks")


# malicious_client.py
import os
import sys
import json
import numpy as np
import time
import logging
import io
import zlib

from kafka import KafkaProducer

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
logger = logging.getLogger("malicious_client")

CLIENT_ID = "Malicious-Attacker-03"
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
UPDATE_INTERVAL = 20  # seconds

ATTACK_TYPE = "random_noise"  # random_noise, gradient_ascent, label_flip
ATTACK_SCALE = 2.0

def serialize_weights_to_bytes(weights_list):
    buf = io.BytesIO()
    savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
    np.savez_compressed(buf, **savez_dict)
    buf.seek(0)
    raw = buf.read()
    return zlib.compress(raw, level=6)

def make_garbage_weights_like(shapes, attack_type="random_noise", scale=1.0, base_model_weights=None):
    out = []
    if attack_type == "random_noise":
        for s in shapes:
            if isinstance(s, tuple) and len(s) == 0:
                arr = np.array([], dtype=np.float32)
            else:
                arr = np.random.normal(0, scale, size=s).astype(np.float32)
            out.append(arr)
    elif attack_type == "gradient_ascent" and base_model_weights is not None:
        for w in base_model_weights:
            out.append(( -w * scale ).astype(np.float32))
    elif attack_type == "label_flip" and base_model_weights is not None:
        for w in base_model_weights:
            out.append(( w + np.random.normal(0, 0.1 * scale, size=w.shape) ).astype(np.float32))
    else:
        # fallback to random noise
        for s in shapes:
            arr = np.random.normal(0, scale, size=s).astype(np.float32)
            out.append(arr)
    return out

def main():
    producer = None
    attack_count = 0
    try:
        logger.warning("="*60)
        logger.warning("[WARNING] MALICIOUS CLIENT STARTING")
        logger.warning("="*60)
        logger.warning(f"Client ID: {CLIENT_ID}")
        logger.warning(f"Attack type: {ATTACK_TYPE}")
        logger.warning(f"Attack scale: {ATTACK_SCALE}")

        # Try to load model to learn the weight shapes
        logger.info(f"Loading model structure from {MODEL_PATH}...")
        shapes = None
        base_weights = None
        try:
            import tensorflow as tf
            if os.path.exists(MODEL_PATH):
                base_model = tf.keras.models.load_model(MODEL_PATH)
                base_weights = base_model.get_weights()
                shapes = [w.shape for w in base_weights]
                logger.info(f"[OK] Model structure loaded - {len(shapes)} layers")
            else:
                logger.warning("Model file not found; will send small test-compatible shapes instead")
                shapes = [(3,)]  # fallback small shape (but note server divergence checks will fail if it expects full-model)
        except Exception as e:
            logger.error(f"Failed to load model for shapes: {e}", exc_info=True)
            shapes = [(3,)]

        logger.info(f"Connecting to Kafka at {KAFKA_SERVER}...")
        timeout_ms = KAFKA_TIMEOUT if KAFKA_TIMEOUT is not None else 30000
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: v,
            acks='all',
            max_request_size=KAFKA_MAX_REQUEST_BYTES,
            request_timeout_ms=timeout_ms
        )
        logger.info("[OK] Kafka connection established")

        logger.warning("\n" + "="*60)
        logger.warning("ATTACK IN PROGRESS (simulated)")
        logger.warning(f"Sending poisoned updates every {UPDATE_INTERVAL} seconds")
        logger.warning("="*60 + "\n")

        while True:
            attack_count += 1
            garbage_weights = make_garbage_weights_like(shapes, attack_type=ATTACK_TYPE, scale=ATTACK_SCALE, base_model_weights=base_weights)

            metadata = {
                'client_id': CLIENT_ID,
                'update_number': attack_count,
                'attack_type': ATTACK_TYPE,
                'attack_scale': ATTACK_SCALE,
                'timestamp': time.time()
            }

            logger.warning(f"\n[Attack #{attack_count}] Building poisoned update...")
            try:
                payload_bytes = serialize_weights_to_bytes(garbage_weights)
                logger.warning(f"[SEND DEBUG] bootstrap={KAFKA_SERVER} topic={KAFKA_TOPIC} key={CLIENT_ID} payload_bytes={len(payload_bytes)} headers_present={bool(metadata)}")
                fut = producer.send(KAFKA_TOPIC, value=payload_bytes, key=CLIENT_ID.encode('utf-8'),
                              headers=[('metadata', json.dumps(metadata).encode('utf-8'))])
                fut.get(timeout=10)
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
            if producer is not None:
                producer.close()
        except Exception:
            pass
        logger.warning(f"\nMalicious client stopped after {attack_count} attacks")

if __name__ == "__main__":
    main()
