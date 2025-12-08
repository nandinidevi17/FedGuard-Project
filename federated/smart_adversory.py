# federated/smart_adversary.py
"""
Smart Adversary (Hybrid Attacker).
Combines Model Poisoning (Integrity) and DDoS (Availability) into a single agent.
Randomly switches modes to test the full spectrum of FedGuard defenses.
"""
import os
import sys
import json
import numpy as np
import time
import logging
import io
import zlib
import random
from kafka import KafkaProducer

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Setup Logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'smart_adversary.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("smart_adversary")

# Configuration
CLIENT_ID = "Hybrid-Attacker-X"
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")

# Attack Parameters
POISON_SCALE = 10.0          # High scale to ensure detection/destruction
DDOS_DURATION = 5            # Seconds to flood before switching back
DDOS_PACKET_SIZE = 1024      # Size of garbage payload in bytes
PROBABILITY_POISON = 0.3     # 30% chance to Poison, 70% chance to DDoS

def serialize_weights_to_bytes(weights_list):
    """Compress weights to simulate a realistic update payload."""
    buf = io.BytesIO()
    savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
    np.savez_compressed(buf, **savez_dict)
    buf.seek(0)
    raw = buf.read()
    return zlib.compress(raw, level=6)

def generate_poison_weights(shapes):
    """Generates high-magnitude random noise."""
    out = []
    for s in shapes:
        if isinstance(s, tuple) and len(s) == 0:
            arr = np.array([], dtype=np.float32)
        else:
            # Massive noise to trigger Loss Check or MAD
            arr = np.random.normal(0, POISON_SCALE, size=s).astype(np.float32)
        out.append(arr)
    return out

def attack_poison(producer, shapes, attack_count):
    """EXECUTE INTEGRITY ATTACK: Send one poisoned update."""
    logger.warning(f"\n[MODE: POISON] Generating malicious weights (Scale={POISON_SCALE})...")
    
    garbage_weights = generate_poison_weights(shapes)
    payload = serialize_weights_to_bytes(garbage_weights)
    
    metadata = {
        'client_id': CLIENT_ID,
        'update_number': attack_count,
        'attack_type': 'hybrid_poison',
        'timestamp': time.time()
    }
    
    try:
        producer.send(
            KAFKA_TOPIC, 
            value=payload, 
            key=CLIENT_ID.encode('utf-8'),
            headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
        )
        producer.flush()
        logger.warning(f" >> Poisoned Update Sent ({len(payload)} bytes)")
    except Exception as e:
        logger.error(f"Poison send failed: {e}")

def attack_ddos(producer, duration):
    """EXECUTE AVAILABILITY ATTACK: Flood server for 'duration' seconds."""
    logger.warning(f"\n[MODE: DDoS] Flooding network for {duration} seconds...")
    
    end_time = time.time() + duration
    count = 0
    
    # Pre-generate a garbage payload to send repeatedly
    garbage_payload = os.urandom(DDOS_PACKET_SIZE)
    
    while time.time() < end_time:
        try:
            # Send without waiting for acknowledgement (Fire and Forget)
            # We spoof the timestamp to try and trick the rate limiter (though FedGuard checks arrival time)
            producer.send(
                KAFKA_TOPIC, 
                value=garbage_payload, 
                key=CLIENT_ID.encode('utf-8')
            )
            count += 1
            
            # Small sleep to prevent local machine freeze, but fast enough to flood
            # In a real attack, this sleep would be 0
            if count % 100 == 0:
                time.sleep(0.001) 
                
        except Exception as e:
            logger.error(f"DDoS send failed: {e}")
            break
            
    logger.warning(f" >> DDoS Burst Complete. Sent {count} packets.")

def main():
    producer = None
    try:
        logger.info("="*60)
        logger.info("SMART ADVERSARY STARTING")
        logger.info("="*60)
        
        # 1. Load Model Shapes (to fake realistic updates)
        logger.info(f"Analyzing target model at {MODEL_PATH}...")
        shapes = [(3,)] # Default fallback
        try:
            import tensorflow as tf
            if os.path.exists(MODEL_PATH):
                base_model = tf.keras.models.load_model(MODEL_PATH)
                shapes = [w.shape for w in base_model.get_weights()]
                logger.info(f"[OK] Model analysis complete - {len(shapes)} layers identified")
        except Exception as e:
            logger.warning(f"Could not analyze model ({e}). Using dummy shapes.")

        # 2. Connect to Kafka
        logger.info(f"Connecting to Kafka Broker at {KAFKA_SERVER}...")
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: v,
            acks=0,  # No Ack needed for speed (DDoS mode benefit)
            linger_ms=0 # Send immediately
        )
        logger.info("[OK] Connection established. Engaging target.")

        attack_cycle = 0
        
        while True:
            attack_cycle += 1
            
            # DECIDE ATTACK MODE
            # Roll a die: < 0.3 = Poison, >= 0.3 = DDoS
            if random.random() < PROBABILITY_POISON:
                # --- POISON MODE ---
                attack_poison(producer, shapes, attack_cycle)
                # Sleep to mimic "training time" so we look slightly legit between poisons
                sleep_time = random.randint(5, 10)
                logger.info(f"   (Sleeping {sleep_time}s to evade detection...)")
                time.sleep(sleep_time)
            else:
                # --- DDOS MODE ---
                attack_ddos(producer, duration=DDOS_DURATION)
                # Sleep briefly to let the logs catch up
                time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Manual Stop.")
    except Exception as e:
        logger.error(f"Fatal Error: {e}", exc_info=True)
    finally:
        if producer: producer.close()

if __name__ == "__main__":
    main()