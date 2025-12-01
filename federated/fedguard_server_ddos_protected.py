"""
FedGuard Server with DDoS Protection (Rate Limiting).
This server:
1. TRACKS client update frequency.
2. BLOCKS updates that are too fast (DDoS defense).
3. ACCEPTS valid updates from honest clients.
4. AGGREGATES and SAVES the global model (Training).
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

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'ddos_defense.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fedguard_ddos_shield")

# --- HELPER FUNCTIONS ---
def deserialize_weights_from_bytes(b):
    if b is None or len(b) == 0:
        raise ValueError("Empty payload")
    try:
        decompressed = zlib.decompress(b)
        buf = io.BytesIO(decompressed)
        buf.seek(0)
        npz = np.load(buf, allow_pickle=True)
        arrays = [npz[f] for f in npz.files]
        return arrays
    except Exception:
        # Fallback for uncompressed or JSON
        try:
            payload = b.decode('utf-8')
            parsed = json.loads(payload)
            if isinstance(parsed, dict) and 'weights' in parsed:
                wlists = parsed['weights']
            elif isinstance(parsed, list):
                wlists = parsed
            else:
                return None
            return [np.array(w, dtype=np.float32) for w in wlists]
        except Exception:
            return None

def shapes_match(weights_a, weights_b):
    if weights_a is None or weights_b is None: return False
    if len(weights_a) != len(weights_b): return False
    for wa, wb in zip(weights_a, weights_b):
        if wa.shape != wb.shape: return False
    return True

# --- MAIN SERVER WITH DEFENSE ---
def main():
    round_number = 0
    total_updates_received = 0
    
    # 🛡️ DEFENSE CONFIGURATION 🛡️
    # We will track the last time we saw a message from each client ID.
    client_last_seen = {} 
    MIN_UPDATE_INTERVAL = 5.0  # Seconds. Honest clients wait ~15s. Attackers wait 0s.

    try:
        logger.info("="*60)
        logger.info("FEDGUARD SERVER - DDOS PROTECTION ENABLED")
        logger.info("="*60)
        logger.info(f"Rate Limit: 1 request every {MIN_UPDATE_INTERVAL} seconds")

        # Load Model
        model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
        import tensorflow as tf
        if os.path.exists(model_path):
            global_model = tf.keras.models.load_model(model_path)
            initial_weights = global_model.get_weights()
            logger.info("[OK] Model loaded successfully")
        else:
            logger.error("Model not found. Please run training first.")
            return

        # Connect to Kafka
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda m: m,
            auto_offset_reset='latest', # Start fresh to see live defense
            group_id='fedguard-ddos-shield-group-v2'
        )
        logger.info("[OK] Kafka Consumer Started")
        logger.info("\nWaiting for updates...\n")

        client_updates = {}

        for message in consumer:
            # 1. Identify Client
            key = message.key.decode('utf-8') if message.key else f"unknown"
            
            # 🛡️ 2. RATE LIMIT CHECK (The Defense) 🛡️
            current_time = time.time()
            last_time = client_last_seen.get(key, 0)
            
            if (current_time - last_time) < MIN_UPDATE_INTERVAL:
                # ATTACK DETECTED: DROP IT
                logger.warning(f"⛔ [BLOCKED] DDoS/Flood detected from {key} (Too fast!)")
                continue  # Skip processing, save CPU/Memory
            
            # Update the last seen time for this client
            client_last_seen[key] = current_time
            # -------------------------------------------------

            # 3. Process Valid Update (Normal Logic)
            total_updates_received += 1
            logger.info(f"✅ [ACCEPTED] Processing update from {key}")

            raw_bytes = message.value
            try:
                received_weights = deserialize_weights_from_bytes(raw_bytes)
                if initial_weights and shapes_match(received_weights, initial_weights):
                    client_updates[key] = {'weights': received_weights}
                    logger.info(f"   Shape verified. Stored update.")
                else:
                    logger.warning(f"   Shape mismatch. Ignored.")
            except Exception:
                logger.warning(f"   Deserialization error. Ignored.")

            # 4. Aggregate if ready
            num_valid = len(client_updates)
            logger.info(f"   Updates collected: {num_valid}/{NUM_CLIENTS}")

            if num_valid >= NUM_CLIENTS:
                round_number += 1
                logger.info(f"\n" + "="*60)
                logger.info(f"ROUND {round_number} AGGREGATION")
                logger.info("="*60)
                
                # --- START AGGREGATION LOGIC ---
                try:
                    # 1. Extract weights
                    client_ids = list(client_updates.keys())
                    weights = [client_updates[cid]['weights'] for cid in client_ids]
                    
                    # 2. Simple Mean Aggregation
                    # (Assuming Rate Limiting filtered the spam, we average the rest)
                    averaged_weights = []
                    for i in range(len(weights[0])):
                        layer_stack = np.stack([w[i] for w in weights], axis=0)
                        averaged_weights.append(np.mean(layer_stack, axis=0))
                    
                    # 3. Save Model
                    global_model.set_weights(averaged_weights)
                    save_path = os.path.join(MODELS_DIR, 'global_model_ddos_secured.h5')
                    global_model.save(save_path)
                    logger.info(f"[OK] Model aggregated and saved to: {save_path}")
                    logger.info(f"     Round {round_number} complete.")
                    
                except Exception as e:
                    logger.error(f"Aggregation failed: {e}")

                # Reset for next round
                client_updates = {} 
                logger.info("="*60 + "\n")

    except KeyboardInterrupt:
        logger.info("Stopping server...")
    finally:
        if 'consumer' in locals():
            consumer.close()

if __name__ == "__main__":
    main()