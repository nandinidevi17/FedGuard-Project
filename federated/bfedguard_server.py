import os
import sys
import json
import numpy as np
import logging
import time
import io
import zlib
import cv2  # Added for local image loading
from datetime import datetime
from kafka import KafkaConsumer
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils.security import (
    detect_outliers_mad,
    aggregate_median,
    validate_weight_list,
    validate_update_quality
)
from utils.metrics import calculate_model_divergence

# Setup Logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'fedguard_unified.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fedguard_unified")

# --- HELPER FUNCTIONS ---
def deserialize_weights_from_bytes(b):
    if b is None or len(b) == 0: raise ValueError("Empty payload")
    try:
        decompressed = zlib.decompress(b)
        buf = io.BytesIO(decompressed)
        buf.seek(0)
        npz = np.load(buf, allow_pickle=True)
        return [npz[f] for f in npz.files]
    except Exception:
        return None 

def extract_metadata_from_headers(headers):
    if not headers: return {}
    for k, v in headers:
        try:
            if k == 'metadata' and v is not None:
                return json.loads(v.decode('utf-8'))
        except Exception: continue
    return {}

# ---------------------------------------------------------
# LOCAL LOADER (Bypasses utils/data_loader.py to fix hang)
# ---------------------------------------------------------
def load_validation_subset_locally(data_path, limit=32, img_size=128):
    """
    Loads exactly 'limit' images from data_path (recursive).
    This prevents the server from reading 7000+ images and hanging.
    """
    frames = []
    count = 0
    logger.info(f"Scanning for validation images in: {data_path}")
    
    for root, dirs, files in os.walk(data_path):
        for filename in sorted(files):
            if count >= limit: break
            
            if filename.endswith(('.tif', '.jpg', '.png')):
                try:
                    img_path = os.path.join(root, filename)
                    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frame = cv2.resize(frame, (img_size, img_size))
                        frame = frame.astype('float32') / 255.0
                        frames.append(frame)
                        count += 1
                except Exception:
                    pass
        if count >= limit: break
        
    if len(frames) == 0:
        return None
        
    logger.info(f"Successfully loaded {len(frames)} validation frames locally.")
    return np.array(frames).reshape(-1, img_size, img_size, 1)

# --- MAIN SERVER ---
def main():
    round_number = 0
    total_updates_received = 0
    
    # Defense Config
    client_last_seen = {}
    MIN_UPDATE_INTERVAL = 5.0
    LOSS_THRESHOLD = 0.15

    try:
        logger.info("="*60)
        logger.info("FEDGUARD UNIFIED SERVER (Availability + Integrity)")
        logger.info("="*60)
        
        # 1. Load Model
        model_path = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
        if not os.path.exists(model_path):
            logger.error(f"Baseline model not found at {model_path}")
            return
        
        try:
            global_model = tf.keras.models.load_model(model_path)
            global_model.compile(optimizer='adam', loss='mse')
            logger.info("[OK] Model loaded.")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return
            
        # 2. Load Validation Data (LOCALLY)
        logger.info("Loading validation set...")
        X_val = load_validation_subset_locally(UCSD_PED1_TEST, limit=32, img_size=IMG_SIZE)
        
        if X_val is None:
            logger.error(f"Failed to load validation set from {UCSD_PED1_TEST}")
            logger.error("Please check if the dataset path in config.py is correct.")
            return
            
        logger.info(f"[OK] Validation set ready: {X_val.shape}")

        # 3. Kafka Consumer
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            value_deserializer=lambda m: m,
            auto_offset_reset='latest',
            group_id='fedguard-unified-final'
        )
        logger.info("[OK] Waiting for updates...\n")

        client_updates = {}

        for message in consumer:
            # Layer 1: Rate Limiting
            key = message.key.decode('utf-8') if message.key else f"unknown"
            current_time = time.time()
            last_time = client_last_seen.get(key, 0)

            if (current_time - last_time) < MIN_UPDATE_INTERVAL:
                continue # Block DDoS
            
            client_last_seen[key] = current_time
            total_updates_received += 1
            
            # Processing
            headers = message.headers if hasattr(message, 'headers') else None
            metadata = extract_metadata_from_headers(headers)
            raw_bytes = message.value
            weights = deserialize_weights_from_bytes(raw_bytes)
            
            if weights is None:
                continue

            client_updates[key] = {'weights': weights, 'metadata': metadata}
            logger.info(f"[Round {round_number + 1}] Received valid update from {key}")
            logger.info(f"  Collected: {len(client_updates)}/{NUM_CLIENTS}")

            # Aggregation Trigger
            if len(client_updates) >= NUM_CLIENTS:
                round_number += 1
                logger.info("\n" + "="*60)
                logger.info(f"ROUND {round_number} - SECURITY CHECKS")
                logger.info("="*60)

                client_ids = list(client_updates.keys())
                validated_clients = []
                malicious_clients = []

                # Layer 2: Loss Check
                logger.info("[LAYER 2] Checking Update Quality...")
                for cid in client_ids:
                    w = client_updates[cid]['weights']
                    is_valid, loss = validate_update_quality(global_model, w, X_val, max_loss_threshold=LOSS_THRESHOLD)
                    
                    if is_valid:
                        logger.info(f"  {cid:25s} Loss: {loss:.4f} -> [PASS]")
                        validated_clients.append(cid)
                    else:
                        logger.warning(f"  {cid:25s} Loss: {loss:.4f} -> [BLOCK] (Poison)")
                        malicious_clients.append(cid)

                if not validated_clients:
                    logger.error("ALL updates blocked! Skipping round.")
                    client_updates = {}
                    continue

                # Layer 3: MAD Check
                final_honest_clients = []
                if len(validated_clients) < 2:
                    final_honest_clients = validated_clients
                else:
                    logger.info("\n[LAYER 3] Checking Statistical Consensus (MAD)...")
                    valid_weights = [client_updates[cid]['weights'] for cid in validated_clients]
                    honest_indices, scores = detect_outliers_mad(valid_weights, threshold=3.5)
                    
                    for i, cid in enumerate(validated_clients):
                        if i in honest_indices:
                            final_honest_clients.append(cid)
                        else:
                            logger.warning(f"  {cid:25s} MAD Score: {scores[i]:.2f} -> [BLOCK] (Outlier)")
                            malicious_clients.append(cid)

                # Update Model
                logger.info(f"\nAggregating {len(final_honest_clients)} honest updates...")
                final_weights_list = [client_updates[cid]['weights'] for cid in final_honest_clients]
                averaged_weights = aggregate_median(final_weights_list)

                try:
                    global_model.set_weights(averaged_weights)
                    save_path = os.path.join(MODELS_DIR, 'global_model_secured.h5')
                    global_model.save(save_path)
                    logger.info(f"[OK] Global Model Updated: {save_path}")
                except Exception as e:
                    logger.error(f"Save failed: {e}")

                client_updates = {} 
                logger.info("="*60 + "\n")

    except KeyboardInterrupt:
        logger.info("Server stopped.")
        if 'consumer' in locals(): consumer.close()

if __name__ == "__main__":
    main()