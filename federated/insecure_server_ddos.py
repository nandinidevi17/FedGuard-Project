"""
Dedicated Insecure Server for DDoS Experiment.
Logs to 'results/logs/ddos_experiment.log' to avoid overwriting integrity results.
"""
import os
import sys
import json
import numpy as np
import time
import logging
from datetime import datetime
from kafka import KafkaConsumer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- CONFIGURATION FOR THIS EXPERIMENT ---
LOG_FILENAME = 'ddos_experiment.log'
GROUP_ID = 'ddos-experiment-group-v1' 
# -----------------------------------------

os.makedirs(LOGS_DIR, exist_ok=True)

# Custom Logger Setup to write to a SEPARATE file
logger = logging.getLogger("insecure_server_ddos")
logger.setLevel(getattr(logging, LOG_LEVEL))
formatter = logging.Formatter(LOG_FORMAT)

# File Handler (The important part!)
file_handler = logging.FileHandler(os.path.join(LOGS_DIR, LOG_FILENAME), encoding='utf-8')
file_handler.setFormatter(formatter)

# Stream Handler (Console)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def main():
    logger.warning("="*60)
    logger.warning("🔥🔥 INSECURE SERVER (DDoS EXPERIMENT) 🔥🔥")
    logger.warning(f"Logging to: {LOG_FILENAME}")
    logger.warning("This server is isolated for Availability testing.")
    logger.warning("="*60)

    # Initialize Kafka consumer
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=KAFKA_SERVER,
            # We don't care about deserializing weights for this test, just counting them
            value_deserializer=lambda m: m, 
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=GROUP_ID
        )
        logger.info(f"[OK] Connected to Kafka (Topic: {KAFKA_TOPIC})")
    except Exception as e:
        logger.error(f"Kafka connection failed: {e}")
        return

    client_updates = set()
    round_number = 0
    
    logger.info("Waiting for updates...")

    for message in consumer:
        # We only care about the TIMING of messages for DDoS proof
        try:
            key_str = message.key.decode('utf-8') if message.key else "unknown"
        except:
            key_str = "unknown"

        logger.info(f"[Round {round_number + 1}] Received update from: {key_str}")

        # Simple logic to track "Round Completion"
        # In a DDoS attack, this list fills up with the Attacker, 
        # but honest clients get stuck in the queue.
        client_updates.add(key_str)
        
        # Log collection progress
        logger.info(f"   Unique clients seen: {len(client_updates)}/{NUM_CLIENTS}")

        if len(client_updates) >= NUM_CLIENTS:
            round_number += 1
            logger.info(f"\n" + "="*60)
            logger.info(f"ROUND {round_number} AGGREGATION (Simulated)")
            logger.info("="*60 + "\n")
            client_updates.clear()

if __name__ == "__main__":
    main()