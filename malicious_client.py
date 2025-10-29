import json
import numpy as np
import tensorflow as tf
from kafka import KafkaProducer
import time

# --- CONFIGURATION ---
CLIENT_ID = "Malicious-Attacker-01"
MODEL_PATH = "ucsd_baseline.h5" # We only need this to get the shape of the weights
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'model-updates'
UPDATE_INTERVAL = 10 # Send a malicious update every 10 seconds
# -------------------

print("Initializing MALICIOUS Client...")

# 1. Load the model structure to know the shape of the weights to create
base_model = tf.keras.models.load_model(MODEL_PATH)
weight_shapes = [w.shape for w in base_model.get_weights()]

# 2. Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"Malicious client is running. Will send garbage updates every {UPDATE_INTERVAL} seconds.")

while True:
    # 3. Generate GARBAGE weights (random noise)
    # This is the "attack". Instead of real learnings, we send junk.
    garbage_weights = [np.random.normal(size=shape).tolist() for shape in weight_shapes]

    update_message = {
        'client_id': CLIENT_ID,
        'weights': garbage_weights
    }
    
    # 4. Send the malicious update to Kafka
    print(f"\nSending malicious update from {CLIENT_ID}...")
    producer.send(KAFKA_TOPIC, value=update_message)
    producer.flush()
    print("Malicious update sent.")

    # Wait before sending the next attack
    time.sleep(UPDATE_INTERVAL)