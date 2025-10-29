import os
import json
import numpy as np
import tensorflow as tf
from kafka import KafkaConsumer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'model-updates'
MODEL_PATH = "ucsd_baseline.h5"
NUM_CLIENTS = 3 # We expect updates from 1 honest + 1 malicious client
SIMILARITY_THRESHOLD = 0.5 # If an update's similarity is below this, it's rejected
# -------------------

# --- Helper Functions for Security ---
def flatten_weights(weights):
    """Converts a list of model weights into a single flat 1D array."""
    return np.concatenate([w.flatten() for w in weights])

def calculate_similarity(weights_list):
    """Calculates the cosine similarity of each client's update to the average."""
    # First, flatten each client's weights into a 1D vector
    flat_weights = [flatten_weights(w) for w in weights_list]
    
    # Calculate the average of all weight vectors
    average_weights = np.mean(flat_weights, axis=0)
    
    # Calculate the similarity of each client's weights to the average
    similarities = []
    for fw in flat_weights:
        # Reshape for sklearn's cosine_similarity function
        sim = cosine_similarity(fw.reshape(1, -1), average_weights.reshape(1, -1))
        similarities.append(sim[0][0])
        
    return similarities
# -----------------------------------


print("Initializing FedGuard Secure Server...")
global_model_secured= tf.keras.models.load_model(MODEL_PATH)

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=KAFKA_SERVER,
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest'
)

print("Server is running. Waiting for model updates...")

client_updates = {} # Use a dictionary to store updates by client_id

for message in consumer:
    update_data = message.value
    client_id = update_data['client_id']
    print(f"\nReceived update from: {client_id}")

    received_weights = [np.array(w) for w in update_data['weights']]
    client_updates[client_id] = received_weights

    if len(client_updates) >= NUM_CLIENTS:
        print(f"\nCollected all {NUM_CLIENTS} updates. Running FedGuard security check...")
        
        client_ids = list(client_updates.keys())
        client_weights_list = list(client_updates.values())
        
        # 1. Calculate the similarity scores
        similarity_scores = calculate_similarity(client_weights_list)
        
        honest_clients_weights = []
        for i, client_id in enumerate(client_ids):
            score = similarity_scores[i]
            print(f"  - Client '{client_id}' similarity score: {score:.4f}")
            if score >= SIMILARITY_THRESHOLD:
                print(f"  - Verdict: HONEST. Including in average.")
                honest_clients_weights.append(client_weights_list[i])
            else:
                print(f"  - Verdict: MALICIOUS DETECTED. Discarding update.")

       # (Keep all the code before this the same)

        # 2. Perform averaging only on the honest clients
        if not honest_clients_weights:
            print("\nNo honest clients found in this round. Skipping model update.")
        else:
            print(f"\nAveraging updates from {len(honest_clients_weights)} honest client(s)...")
            
            # --- OLD CODE TO REMOVE ---
            # averaged_weights = np.mean(honest_clients_weights, axis=0)
            
            # --- NEW, ROBUST CODE TO ADD ---
            averaged_weights = []
            num_layers = len(honest_clients_weights[0])
            for i in range(num_layers):
                layer_weights = [client_weights[i] for client_weights in honest_clients_weights]
                averaged_layer = np.mean(layer_weights, axis=0)
                averaged_weights.append(averaged_layer)
            # -----------------------------

            global_model_secured.set_weights(averaged_weights)
            global_model_secured.save('global_model_secured.h5')
            print("Secure global model has been updated and saved.")
        
        client_updates = {} # Reset for the next round