import os
import cv2
import numpy as np
import json
import tensorflow as tf
from kafka import KafkaProducer

# --- CONFIGURATION ---
CLIENT_ID = "Client-02"
VIDEO_SOURCE = "C:\\Users\\Nandini Devi\\Downloads\\uropds\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\Train002" 
MODEL_PATH = "ucsd_baseline.h5"
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'model-updates'
BATCH_SIZE = 100 # How many frames to process before sending an update
IMG_SIZE = 256
# -------------------

print("Initializing Federated Client...")

# 1. Load the pre-trained Anomaly Detection Model
print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)

model.compile(optimizer='adam', loss='mean_squared_error')

# 2. Initialize the Kafka Producer
print(f"Connecting to Kafka at {KAFKA_SERVER}...")
producer = KafkaProducer(
    bootstrap_servers=KAFKA_SERVER,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 3. Open the video stream
# 3. Load the image frames from the dataset folder
print(f"Loading frames from: {VIDEO_SOURCE}")
frame_files = [f for f in sorted(os.listdir(VIDEO_SOURCE)) if f.endswith('.tif') or f.endswith('.jpg')]
if not frame_files:
    print(f"Error: No image files found in {VIDEO_SOURCE}")
    exit()

print(f"Client is running. Processing {len(frame_files)} frames in a loop...")

frames_batch = []
frame_count = 0
frame_index = 0

while True:
    # Load the next frame from the list
    frame_path = os.path.join(VIDEO_SOURCE, frame_files[frame_index])
    frame = cv2.imread(frame_path)

    # Move to the next frame and loop back to the start if we reach the end
    frame_index = (frame_index + 1) % len(frame_files)

    # Preprocess the frame for the model
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
    normalized_frame = resized_frame.astype('float32') / 255.0
    
    frames_batch.append(normalized_frame)
    frame_count += 1

    # When the batch is full, train and send an update
    if frame_count >= BATCH_SIZE:
        print(f"\nCollected a batch of {len(frames_batch)} frames. Performing local training...")
        
        # Convert the batch of frames into the correct shape for the model
        train_data = np.array(frames_batch)
        train_data = np.reshape(train_data, (len(train_data), IMG_SIZE, IMG_SIZE, 1))

        # 4. Perform a quick local training update (1 epoch)
        model.fit(train_data, train_data, epochs=1, batch_size=16, verbose=0)
        print("Local training complete.")

        # 5. Extract the model's new weights (the "learnings")
        new_weights = model.get_weights()
        
        # Convert NumPy arrays to lists so they can be sent as JSON
        serializable_weights = [w.tolist() for w in new_weights]

        # 6. Send the learnings to Kafka
        update_message = {
            'client_id': CLIENT_ID,
            'weights': serializable_weights
        }
        
        print(f"Sending model update to Kafka topic: '{KAFKA_TOPIC}'...")
        producer.send(KAFKA_TOPIC, value=update_message)
        producer.flush()
        print("Update sent successfully.")

        # Reset for the next batch
        frames_batch = []
        frame_count = 0

    # Optional: Display the video stream
    cv2.imshow('Federated Client View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Client shutting down.")
cap.release()
cv2.destroyAllWindows()