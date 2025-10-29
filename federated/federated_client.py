"""
Honest Federated Learning Client.
Performs local training and sends updates to the server.
"""
import os
import sys
import cv2
import numpy as np
import json
import tensorflow as tf
from kafka import KafkaProducer
import logging
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'client_{time.time()}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
CLIENT_ID = "Client-01"  # CHANGE THIS FOR EACH CLIENT (Client-01, Client-02, etc.)
VIDEO_SOURCE = os.path.join(UCSD_PED1_TRAIN, "Train001")  # CHANGE FOR EACH CLIENT
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")
# -------------------

logger.info("="*60)
logger.info(f"FEDERATED CLIENT: {CLIENT_ID}")
logger.info("="*60)

try:
    # 1. Load the pre-trained model
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("✓ Model loaded successfully")
    
    # 2. Initialize Kafka Producer
    logger.info(f"Connecting to Kafka at {KAFKA_SERVER}...")
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        request_timeout_ms=KAFKA_TIMEOUT
    )
    logger.info("✓ Kafka connection established")
    
    # 3. Load frames from dataset
    logger.info(f"Loading frames from: {VIDEO_SOURCE}")
    frame_files = sorted([f for f in os.listdir(VIDEO_SOURCE) 
                         if f.endswith(('.tif', '.jpg', '.png'))])
    
    if not frame_files:
        logger.error(f"No image files found in {VIDEO_SOURCE}")
        sys.exit(1)
    
    logger.info(f"✓ Found {len(frame_files)} frames")
    logger.info(f"Batch size: {FRAMES_PER_UPDATE} frames")
    logger.info("\n" + "="*60)
    logger.info("CLIENT RUNNING - Processing frames in loop")
    logger.info("="*60 + "\n")
    
    frames_batch = []
    frame_count = 0
    frame_index = 0
    update_count = 0
    total_training_time = 0
    
    while True:
        # Load next frame
        frame_path = os.path.join(VIDEO_SOURCE, frame_files[frame_index])
        frame = cv2.imread(frame_path)
        
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}")
            frame_index = (frame_index + 1) % len(frame_files)
            continue
        
        # Move to next frame (loop)
        frame_index = (frame_index + 1) % len(frame_files)
        
        # Preprocess
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
        normalized_frame = resized_frame.astype('float32') / 255.0
        
        frames_batch.append(normalized_frame)
        frame_count += 1
        
        # When batch is full, train and send update
        if frame_count >= FRAMES_PER_UPDATE:
            logger.info(f"\n[Update #{update_count + 1}] Collected {len(frames_batch)} frames")
            
            # Convert to model input format
            train_data = np.array(frames_batch)
            train_data = np.reshape(train_data, (len(train_data), IMG_SIZE, IMG_SIZE, 1))
            
            # Local training
            logger.info("Performing local training...")
            train_start = time.time()
            
            history = model.fit(
                train_data, train_data,
                epochs=1,
                batch_size=16,
                verbose=0
            )
            
            train_time = time.time() - train_start
            total_training_time += train_time
            
            loss = history.history['loss'][0]
            logger.info(f"✓ Training complete - Loss: {loss:.6f}, Time: {train_time:.2f}s")
            
            # Extract weights
            new_weights = model.get_weights()
            serializable_weights = [w.tolist() for w in new_weights]
            
            # Send to Kafka
            update_message = {
                'client_id': CLIENT_ID,
                'weights': serializable_weights,
                'metadata': {
                    'update_number': update_count + 1,
                    'training_loss': float(loss),
                    'training_time': train_time,
                    'num_frames': len(frames_batch)
                }
            }
            
            logger.info(f"Sending update to Kafka topic: '{KAFKA_TOPIC}'...")
            send_start = time.time()
            
            producer.send(KAFKA_TOPIC, value=update_message)
            producer.flush()
            
            send_time = time.time() - send_start
            logger.info(f"✓ Update sent successfully (Send time: {send_time:.2f}s)")
            
            # Stats
            update_count += 1
            avg_training_time = total_training_time / update_count
            logger.info(f"Total updates sent: {update_count}")
            logger.info(f"Avg training time: {avg_training_time:.2f}s")
            
            # Reset batch
            frames_batch = []
            frame_count = 0
        
        # Display frame (optional)
        cv2.imshow(f'Federated Client: {CLIENT_ID}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("\nUser pressed 'q' - shutting down client")
            break

except KeyboardInterrupt:
    logger.info("\n\nKeyboard interrupt received - shutting down client")
except Exception as e:
    logger.error(f"\n\nERROR: {str(e)}", exc_info=True)
finally:
    logger.info("\nCleaning up...")
    cv2.destroyAllWindows()
    if 'producer' in locals():
        producer.close()
    logger.info(f"Client {CLIENT_ID} shut down successfully")