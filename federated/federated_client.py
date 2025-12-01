# """
# Honest Federated Learning Client (robust).
# - Uses config.py constants (IMG_SIZE, FRAMES_PER_UPDATE, BATCH_SIZE)
# - Handles OOM during training with fallback
# - Serializes model weights with numpy.savez_compressed + zlib
# - Sends weights as bytes to Kafka topic with metadata in headers
# """
# import os
# import sys
# import cv2
# import numpy as np
# import json
# import logging
# import time
# import io
# import zlib
# import gc

# from kafka import KafkaProducer

# # Add parent dir to path to import config
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from config import *

# # Setup logging
# os.makedirs(LOGS_DIR, exist_ok=True)
# logging.basicConfig(
#     level=getattr(logging, LOG_LEVEL),
#     format=LOG_FORMAT,
#     handlers=[
#         logging.FileHandler(os.path.join(LOGS_DIR, f'client_{int(time.time())}.log'), encoding='utf-8'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("fedguard_client")

# CLIENT_ID = "Client-01"  # CHANGE THIS FOR EACH CLIENT
# VIDEO_SOURCE = os.path.join(UCSD_PED1_TRAIN, "Train001")
# MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")

# def serialize_weights_to_bytes(weights_list):
#     """
#     weights_list: list of numpy arrays
#     Returns: compressed bytes (npz compressed + zlib)
#     """
#     buf = io.BytesIO()
#     savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
#     np.savez_compressed(buf, **savez_dict)
#     buf.seek(0)
#     raw = buf.read()
#     packed = zlib.compress(raw, level=6)
#     return packed

# def main():
#     try:
#         logger.info("="*60)
#         logger.info(f"FEDERATED CLIENT: {CLIENT_ID}")
#         logger.info("="*60)

#         # Load model if available
#         logger.info(f"Loading model from {MODEL_PATH}...")
#         model = None
#         try:
#             if os.path.exists(MODEL_PATH):
#                 import tensorflow as tf
#                 model = tf.keras.models.load_model(MODEL_PATH)
#                 model.compile(optimizer='adam', loss='mean_squared_error')
#                 logger.info("[OK] Model loaded successfully")
#             else:
#                 logger.warning(f"Model path {MODEL_PATH} not found. Client will still run but cannot train without model.")
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}", exc_info=True)
#             model = None

#         # Initialize Kafka Producer
#         logger.info(f"Connecting to Kafka at {KAFKA_SERVER} (topic: {KAFKA_TOPIC})...")
#         producer = KafkaProducer(
#             bootstrap_servers=KAFKA_SERVER,
#             value_serializer=lambda v: v,
#             compression_type='gzip',
#             acks='all',
#             max_request_size=KAFKA_MAX_REQUEST_BYTES,
#             linger_ms=50,
#             request_timeout_ms=KAFKA_TIMEOUT
#         )
#         logger.info("[OK] Kafka producer initialized")

#         # Load frames
#         logger.info(f"Loading frames from: {VIDEO_SOURCE}")
#         if not os.path.isdir(VIDEO_SOURCE):
#             logger.error(f"Video source directory not found: {VIDEO_SOURCE}")
#             return

#         frame_files = sorted([f for f in os.listdir(VIDEO_SOURCE) if f.endswith(('.tif', '.jpg', '.png'))])
#         if not frame_files:
#             logger.error(f"No image files found in {VIDEO_SOURCE}")
#             return

#         logger.info(f"[OK] Found {len(frame_files)} frames")
#         logger.info(f"Frames per update: {FRAMES_PER_UPDATE}, Train batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}")

#         frames_batch = []
#         frame_count = 0
#         frame_index = 0
#         update_count = 0
#         total_training_time = 0

#         import tensorflow as tf
#         from tensorflow.python.framework.errors_impl import ResourceExhaustedError

#         while True:
#             frame_path = os.path.join(VIDEO_SOURCE, frame_files[frame_index])
#             frame = cv2.imread(frame_path)
#             if frame is None:
#                 logger.warning(f"Could not read frame: {frame_path}")
#                 frame_index = (frame_index + 1) % len(frame_files)
#                 continue

#             frame_index = (frame_index + 1) % len(frame_files)
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
#             normalized_frame = resized_frame.astype('float32') / 255.0

#             frames_batch.append(normalized_frame)
#             frame_count += 1

#             if frame_count >= FRAMES_PER_UPDATE:
#                 logger.info(f"\n[Update #{update_count + 1}] Collected {len(frames_batch)} frames")
#                 train_data = np.array(frames_batch)
#                 train_data = np.reshape(train_data, (len(train_data), IMG_SIZE, IMG_SIZE, 1))

#                 if model is None:
#                     logger.warning("No model loaded - sending metadata-only update")
#                     metadata = {
#                         'client_id': CLIENT_ID,
#                         'update_number': update_count + 1,
#                         'training_loss': None,
#                         'training_time': 0.0,
#                         'num_frames': len(frames_batch),
#                         'status': 'no_model',
#                         'timestamp': time.time()
#                     }
#                     producer.send(
#                         KAFKA_TOPIC,
#                         value=b'',
#                         key=CLIENT_ID.encode('utf-8'),
#                         headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
#                     )
#                     producer.flush()
#                     frames_batch = []
#                     frame_count = 0
#                     update_count += 1
#                     continue

#                 # Sanity check: ensure model input shape matches IMG_SIZE
#                 try:
#                     model_input_shape = model.input_shape  # (None, H, W, C)
#                     expected_h = model_input_shape[1]
#                     expected_w = model_input_shape[2]
#                     expected_c = model_input_shape[3] if len(model_input_shape) > 3 else 1
#                     if (IMG_SIZE != expected_h) or (expected_w != expected_w) or (expected_c not in (1,3)):
#                         logger.warning(f"IMG_SIZE mismatch with model input. Model expects {model_input_shape}, client has IMG_SIZE={IMG_SIZE}")
#                         # Resize train_data to model expected dims:
#                         train_data = np.array([cv2.resize(f.squeeze(), (expected_w, expected_h)) for f in train_data])
#                         train_data = train_data.reshape((-1, expected_h, expected_w, expected_c))
#                         logger.info(f"Resized train_data to {train_data.shape} to match model")
#                 except Exception:
#                     pass

#                 # Local training with OOM handling
#                 logger.info("Performing local training...")
#                 train_start = time.time()
#                 loss = None
#                 training_succeeded = False

#                 try:
#                     history = model.fit(
#                         train_data, train_data,
#                         epochs=1,
#                         batch_size=BATCH_SIZE,
#                         verbose=0
#                     )
#                     loss = float(history.history['loss'][0])
#                     training_succeeded = True

#                 except ResourceExhaustedError as oom_err:
#                     logger.warning("ResourceExhaustedError during model.fit() — attempting fallback small-batch training", exc_info=True)
#                     try:
#                         mini_batch = max(1, min(4, BATCH_SIZE))
#                         n_samples = len(train_data)
#                         for i in range(0, n_samples, mini_batch):
#                             xb = train_data[i:i+mini_batch]
#                             if xb.shape[0] == 0:
#                                 continue
#                             model.train_on_batch(xb, xb)
#                         loss = float(model.evaluate(train_data[:mini_batch], train_data[:mini_batch], verbose=0))
#                         training_succeeded = True
#                         logger.info("Fallback train_on_batch succeeded")
#                     except Exception as e2:
#                         logger.error("Fallback training also failed", exc_info=True)
#                         training_succeeded = False

#                 except Exception as e:
#                     logger.error("Unexpected error during training", exc_info=True)
#                     training_succeeded = False

#                 train_time = time.time() - train_start
#                 total_training_time += train_time

#                 if training_succeeded:
#                     logger.info(f"[OK] Training complete - Loss: {loss:.6f}, Time: {train_time:.2f}s")
#                 else:
#                     logger.warning(f"Training failed or OOM. Training time: {train_time:.2f}s")

#                 payload_bytes = b''
#                 metadata = {
#                     'client_id': CLIENT_ID,
#                     'update_number': update_count + 1,
#                     'training_loss': float(loss) if loss is not None else None,
#                     'training_time': train_time,
#                     'num_frames': len(frames_batch),
#                     'status': 'trained' if training_succeeded else 'training_failed',
#                     'timestamp': time.time()
#                 }

#                 try:
#                     new_weights = model.get_weights()
#                     payload_bytes = serialize_weights_to_bytes(new_weights)
#                 except Exception as e:
#                     logger.error(f"Failed to extract/serialize weights: {e}", exc_info=True)
#                     payload_bytes = b''

#                 try:
#                     tf.keras.backend.clear_session()
#                 except Exception:
#                     pass
#                 gc.collect()

#                 logger.info(f"Sending update to Kafka topic: '{KAFKA_TOPIC}' (payload size: {len(payload_bytes)} bytes, status: {metadata['status']})...")
#                 send_start = time.time()
#                 try:
#                     producer.send(
#                         KAFKA_TOPIC,
#                         value=payload_bytes,
#                         key=CLIENT_ID.encode('utf-8'),
#                         headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
#                     )
#                     producer.flush()
#                     send_time = time.time() - send_start
#                     logger.info(f"[OK] Update sent successfully (Send time: {send_time:.2f}s)")
#                 except Exception as send_e:
#                     logger.error(f"Failed to send update to Kafka: {send_e}", exc_info=True)

#                 update_count += 1
#                 avg_training_time = total_training_time / update_count if update_count > 0 else 0.0
#                 logger.info(f"Total updates sent: {update_count}")
#                 logger.info(f"Avg training time: {avg_training_time:.2f}s")

#                 frames_batch = []
#                 frame_count = 0

#             # Optional display
#             try:
#                 cv2.imshow(f'Federated Client: {CLIENT_ID}', frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     logger.info("\nUser pressed 'q' - shutting down client")
#                     break
#             except Exception:
#                 pass

#     except KeyboardInterrupt:
#         logger.info("Keyboard interrupt - shutting down client")
#     except Exception as e:
#         logger.error(f"FATAL ERROR (client): {e}", exc_info=True)
#     finally:
#         logger.info("Cleaning up client...")
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         try:
#             producer.close()
#         except Exception:
#             pass
#         logger.info(f"Client {CLIENT_ID} shut down successfully")

# if __name__ == "__main__":
#     main()


# client_honest.py
import os
import sys
import cv2
import numpy as np
import json
import logging
import time
import io
import zlib
import gc

from kafka import KafkaProducer

# Add parent dir to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'client_{int(time.time())}.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fedguard_client")

CLIENT_ID = "Client-03"  # CHANGE THIS FOR EACH CLIENT
VIDEO_SOURCE = os.path.join(UCSD_PED1_TRAIN, "Train003")
MODEL_PATH = os.path.join(MODELS_DIR, "ucsd_baseline.h5")

def serialize_weights_to_bytes(weights_list):
    """
    weights_list: list of numpy arrays
    Returns: compressed bytes (npz compressed + zlib)
    """
    buf = io.BytesIO()
    savez_dict = {f"arr_{i}": w for i, w in enumerate(weights_list)}
    np.savez_compressed(buf, **savez_dict)
    buf.seek(0)
    raw = buf.read()
    packed = zlib.compress(raw, level=6)
    return packed

def main():
    producer = None
    try:
        logger.info("="*60)
        logger.info(f"FEDERATED CLIENT: {CLIENT_ID}")
        logger.info("="*60)

        # Load model if available
        logger.info(f"Loading model from {MODEL_PATH}...")
        model = None
        try:
            if os.path.exists(MODEL_PATH):
                import tensorflow as tf
                model = tf.keras.models.load_model(MODEL_PATH)
                model.compile(optimizer='adam', loss='mean_squared_error')
                logger.info("[OK] Model loaded successfully")
            else:
                logger.warning(f"Model path {MODEL_PATH} not found. Client will still run but cannot train without model.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            model = None

        # Initialize Kafka Producer
        logger.info(f"Connecting to Kafka at {KAFKA_SERVER} (topic: {KAFKA_TOPIC})...")
        timeout_ms = KAFKA_TIMEOUT if KAFKA_TIMEOUT is not None else 30000
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_SERVER,
            value_serializer=lambda v: v,
            compression_type='gzip',
            acks='all',
            max_request_size=KAFKA_MAX_REQUEST_BYTES,
            linger_ms=50,
            request_timeout_ms=timeout_ms
        )
        logger.info("[OK] Kafka producer initialized")

        # Load frames
        logger.info(f"Loading frames from: {VIDEO_SOURCE}")
        if not os.path.isdir(VIDEO_SOURCE):
            logger.error(f"Video source directory not found: {VIDEO_SOURCE}")
            return

        frame_files = sorted([f for f in os.listdir(VIDEO_SOURCE) if f.lower().endswith(('.tif', '.jpg', '.png'))])
        if not frame_files:
            logger.error(f"No image files found in {VIDEO_SOURCE}")
            return

        logger.info(f"[OK] Found {len(frame_files)} frames")
        logger.info(f"Frames per update: {FRAMES_PER_UPDATE}, Train batch size: {BATCH_SIZE}, Image size: {IMG_SIZE}")

        frames_batch = []
        frame_count = 0
        frame_index = 0
        update_count = 0
        total_training_time = 0

        import tensorflow as tf
        from tensorflow.python.framework.errors_impl import ResourceExhaustedError

        while True:
            frame_path = os.path.join(VIDEO_SOURCE, frame_files[frame_index])
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                frame_index = (frame_index + 1) % len(frame_files)
                continue

            frame_index = (frame_index + 1) % len(frame_files)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (IMG_SIZE, IMG_SIZE))
            normalized_frame = resized_frame.astype('float32') / 255.0

            frames_batch.append(normalized_frame)
            frame_count += 1

            if frame_count >= FRAMES_PER_UPDATE:
                logger.info(f"\n[Update #{update_count + 1}] Collected {len(frames_batch)} frames")
                train_data = np.array(frames_batch)
                train_data = np.reshape(train_data, (len(train_data), IMG_SIZE, IMG_SIZE, 1))

                if model is None:
                    logger.warning("No model loaded - sending metadata-only update")
                    metadata = {
                        'client_id': CLIENT_ID,
                        'update_number': update_count + 1,
                        'training_loss': None,
                        'training_time': 0.0,
                        'num_frames': len(frames_batch),
                        'status': 'no_model',
                        'timestamp': time.time()
                    }
                    producer.send(
                        KAFKA_TOPIC,
                        value=b'',
                        key=CLIENT_ID.encode('utf-8'),
                        headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
                    )
                    producer.flush()
                    frames_batch = []
                    frame_count = 0
                    update_count += 1
                    continue

                # Sanity check: ensure model input shape matches IMG_SIZE
                try:
                    model_input_shape = model.input_shape  # (None, H, W, C)
                    expected_h = model_input_shape[1]
                    expected_w = model_input_shape[2]
                    expected_c = model_input_shape[3] if len(model_input_shape) > 3 else 1
                    if (IMG_SIZE != expected_h) or (IMG_SIZE != expected_w) or (expected_c not in (1, 3)):
                        logger.warning(f"IMG_SIZE mismatch with model input. Model expects {model_input_shape}, client has IMG_SIZE={IMG_SIZE}")
                        # Resize train_data to model expected dims:
                        train_data = np.array([cv2.resize(f.squeeze(), (expected_w, expected_h)) for f in train_data])
                        if expected_c == 1:
                            train_data = train_data.reshape((-1, expected_h, expected_w, 1))
                        else:
                            # duplicate single-channel into expected_c channels if needed
                            train_data = np.stack([train_data] * expected_c, axis=-1)
                        logger.info(f"Resized train_data to {train_data.shape} to match model")
                except Exception:
                    pass

                # Local training with OOM handling
                logger.info("Performing local training...")
                train_start = time.time()
                loss = None
                training_succeeded = False

                try:
                    history = model.fit(
                        train_data, train_data,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=0
                    )
                    loss = float(history.history['loss'][0])
                    training_succeeded = True

                except ResourceExhaustedError:
                    logger.warning("ResourceExhaustedError during model.fit() — attempting fallback small-batch training")
                    try:
                        mini_batch = max(1, min(4, BATCH_SIZE))
                        n_samples = len(train_data)
                        for i in range(0, n_samples, mini_batch):
                            xb = train_data[i:i+mini_batch]
                            if xb.shape[0] == 0:
                                continue
                            model.train_on_batch(xb, xb)
                        loss = float(model.evaluate(train_data[:mini_batch], train_data[:mini_batch], verbose=0))
                        training_succeeded = True
                        logger.info("Fallback train_on_batch succeeded")
                    except Exception:
                        logger.error("Fallback training also failed", exc_info=True)
                        training_succeeded = False

                except Exception:
                    logger.error("Unexpected error during training", exc_info=True)
                    training_succeeded = False

                train_time = time.time() - train_start
                total_training_time += train_time

                if training_succeeded:
                    logger.info(f"[OK] Training complete - Loss: {loss:.6f}, Time: {train_time:.2f}s")
                else:
                    logger.warning(f"Training failed or OOM. Training time: {train_time:.2f}s")

                payload_bytes = b''
                metadata = {
                    'client_id': CLIENT_ID,
                    'update_number': update_count + 1,
                    'training_loss': float(loss) if loss is not None else None,
                    'training_time': train_time,
                    'num_frames': len(frames_batch),
                    'status': 'trained' if training_succeeded else 'training_failed',
                    'timestamp': time.time()
                }

                try:
                    new_weights = model.get_weights()
                    payload_bytes = serialize_weights_to_bytes(new_weights)
                except Exception as e:
                    logger.error(f"Failed to extract/serialize weights: {e}", exc_info=True)
                    payload_bytes = b''

                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
                gc.collect()

                logger.info(f"Sending update to Kafka topic: '{KAFKA_TOPIC}' (payload size: {len(payload_bytes)} bytes, status: {metadata['status']})...")
                send_start = time.time()
                try:
                    fut = producer.send(
                        KAFKA_TOPIC,
                        value=payload_bytes,
                        key=CLIENT_ID.encode('utf-8'),
                        headers=[('metadata', json.dumps(metadata).encode('utf-8'))]
                    )
                    fut.get(timeout=10)
                    producer.flush()
                    send_time = time.time() - send_start
                    logger.info(f"[OK] Update sent successfully (Send time: {send_time:.2f}s)")
                except Exception as send_e:
                    logger.error(f"Failed to send update to Kafka: {send_e}", exc_info=True)

                update_count += 1
                avg_training_time = total_training_time / update_count if update_count > 0 else 0.0
                logger.info(f"Total updates sent: {update_count}")
                logger.info(f"Avg training time: {avg_training_time:.2f}s")

                frames_batch = []
                frame_count = 0

            # Optional display
            try:
                cv2.imshow(f'Federated Client: {CLIENT_ID}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("\nUser pressed 'q' - shutting down client")
                    break
            except Exception:
                pass

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt - shutting down client")
    except Exception as e:
        logger.error(f"FATAL ERROR (client): {e}", exc_info=True)
    finally:
        logger.info("Cleaning up client...")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if producer is not None:
                producer.close()
        except Exception:
            pass
        logger.info(f"Client {CLIENT_ID} shut down successfully")

if __name__ == "__main__":
    main()
