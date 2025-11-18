"""
Train a high-quality baseline autoencoder model using the UCSD dataset.
Uses data generators, validation split, and proper callbacks.
"""
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
import json
from datetime import datetime
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from results.models.anomaly_model import create_autoencoder
from utils.data_loader import VideoFrameGenerator

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'training.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        logger.warning("Could not enable GPU memory growth")

def train_baseline_model(dataset='ped1', save_name='ucsd_baseline.h5'):
    logger.info("="*60)
    logger.info("FEDGUARD BASELINE MODEL TRAINING")
    logger.info("="*60)

    if dataset == 'ped1':
        train_path = UCSD_PED1_TRAIN
    elif dataset == 'ped2':
        train_path = UCSD_PED2_TRAIN
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info(f"Dataset: {dataset}")
    logger.info(f"Training data: {train_path}")
    logger.info(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Batch size: {BATCH_SIZE}")

    logger.info("\nCreating data generators...")
    train_generator = VideoFrameGenerator(
        train_path,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    train_folders = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    num_val_folders = max(1, int(len(train_folders) * VALIDATION_SPLIT))
    val_folders = train_folders[-num_val_folders:] if train_folders else []

    # Build validation file list
    val_frame_paths = []
    for vf in val_folders:
        vfp = os.path.join(train_path, vf)
        if os.path.exists(vfp):
            for root, _, files in os.walk(vfp):
                for f in sorted(files):
                    if f.endswith(('.tif', '.jpg', '.png')):
                        val_frame_paths.append(os.path.join(root, f))

    if len(val_frame_paths) == 0:
        logger.warning("No validation frames found; using a split of train generator as fallback.")
        val_generator = VideoFrameGenerator(train_path, img_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
        val_samples = len(val_generator.frame_paths)
    else:
        # Simple list-based val generator
        class SimpleListFrameGenerator(tf.keras.utils.Sequence):
            def __init__(self, paths, img_size, batch_size):
                self.paths = paths
                self.img_size = img_size
                self.batch_size = batch_size
                self.indices = np.arange(len(self.paths))

            def __len__(self):
                return int(np.ceil(len(self.paths) / self.batch_size))

            def __getitem__(self, idx):
                batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_paths = [self.paths[i] for i in batch_indices]
                batch = []
                for p in batch_paths:
                    f = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if f is None:
                        continue
                    f = cv2.resize(f, (self.img_size, self.img_size))
                    f = f.astype('float32') / 255.0
                    batch.append(f)
                if len(batch) == 0:
                    return np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32), np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32)
                X = np.array(batch).reshape(-1, self.img_size, self.img_size, 1)
                return X, X

        val_generator = SimpleListFrameGenerator(val_frame_paths, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
        val_samples = len(val_frame_paths)

    train_samples = len(train_generator.frame_paths) if hasattr(train_generator, 'frame_paths') else int(np.ceil(len(train_generator) * BATCH_SIZE))
    logger.info(f"Training samples (approx): {train_samples}")
    logger.info(f"Validation samples (approx): {val_samples}")

    logger.info("\nCreating autoencoder model...")
    model = create_autoencoder(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    model_save_path = os.path.join(MODELS_DIR, save_name)
    callbacks = [
        ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    start_time = datetime.now()
    steps_per_epoch = int(np.ceil(train_samples / BATCH_SIZE))
    validation_steps = int(np.ceil(val_samples / BATCH_SIZE)) if val_samples > 0 else None

    try:
        history = model.fit(
            train_generator,
            validation_data=val_generator if val_samples>0 else None,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            workers=4,
            use_multiprocessing=False
        )
    except tf.errors.ResourceExhaustedError:
        logger.error("OOM during training. Consider lowering BATCH_SIZE or IMG_SIZE.", exc_info=True)
        raise
    except Exception as e:
        logger.error("Training failed with exception", exc_info=True)
        try:
            fallback_path = os.path.join(MODELS_DIR, save_name.replace('.h5', '_partial.h5'))
            model.save(fallback_path)
            logger.info(f"Saved partial model to {fallback_path}")
        except Exception:
            logger.warning("Failed to save partial model")
        raise

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    history_path = os.path.join(METRICS_DIR, 'training_history.json')
    history_dict = {
        'loss': [float(x) for x in history.history.get('loss', [])],
        'val_loss': [float(x) for x in history.history.get('val_loss', [])],
        'training_time_seconds': training_time,
        'dataset': dataset,
        'img_size': IMG_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    logger.info("\nTRAINING COMPLETE")
    logger.info(f"Training time: {training_time:.2f} seconds")
    if history.history.get('loss'):
        logger.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
    if history.history.get('val_loss'):
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        logger.info(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"History saved to: {history_path}")

    return model, history

if __name__ == "__main__":
    model, history = train_baseline_model(dataset='ped1', save_name='ucsd_baseline.h5')
    print("\n✅ Baseline model training complete!")
    print(f"Model ready for federated learning at: {os.path.join(MODELS_DIR, 'ucsd_baseline.h5')}")
