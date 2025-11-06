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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from results.models.anomaly_model import create_autoencoder  # ✅ FIXED THIS LINE
from utils.data_loader import VideoFrameGenerator

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def train_baseline_model(dataset='ped1', save_name='ucsd_baseline.h5'):
    """
    Train the baseline model with best practices.
    
    Args:
        dataset: 'ped1' or 'ped2'
        save_name: Output model filename
    """
    logger.info("="*60)
    logger.info("FEDGUARD BASELINE MODEL TRAINING")
    logger.info("="*60)
    
    # Select dataset
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
    
    # Create data generators
    logger.info("\nCreating data generators...")
    train_generator = VideoFrameGenerator(
        train_path,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    # Create validation generator (split from training data)
    # We'll use a subset of training folders for validation
    train_folders = sorted([d for d in os.listdir(train_path) 
                           if os.path.isdir(os.path.join(train_path, d))])
    
    num_val_folders = max(1, int(len(train_folders) * VALIDATION_SPLIT))
    val_folders = train_folders[-num_val_folders:]
    
    # Create temporary validation path
    val_paths = [os.path.join(train_path, f) for f in val_folders]
    
    # For simplicity, we'll just create a validation generator from last folder
    val_generator = VideoFrameGenerator(
        os.path.join(train_path, val_folders[0]),
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    logger.info(f"Training samples: ~{len(train_generator) * BATCH_SIZE}")
    logger.info(f"Validation samples: ~{len(val_generator) * BATCH_SIZE}")
    
    # Create model
    logger.info("\nCreating autoencoder model...")
    model = create_autoencoder(input_shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # Callbacks
    model_save_path = os.path.join(MODELS_DIR, save_name)
    callbacks = [
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60 + "\n")
    
    start_time = datetime.now()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save training history
    history_path = os.path.join(METRICS_DIR, 'training_history.json')
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'training_time_seconds': training_time,
        'dataset': dataset,
        'img_size': IMG_SIZE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Training time: {training_time:.2f} seconds")
    logger.info(f"Final training loss: {history.history['loss'][-1]:.6f}")
    logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    logger.info(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"History saved to: {history_path}")
    
    return model, history


if __name__ == "__main__":
    # Train on Ped1 dataset
    model, history = train_baseline_model(dataset='ped1', save_name='ucsd_baseline.h5')
    
    print("\n✅ Baseline model training complete!")
    print(f"Model ready for federated learning at: {os.path.join(MODELS_DIR, 'ucsd_baseline.h5')}")