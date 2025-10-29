"""
Efficient data loading with generators to avoid memory issues.
"""
import os
import cv2
import numpy as np
import logging
from tensorflow.keras.utils import Sequence

logger = logging.getLogger(__name__)

class VideoFrameGenerator(Sequence):
    """
    Keras Sequence for loading video frames on-the-fly.
    Prevents memory overflow with large datasets.
    """
    def __init__(self, data_path, img_size=256, batch_size=32, shuffle=True):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Load all frame paths
        self.frame_paths = []
        self._load_frame_paths()
        
        self.indices = np.arange(len(self.frame_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _load_frame_paths(self):
        """Recursively find all image files."""
        if os.path.isfile(self.data_path):
            # Single video file
            logger.error("Video file input not yet supported in generator")
            return
        
        # Directory of images
        for root, dirs, files in os.walk(self.data_path):
            for filename in sorted(files):
                if filename.endswith(('.tif', '.jpg', '.png')):
                    self.frame_paths.append(os.path.join(root, filename))
        
        logger.info(f"Found {len(self.frame_paths)} frames in {self.data_path}")
    
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.frame_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get one batch of data."""
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = [self.frame_paths[i] for i in batch_indices]
        
        # Load and preprocess batch
        batch_frames = []
        for path in batch_paths:
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.astype('float32') / 255.0
                batch_frames.append(frame)
        
        X = np.array(batch_frames).reshape(-1, self.img_size, self.img_size, 1)
        return X, X  # Autoencoder: input = output
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_test_frames(test_folder, img_size=256):
    """
    Load all test frames into memory (test sets are small).
    
    Returns:
        frames: numpy array of preprocessed frames
        frame_names: list of filenames for reference
    """
    frames = []
    frame_names = []
    
    for filename in sorted(os.listdir(test_folder)):
        if filename.endswith(('.tif', '.jpg', '.png')):
            path = os.path.join(test_folder, filename)
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (img_size, img_size))
                frame = frame.astype('float32') / 255.0
                frames.append(frame)
                frame_names.append(filename)
    
    frames = np.array(frames).reshape(-1, img_size, img_size, 1)
    logger.info(f"Loaded {len(frames)} test frames from {test_folder}")
    
    return frames, frame_names


def get_ground_truth_labels(test_folder):
    """
    Load ground truth labels for UCSD dataset.
    UCSD provides frame-level labels in Test***_gt folders.
    
    Returns:
        labels: Binary array (0=normal, 1=anomaly) for each frame
    """
    # Try to find ground truth file
    gt_folder = test_folder + "_gt"
    
    if not os.path.exists(gt_folder):
        logger.warning(f"Ground truth not found at {gt_folder}")
        # Return all zeros (assume all normal) as fallback
        num_frames = len([f for f in os.listdir(test_folder) if f.endswith(('.tif', '.jpg'))])
        return np.zeros(num_frames)
    
    # UCSD format: Each frame has a corresponding .bmp mask
    # If mask exists and is not all black, frame contains anomaly
    labels = []
    frame_files = sorted([f for f in os.listdir(test_folder) if f.endswith(('.tif', '.jpg'))])
    
    for frame_file in frame_files:
        # Ground truth files have same name but .bmp extension
        gt_file = os.path.splitext(frame_file)[0] + ".bmp"
        gt_path = os.path.join(gt_folder, gt_file)
        
        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # If mask has any white pixels, it's an anomaly
            has_anomaly = np.any(gt_mask > 0)
            labels.append(1 if has_anomaly else 0)
        else:
            labels.append(0)  # Assume normal if no GT
    
    num_anomalies = sum(labels)
    logger.info(f"Ground truth: {num_anomalies}/{len(labels)} anomaly frames")
    
    return np.array(labels)