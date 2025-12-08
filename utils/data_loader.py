# """
# Efficient data loading with generators to avoid memory issues.
# """
# import os
# import cv2
# import numpy as np
# import logging
# import tensorflow as tf
# from tensorflow.keras.utils import Sequence

# logger = logging.getLogger(__name__)

# class VideoFrameGenerator(Sequence):
#     """
#     Keras Sequence for loading video frames on-the-fly.
#     Prevents memory overflow with large datasets.
#     """
#     def __init__(self, data_path, img_size=256, batch_size=32, shuffle=True):
#         self.data_path = data_path
#         self.img_size = img_size
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#         self.frame_paths = []
#         self._load_frame_paths()

#         self.indices = np.arange(len(self.frame_paths))
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#     def _load_frame_paths(self):
#         """Recursively find all image files."""
#         if os.path.isfile(self.data_path):
#             logger.error("Video file input not yet supported in generator")
#             return

#         for root, dirs, files in os.walk(self.data_path):
#             for filename in sorted(files):
#                 if filename.endswith(('.tif', '.jpg', '.png')):
#                     self.frame_paths.append(os.path.join(root, filename))

#         logger.info(f"Found {len(self.frame_paths)} frames in {self.data_path}")

#     def __len__(self):
#         """Number of batches per epoch."""
#         if self.batch_size <= 0:
#             return 0
#         return int(np.ceil(len(self.frame_paths) / self.batch_size))

#     def __getitem__(self, idx):
#         """Get one batch of data."""
#         if len(self.frame_paths) == 0:
#             X = np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32)
#             return X, X

#         batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_indices = [i % len(self.frame_paths) for i in batch_indices]
#         batch_paths = [self.frame_paths[i] for i in batch_indices]

#         batch_frames = []
#         for path in batch_paths:
#             frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             if frame is not None:
#                 frame = cv2.resize(frame, (self.img_size, self.img_size))
#                 frame = frame.astype('float32') / 255.0
#                 batch_frames.append(frame)
#             else:
#                 logger.warning(f"Could not read frame in generator: {path}")

#         if len(batch_frames) == 0:
#             X = np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32)
#         else:
#             X = np.array(batch_frames).reshape(-1, self.img_size, self.img_size, 1)

#         return X, X

#     def on_epoch_end(self):
#         """Shuffle indices after each epoch."""
#         if self.shuffle:
#             np.random.shuffle(self.indices)


# def load_test_frames(test_folder, img_size=256):
#     """
#     Load all test frames into memory (test sets are small).
#     Returns frames (N,H,W,1) and frame_names.
#     """
#     frames = []
#     frame_names = []

#     for filename in sorted(os.listdir(test_folder)):
#         if filename.endswith(('.tif', '.jpg', '.png')):
#             path = os.path.join(test_folder, filename)
#             frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             if frame is not None:
#                 frame = cv2.resize(frame, (img_size, img_size))
#                 frame = frame.astype('float32') / 255.0
#                 frames.append(frame)
#                 frame_names.append(filename)

#     if len(frames) == 0:
#         logger.warning(f"No frames found in test folder: {test_folder}")

#     frames = np.array(frames).reshape(-1, img_size, img_size, 1) if frames else np.zeros((0, img_size, img_size, 1), dtype=np.float32)
#     logger.info(f"Loaded {len(frames)} test frames from {test_folder}")
#     return frames, frame_names


# def get_ground_truth_labels(test_folder):
#     """
#     Load ground truth labels for UCSD dataset.
#     Returns binary array (0=normal,1=anomaly)
#     """
#     gt_folder = test_folder + "_gt"

#     if not os.path.exists(gt_folder):
#         logger.warning(f"Ground truth not found at {gt_folder}")
#         num_frames = len([f for f in os.listdir(test_folder) if f.endswith(('.tif', '.jpg', '.png'))])
#         return np.zeros(num_frames, dtype=int)

#     labels = []
#     frame_files = sorted([f for f in os.listdir(test_folder) if f.endswith(('.tif', '.jpg', '.png'))])

#     for frame_file in frame_files:
#         gt_file = os.path.splitext(frame_file)[0] + ".bmp"
#         gt_path = os.path.join(gt_folder, gt_file)
#         if os.path.exists(gt_path):
#             gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
#             has_anomaly = np.any(gt_mask > 0)
#             labels.append(1 if has_anomaly else 0)
#         else:
#             labels.append(0)

#     num_anomalies = sum(labels)
#     logger.info(f"Ground truth: {num_anomalies}/{len(labels)} anomaly frames")
#     return np.array(labels, dtype=int)

"""
Efficient data loading with generators to avoid memory issues.
"""
import os
import cv2
import numpy as np
import logging
import tensorflow as tf
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

        self.frame_paths = []
        self._load_frame_paths()

        self.indices = np.arange(len(self.frame_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_frame_paths(self):
        """Recursively find all image files."""
        if os.path.isfile(self.data_path):
            logger.error("Video file input not yet supported in generator")
            return

        for root, dirs, files in os.walk(self.data_path):
            for filename in sorted(files):
                if filename.endswith(('.tif', '.jpg', '.png')):
                    self.frame_paths.append(os.path.join(root, filename))

        logger.info(f"Found {len(self.frame_paths)} frames in {self.data_path}")

    def __len__(self):
        """Number of batches per epoch."""
        if self.batch_size <= 0:
            return 0
        return int(np.ceil(len(self.frame_paths) / self.batch_size))

    def __getitem__(self, idx):
        """Get one batch of data."""
        if len(self.frame_paths) == 0:
            X = np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32)
            return X, X

        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_indices = [i % len(self.frame_paths) for i in batch_indices]
        batch_paths = [self.frame_paths[i] for i in batch_indices]

        batch_frames = []
        for path in batch_paths:
            frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = frame.astype('float32') / 255.0
                batch_frames.append(frame)
            else:
                logger.warning(f"Could not read frame in generator: {path}")

        if len(batch_frames) == 0:
            X = np.zeros((0, self.img_size, self.img_size, 1), dtype=np.float32)
        else:
            X = np.array(batch_frames).reshape(-1, self.img_size, self.img_size, 1)

        return X, X

    def on_epoch_end(self):
        """Shuffle indices after each epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_test_frames(test_folder, img_size=256):
    """
    Load all test frames into memory (test sets are small).
    UPDATED: Now searches recursively in subfolders.
    Returns frames (N,H,W,1) and frame_names.
    """
    frames = []
    frame_names = []

    # Use os.walk to find files in subdirectories (Test001, Test002, etc.)
    for root, dirs, files in os.walk(test_folder):
        for filename in sorted(files):
            if filename.endswith(('.tif', '.jpg', '.png')):
                path = os.path.join(root, filename)
                try:
                    frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frame = cv2.resize(frame, (img_size, img_size))
                        frame = frame.astype('float32') / 255.0
                        frames.append(frame)
                        frame_names.append(filename)
                except Exception as e:
                    logger.warning(f"Failed to load image {path}: {e}")

    if len(frames) == 0:
        logger.warning(f"No frames found in test folder (checked subdirs): {test_folder}")

    frames = np.array(frames).reshape(-1, img_size, img_size, 1) if frames else np.zeros((0, img_size, img_size, 1), dtype=np.float32)
    logger.info(f"Loaded {len(frames)} test frames from {test_folder}")
    return frames, frame_names


def get_ground_truth_labels(test_folder):
    """
    Load ground truth labels for UCSD dataset.
    Returns binary array (0=normal,1=anomaly)
    """
    gt_folder = test_folder + "_gt"

    if not os.path.exists(gt_folder):
        logger.warning(f"Ground truth not found at {gt_folder}")
        # Return zeros based on however many files we might find (rough estimate)
        return np.array([], dtype=int)

    labels = []
    # Note: GT files are usually in the root of _gt folder, not subdirs
    frame_files = sorted([f for f in os.listdir(test_folder) if f.endswith(('.tif', '.jpg', '.png'))])

    # If frame_files is empty (because they are in subdirs), we might need a different strategy
    # For now, this function assumes standard UCSD structure for GT which is tricky.
    # We will just walk the GT folder directly.
    
    for root, dirs, files in os.walk(gt_folder):
        for filename in sorted(files):
            if filename.endswith('.bmp'):
                gt_path = os.path.join(root, filename)
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    has_anomaly = np.any(gt_mask > 0)
                    labels.append(1 if has_anomaly else 0)
    
    # If no GT labels found, return empty or zeros
    if not labels:
        return np.array([], dtype=int)

    num_anomalies = sum(labels)
    logger.info(f"Ground truth: {num_anomalies}/{len(labels)} anomaly frames")
    return np.array(labels, dtype=int)