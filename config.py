"""
Central configuration file for FedGuard project.
All team members should copy config.template.py and rename to config.py
"""
import os

# ============ PATHS ============
# TODO: Each team member should update these paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = "C:\\Users\\Nandini Devi\\Downloads\\uropds\\UCSD_Anomaly_Dataset.v1p2"

# Auto-generated paths (don't modify)
UCSD_PED1_TRAIN = os.path.join(DATASET_ROOT, "UCSDped1", "Train")
UCSD_PED1_TEST = os.path.join(DATASET_ROOT, "UCSDped1", "Test")
UCSD_PED2_TRAIN = os.path.join(DATASET_ROOT, "UCSDped2", "Train")
UCSD_PED2_TEST = os.path.join(DATASET_ROOT, "UCSDped2", "Test")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============ MODEL PARAMETERS ============
IMG_SIZE = 256
LATENT_DIM = 128  # Bottleneck size in autoencoder
EPOCHS = 20  # Increased from 10
BATCH_SIZE = 32  # Increased from 16
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============ FEDERATED LEARNING ============
NUM_CLIENTS = 3
NUM_FEDERATED_ROUNDS = 5
FRAMES_PER_UPDATE = 100

# ============ KAFKA CONFIGURATION ============
KAFKA_SERVER = 'localhost:9092'
KAFKA_TOPIC = 'model-updates'
KAFKA_TIMEOUT = 30000  # 30 seconds

# ============ SECURITY PARAMETERS ============
SIMILARITY_THRESHOLD = 0.5  # Will be replaced with adaptive method
USE_ADAPTIVE_THRESHOLD = True
MAD_THRESHOLD = 3.5  # Median Absolute Deviation threshold
MIN_HONEST_CLIENTS = 2  # Minimum clients needed to proceed

# ============ EVALUATION ============
ANOMALY_PERCENTILE = 95  # Top 5% reconstruction errors are anomalies
TEST_FOLDERS = {
    'ped1': 'Test001',  # Contains anomalies
    'ped2': 'Test001'
}

# ============ LOGGING ============
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
