import os

# ============ PATHS ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Update this path on your machine if necessary:
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
IMG_SIZE = 128
LATENT_DIM = 128
EPOCHS = 1
BATCH_SIZE = 4
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ============ FEDERATED LEARNING ============
NUM_CLIENTS = 4
NUM_FEDERATED_ROUNDS = 5
FRAMES_PER_UPDATE = 16

# ============ KAFKA CONFIGURATION ============
KAFKA_SERVER = 'localhost:9092'   # or 'broker-ip:9092' if remote
KAFKA_TOPIC = 'model-updates'
KAFKA_TIMEOUT = 30000  # milliseconds (30 seconds)

# For large messages: increase these on both client and broker if needed
KAFKA_MAX_REQUEST_BYTES = 60_000_000
KAFKA_FETCH_MAX_BYTES = 60_000_000

# ============ SECURITY PARAMETERS ============
SIMILARITY_THRESHOLD = 0.5
USE_ADAPTIVE_THRESHOLD = True
MAD_THRESHOLD = 3.5
MIN_HONEST_CLIENTS = 1

# ============ EVALUATION ============
ANOMALY_PERCENTILE = 95
TEST_FOLDERS = {
    'ped1': 'Test003',
    'ped2': 'Test003'
}

# ============ LOGGING ============
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# ============ SECURITY / DEFENSE (server) ============
# Defense method on server: one of "cosine", "mad", "krum", "median", "trimmed_mean"
DEFENSE_METHOD = "mad"

# Aggregation method used after filtering: "mean", "median", or "trimmed_mean"
AGGREGATION_METHOD = "mean"

