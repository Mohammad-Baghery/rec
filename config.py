# config.py

import os

# --- Project Root ---
# The absolute path to the directory where this config.py file is located.
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- General Settings ---
APP_VERSION = "2.0.0"
EMBEDDING_DIMENSION = 512 # FaceNet model's output dimension

# --- Database Settings ---
# You can use a local MongoDB instance or a cloud-based one like MongoDB Atlas.
# Example for a local instance: "mongodb://localhost:27017/"
# Example for MongoDB Atlas: "mongodb+srv://<user>:<password>@<cluster-url>/?retryWrites=true&w=majority"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DATABASE_NAME = "face_identity_db"

# --- Annoy Index Settings ---
# These files will be created inside the 'database' folder.
ANNOY_INDEX_FILE = os.path.join(BASE_PROJECT_DIR, "database", "identity_index.ann")
ANNOY_NUM_TREES = 20 # More trees give higher precision but take longer to build.

# --- Paths ---
RESULTS_FOLDER = os.path.join(BASE_PROJECT_DIR, "results")
SORTING_OUTPUT_FOLDER = os.path.join(BASE_PROJECT_DIR, "sorted_faces_output")
LOGS_FOLDER = os.path.join(BASE_PROJECT_DIR, "logs")

# --- Logging Settings ---
LOG_FILE_PATH = os.path.join(LOGS_FOLDER, "app_log.log")
CONSOLE_LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
FILE_LOG_LEVEL = "DEBUG"

# --- Default Operation Parameters ---
DEFAULT_THRESHOLD = 0.7
DEFAULT_SORTING_THRESHOLD = 0.85
DEFAULT_MIN_FACES_PER_GROUP = 2
DEFAULT_BATCH_PROCESSING_ENABLED = True
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MAX_DISPLAY_RESULTS = 10
DEFAULT_MERGE_FOLDERS_ENABLED = True

# --- Status Constants ---
STATUS_IDLE = "IDLE"
STATUS_INITIALIZING = "INITIALIZING"
STATUS_WAITING = "WAITING"
STATUS_PROCESSING = "PROCESSING"
STATUS_COMPLETE = "COMPLETE"
STATUS_ERROR = "ERROR"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_PAUSED = "PAUSED"
