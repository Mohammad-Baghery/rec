# config.py

import os

# BASE_PROJECT_DIR: آدرس مطلق پوشه ای که این فایل (config.py) در آن قرار دارد.
# این پوشه همان ریشه پروژه شماست که main.py, app2.py و src/ هم در آن قرار دارند.
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- General Settings ---
APP_VERSION = "1.2.0"
EMBEDDING_DIMENSION = 512 # FaceNet output dimension

# --- Paths ---
# تمامی مسیرها بر اساس BASE_PROJECT_DIR ساخته می شوند.
RESULTS_FOLDER = os.path.join(BASE_PROJECT_DIR, "results")
SORTING_OUTPUT_FOLDER = os.path.join(BASE_PROJECT_DIR, "sorted_faces_output")
DATABASE_FOLDER = os.path.join(BASE_PROJECT_DIR, "database")
LOGS_FOLDER = os.path.join(BASE_PROJECT_DIR, "logs")

# --- Database Settings ---
FACE_DATABASE_FILE = os.path.join(DATABASE_FOLDER, "face_db.db")

# --- Annoy Index Settings ---
ANNOY_INDEX_FILE = os.path.join(DATABASE_FOLDER, "face_embeddings.ann")
ANNOY_NUM_TREES = 10 # Number of trees for Annoy index, trade-off between speed and accuracy

# --- Logging Settings ---
# FIX: LOG_FILE_PATH را مجدداً اضافه می کنیم
LOG_FILE_PATH = os.path.join(LOGS_FOLDER, "app_log.log")
CONSOLE_LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
FILE_LOG_LEVEL = "DEBUG"   # DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Feature Extraction Settings ---
DEFAULT_ENABLE_FACE_LANDMARKS = False
DEFAULT_ENABLE_AGE_GENDER_DETECTION = False
DEFAULT_ENABLE_QUALITY_ENHANCEMENT = False

# --- Comparison Settings ---
DEFAULT_THRESHOLD = 0.6
DEFAULT_BATCH_PROCESSING_ENABLED = True
DEFAULT_BATCH_SIZE = 256
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_MAX_DISPLAY_RESULTS = 10

# --- Sorting Settings ---
DEFAULT_SORTING_THRESHOLD = 0.85
DEFAULT_MIN_FACES_PER_GROUP = 2
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