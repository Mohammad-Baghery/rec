# test_runner.py (Control Panel Version)

import os
import time
import shutil
import logging
from app2 import App2
from config import *

# ===================================================================
# --- USER CONTROL PANEL ---
# ===================================================================

# 1. CHOOSE THE MODE TO RUN
# This is the main switch to decide what the script will do.
# Available options: "index", "compare", "folder_to_folder", "sort"
SELECTED_MODE = "sort"

# 2. DEFINE ALL YOUR PATHS HERE
# Define all potential paths you will use. You don't need to change them
# every time, just set them up once.
PATH_TARGET_IMAGE = "D:\\python\\Face Compare\\Face\\Cut\\IMG_0020c.jpg"
PATH_FOLDER_A = "D:\\python\\Face Compare\\Face\\33-stream"
PATH_FOLDER_B = "D:\\python\\Face Compare\\Face\\Cut"  # Example for folder-to-folder
PATH_UNSORTED_FACES = "D:\\python\\Face Compare\\Face\\33-stream"  # Example for sorting
PATH_NEW_FACES_TO_INDEX = "D:\\python\\Face Compare\\Face\\new_day_faces"  # Example for indexing

# Define output paths
PATH_SORTING_OUTPUT = "D:\\python\\Face Compare\\Face\\sorted_output"
PATH_RESULTS_OUTPUT = "D:\\test-result"

# 3. CONFIGURE PARAMETERS
# Adjust these values as needed for your tasks.
THRESHOLD_COMPARISON = 0.7
THRESHOLD_SORTING = 0.90
MIN_FACES_FOR_SORT_GROUP = 2
MAX_RESULTS_TO_DISPLAY = 20
SHOULD_UPDATE_DB = True  # Set to True to index folders before comparing


# ===================================================================
# --- SCRIPT EXECUTION LOGIC (No need to edit below this line) ---
# ===================================================================

def clean_up_old_files():
    """Deletes old database, log, and results folders."""
    print("Cleaning up old database, log, and results folders...")
    logger_name = "FaceRecAppLogger"
    if logging.getLogger(logger_name).handlers:
        for handler in list(logging.getLogger(logger_name).handlers):
            handler.close()
            logging.getLogger(logger_name).removeHandler(handler)
        print("Closed existing log handlers.")

    folders_to_clean = [
        os.path.join(BASE_PROJECT_DIR, "database"),
        os.path.join(BASE_PROJECT_DIR, "logs"),
        PATH_RESULTS_OUTPUT,
        PATH_SORTING_OUTPUT
    ]
    for folder_path in folders_to_clean:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Removed: {folder_path}")
            except OSError as e:
                print(f"Error removing {folder_path}: {e}")
    print("Clean-up complete.")


if __name__ == "__main__":
    # If you want a fresh start for every run, uncomment the next line.
    # For real use (like daily indexing), you should keep it commented.
    # clean_up_old_files()

    # Ensure necessary directories exist
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "database"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "logs"), exist_ok=True)
    os.makedirs(PATH_RESULTS_OUTPUT, exist_ok=True)
    os.makedirs(PATH_SORTING_OUTPUT, exist_ok=True)

    app_instance = App2(use_gpu=True)

    config_to_set = {}
    print(f"\n--- Preparing to run mode: '{SELECTED_MODE}' ---")

    # --- Configure the application based on the SELECTED_MODE ---
    if SELECTED_MODE == "index":
        print(f"Source Folder for Indexing: {PATH_NEW_FACES_TO_INDEX}")
        config_to_set = {
            "mode": "index",
            "source_folder_path": PATH_NEW_FACES_TO_INDEX
        }

    elif SELECTED_MODE == "compare":
        print(f"Target Image: {PATH_TARGET_IMAGE}")
        print(f"Source Folder: {PATH_FOLDER_A}")
        config_to_set = {
            "mode": "compare",
            "target_path": PATH_TARGET_IMAGE,
            "source_folder_path": PATH_FOLDER_A,
            "comparison_threshold": THRESHOLD_COMPARISON,
            "max_display_results": MAX_RESULTS_TO_DISPLAY,
            "update_db": SHOULD_UPDATE_DB
        }

    elif SELECTED_MODE == "folder_to_folder":
        print(f"Folder A: {PATH_FOLDER_A}")
        print(f"Folder B: {PATH_FOLDER_B}")
        config_to_set = {
            "mode": "folder_to_folder",
            "target_folder_path": PATH_FOLDER_A,
            "source_folder_path": PATH_FOLDER_B,
            "comparison_threshold": THRESHOLD_COMPARISON,
            "update_db": SHOULD_UPDATE_DB
        }

    elif SELECTED_MODE == "sort":
        print(f"Source Folder for Sorting: {PATH_UNSORTED_FACES}")
        print(f"Output Path: {PATH_SORTING_OUTPUT}")
        config_to_set = {
            "mode": "sort",
            "source_folder_path": PATH_UNSORTED_FACES,
            "sorting_output_path": PATH_SORTING_OUTPUT,
            "sorting_threshold": THRESHOLD_SORTING,
            "min_faces_per_group": MIN_FACES_FOR_SORT_GROUP
        }

    else:
        print(f"ERROR: Invalid mode '{SELECTED_MODE}' selected. Exiting.")
        exit()

    # --- Common Execution Logic ---
    if config_to_set:
        app_instance.set_config(**config_to_set)

        result = app_instance.run_operation()
        print(f"--- Operation start command sent. Result: {result} ---")

        i = 0
        timeout_seconds = 1800  # 30 minutes, adjust if needed
        while True:
            status = app_instance.get_status()
            print(f"--- Status: {status['status']} ({status['progress']}%) - {status['details']}")

            if status['status'] in [STATUS_COMPLETE, STATUS_ERROR, STATUS_INCOMPLETE]:
                print(f"\n--- Final Status Received: {status} ---")
                break

            i += 1
            if i > timeout_seconds:
                print(f"\n--- Timeout after {timeout_seconds} seconds. Sending stop request. ---")
                app_instance.stop_processing()
                break

            time.sleep(1)

    print("\nTest script finished.")