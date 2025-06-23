# test_runner.py (Final Master Runner)

import os
import time
import logging
from app2 import App2
from config import *

# ===================================================================
# --- USER CONTROL PANEL ---
# ===================================================================

# 1. CHOOSE THE MODE TO RUN
# Available options: "enroll", "identify"
# (Other modes like "sort" can be added here later)
SELECTED_MODE = "identify"

# 2. CONFIGURE PATHS AND NAMES
# Define all potential paths you will use.
# --- For 'enroll' mode ---
PERSON_NAME_TO_ENROLL = "Ali Mohammadi"
PATH_ENROLL_INPUT = "D:\\python\\Face Compare\\Face\\enroll_ali"

# --- For 'identify' mode ---
PATH_TO_IDENTIFY = "D:\\python\\Face Compare\\Face\\Cut\\(670).jpg"

# 3. CONFIGURE PARAMETERS
THRESHOLD_IDENTIFY = 0.4

# ===================================================================
# --- SCRIPT EXECUTION LOGIC (No need to edit below this line) ---
# ===================================================================

if __name__ == "__main__":
    # Ensure necessary directories exist
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "database"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "results"), exist_ok=True)

    app_instance = App2(use_gpu=True)

    config_to_set = {}
    print(f"\n--- Preparing to run mode: '{SELECTED_MODE}' ---")

    # --- Configure the application based on the SELECTED_MODE ---
    if SELECTED_MODE == "enroll":
        print(f"Enrolling new person: '{PERSON_NAME_TO_ENROLL}'")
        print(f"Using photos from: {PATH_ENROLL_INPUT}")
        config_to_set = {
            "mode": "enroll",
            "person_name": PERSON_NAME_TO_ENROLL,
            "source_folder_path": PATH_ENROLL_INPUT,
        }

    elif SELECTED_MODE == "identify":
        print(f"Attempting to identify photo: '{PATH_TO_IDENTIFY}'")
        config_to_set = {
            "mode": "identify",
            "target_path": PATH_TO_IDENTIFY,
            "comparison_threshold": THRESHOLD_IDENTIFY
        }

    # You can add other modes here later, for example:
    # elif SELECTED_MODE == "sort":
    #     config_to_set = { ... }

    else:
        print(f"ERROR: Invalid mode '{SELECTED_MODE}' selected. Exiting.")
        exit()

    # --- Common Execution Logic ---
    if not config_to_set:
        print("Configuration dictionary is empty. Nothing to run.")
        exit()

    app_instance.set_config(**config_to_set)

    result = app_instance.run_operation()
    print(f"--- Operation start command sent. Result: {result} ---")

    # Monitor status until completion
    while True:
        status = app_instance.get_status()
        print(f"--- Status: {status['status']} ({status['progress']}%) - {status['details']}")
        if status['status'] in [STATUS_COMPLETE, STATUS_ERROR, STATUS_INCOMPLETE]:
            print(f"\n--- Final Status Received: {status} ---")
            break
        time.sleep(1)

    print("\nTest script finished.")