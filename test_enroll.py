# test_runner.py (Example for ENROLL mode)

import os
import time
import logging
from app2 import App2  # Assuming app2 is updated to call the new run() method
from config import *

# ===================================================================
# --- USER CONTROL PANEL ---
# ===================================================================
# 1. CHOOSE THE MODE
SELECTED_MODE = "enroll"

# 2. PROVIDE THE NAME FOR THE NEW PERSON
PERSON_NAME_TO_ENROLL = "Mahdi Mirza Khani"

# 3. DEFINE PATHS
# Path to the folder containing ONLY photos of the person being enrolled
PATH_ENROLL_INPUT = "D:\\python\\Face Compare\\Face\\sorted_output\\enroll_mahdi"
# Other paths are not used in this mode
# ...

# ===================================================================
# --- SCRIPT EXECUTION LOGIC ---
# ===================================================================
if __name__ == "__main__":
    # In a real scenario, you DON'T clean up when enrolling new people.
    # clean_up_old_files()

    os.makedirs(os.path.join(BASE_PROJECT_DIR, "database"), exist_ok=True)
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "logs"), exist_ok=True)

    app_instance = App2(use_gpu=True)

    config_to_set = {}
    print(f"\n--- Preparing to run mode: '{SELECTED_MODE}' ---")

    if SELECTED_MODE == "enroll":
        print(f"Enrolling new person: '{PERSON_NAME_TO_ENROLL}'")
        print(f"Using photos from: {PATH_ENROLL_INPUT}")
        config_to_set = {
            "mode": "enroll",
            "person_name": PERSON_NAME_TO_ENROLL,
            "source_folder_path": PATH_ENROLL_INPUT,
            "batch_size": 64  # You can use a smaller batch size for enrollment
        }
    else:
        print(f"This runner is configured for 'enroll' mode. Please change SELECTED_MODE if needed.")
        exit()

    # --- Common Execution Logic ---
    if config_to_set:
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