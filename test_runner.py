# test_runner.py (Final Control Panel for Data-Centric Architecture)

import os
import time
from app2 import App2
from src.face_db_manager import FaceDBManager
from config import *

# ===================================================================
# --- USER CONTROL PANEL ---
# ===================================================================

# 1. CHOOSE THE MODE TO RUN
#    - "add_faces":  (Simulates UI) Adds file paths from a folder to the DB.
#    - "index":      (Backend Task) Processes all unindexed faces in the DB.
#    - "sort":       (Backend Task) Groups unassigned, indexed faces into new Person documents.
#    - "compare":    (Backend Task) Compares one specific face against a list of others.
#    - Other modes like "enroll_from_ids", "identify_from_id" can be added here.
SELECTED_MODE = "add_faces"

# 2. CONFIGURE PATHS
#    This is mainly for the 'add_faces' simulation.
PATH_TO_ADD_TO_DB = "D:\\python\\Face Compare\\Face\\33-stream"

# 3. CONFIGURE PARAMETERS
SORTING_THRESHOLD = 0.88
MIN_FACES_FOR_SORT_GROUP = 3
COMPARISON_THRESHOLD = 0.7


# ===================================================================
# --- SCRIPT EXECUTION LOGIC (No need to edit below this line) ---
# ===================================================================

def simulate_ui_adding_faces(db_manager: FaceDBManager, folder_path: str):
    """This function simulates a UI adding file paths to the database."""
    print(f"\n--- Simulating UI: Adding file paths from '{folder_path}' to the database ---")
    if not os.path.isdir(folder_path):
        print(f"ERROR: The folder '{folder_path}' does not exist.")
        return

    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    added_count = db_manager.add_face_paths(file_paths)
    print(f"--- UI Simulation Complete: Added {added_count} new file paths to the 'faces' collection. ---")


if __name__ == "__main__":
    # --- Basic Setup ---
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "database"), exist_ok=True)  # For Annoy index
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print("Reminder: Make sure your MongoDB server is running.")

    app_instance = App2(use_gpu=True)
    db_manager_for_setup = FaceDBManager()
    config_to_set = {}
    print(f"\n--- Preparing to run mode: '{SELECTED_MODE}' ---")

    # --- Mode-Specific Logic ---
    if SELECTED_MODE == "add_faces":
        simulate_ui_adding_faces(db_manager_for_setup, PATH_TO_ADD_TO_DB)
        exit()

    elif SELECTED_MODE == "index":
        config_to_set = {"mode": "index"}

    elif SELECTED_MODE == "sort":
        # Get all faces that have an embedding but no person_id assigned yet
        unassigned_faces = list(db_manager_for_setup.db.faces.find(
            {"person_id": None, "embedding": {"$ne": None}},
            {"_id": 1}
        ))
        face_ids_to_sort = [str(face['_id']) for face in unassigned_faces]
        print(f"Found {len(face_ids_to_sort)} unassigned faces to sort.")
        config_to_set = {
            "mode": "sort", "sorting_face_ids": face_ids_to_sort,
            "sorting_threshold": SORTING_THRESHOLD, "min_faces_per_group": MIN_FACES_FOR_SORT_GROUP
        }

    elif SELECTED_MODE == "compare":
        # For this test, let's compare the first face against the next 100
        faces = list(db_manager_for_setup.db.faces.find(
            {"embedding": {"$ne": None}}, {"_id": 1}).limit(101))

        if len(faces) < 2:
            print("ERROR: Need at least 2 indexed faces to run comparison.")
            exit()

        target_id = str(faces[0]['_id'])
        source_ids = [str(face['_id']) for face in faces[1:]]
        print(f"Comparing face ID {target_id} against {len(source_ids)} other faces.")
        config_to_set = {
            "mode": "compare", "target_face_id": target_id,
            "source_face_ids": source_ids, "comparison_threshold": COMPARISON_THRESHOLD
        }

    else:
        print(f"ERROR: Invalid mode '{SELECTED_MODE}' selected.")
        exit()

    # --- Common Execution Logic ---
    if config_to_set:
        app_instance.config(**config_to_set)
        app_instance.run_operation()

        while app_instance.is_active():
            time.sleep(1)

        final_status = app_instance.get_status()
        print("\n--- Final Status from Core App ---")
        print(f"Status: {final_status['status']}, Details: {final_status['details']}")

    print("\nTest script finished.")
