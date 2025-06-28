# test_runner.py (Final Master Control Panel for All Modes)

import os
import time
import logging
from app2 import App2
from src.face_db_manager import FaceDBManager
from config import *

# ===================================================================
# --- USER CONTROL PANEL ---
# ===================================================================

# 1. CHOOSE THE MODE TO RUN
#    - "add_faces":          Adds file paths from a folder to the DB.
#    - "index":              Processes all unindexed faces in the DB.
#    - "enroll":             A shortcut for 'add_faces' + 'index' for one person.
#    - "identify":           Identifies a face in a single photo against known Persons.
#    - "sort":               Groups unassigned faces into new 'Person' documents.
#    - "compare_one_vs_many":Compares one face ID against all other indexed faces.
#    - "batch_compare":      Compares all faces from Folder A against all from Folder B.
#    - "find_duplicates":    Compares all enrolled Persons to find likely duplicates.
#    - "merge_persons":      Merges two specified Person documents into one.
SELECTED_MODE = "enroll"

# 2. CONFIGURE PATHS & NAMES
# --- For 'add_faces', 'sort' ---
PATH_SOURCE_FOLDER = "D:\\python\\Face Compare\\Face\\33-stream"

# --- For 'enroll' mode ---
PERSON_NAME_TO_ENROLL = "Ali Mohammadi"
PATH_ENROLL_INPUT = "D:\\python\\Face Compare\\Face\\enroll_ali"

# --- For 'identify' mode ---
PATH_TO_IDENTIFY = "D:\\python\\Face Compare\\Face\\enroll_ali\\(1).jpg"

# --- For 'compare_one_vs_many' mode ---
# The script will automatically pick the first indexed face as the target.

# --- For 'batch_compare' mode ---
PATH_FOLDER_A = "D:\\python\\Face Compare\\Face\\Folder_A_Day1"
PATH_FOLDER_B = "D:\\python\\Face Compare\\Face\\Folder_B_Day2"

# --- For 'merge_persons' mode ---
PERSON_TO_MERGE_SOURCE = "Duplicate Of Ali"
PERSON_TO_MERGE_DESTINATION = "Ali Mohammadi"

# 3. CONFIGURE PARAMETERS
COMPARISON_THRESHOLD = 0.7
SORTING_THRESHOLD = 0.88
MIN_FACES_FOR_SORT_GROUP = 3
DUPLICATE_THRESHOLD = 0.90


# ===================================================================
# --- SCRIPT EXECUTION LOGIC (No need to edit below this line) ---
# ===================================================================

def simulate_ui_adding_faces(db_manager: FaceDBManager, folder_path: str):
    print(f"\n--- Simulating UI: Adding paths from '{folder_path}' ---")
    if not os.path.isdir(folder_path):
        print(f"ERROR: Folder '{folder_path}' not found.")
        return
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    added_count = db_manager.add_face_paths(file_paths)
    print(f"--- UI Sim Complete: Added {added_count} new paths. ---")


def get_face_ids_from_path(db_manager: FaceDBManager, folder_path: str) -> list:
    if not os.path.isdir(folder_path): return []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                  f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    cursor = db_manager.db.faces.find({"file_path": {"$in": file_paths}}, {"_id": 1})
    return [doc['_id'] for doc in cursor]


if __name__ == "__main__":
    # Setup
    os.makedirs(os.path.join(BASE_PROJECT_DIR, "database"), exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    print("Reminder: Make sure your MongoDB server is running.")

    app_instance = App2(use_gpu=True)
    db_manager_for_setup = FaceDBManager()
    config_to_set = {}
    print(f"\n--- Preparing to run mode: '{SELECTED_MODE}' ---")

    # Mode-Specific Logic
    if SELECTED_MODE == "add_faces":
        simulate_ui_adding_faces(db_manager_for_setup, PATH_SOURCE_FOLDER)
        exit()

    elif SELECTED_MODE == "index":
        config_to_set = {"mode": "index"}

    elif SELECTED_MODE == "enroll":
        config_to_set = {"mode": "enroll", "person_name": PERSON_NAME_TO_ENROLL,
                         "source_folder_path": PATH_ENROLL_INPUT}

    elif SELECTED_MODE == "identify":
        config_to_set = {"mode": "identify", "target_path": PATH_TO_IDENTIFY,
                         "comparison_threshold": COMPARISON_THRESHOLD}

    elif SELECTED_MODE == "sort":
        face_ids_to_sort = [f['_id'] for f in
                            db_manager_for_setup.db.faces.find({"person_id": None, "embedding": {"$ne": None}},
                                                               {"_id": 1})]
        print(f"Found {len(face_ids_to_sort)} unassigned faces to sort.")
        config_to_set = {"mode": "sort", "sorting_face_ids": face_ids_to_sort, "sorting_threshold": SORTING_THRESHOLD,
                         "min_faces_per_group": MIN_FACES_FOR_SORT_GROUP}

    elif SELECTED_MODE == "compare_one_vs_many":
        all_faces = list(db_manager_for_setup.db.faces.find({"embedding": {"$ne": None}}, {"_id": 1}))
        if len(all_faces) < 2:
            print("ERROR: Need at least 2 indexed faces.")
            exit()
        target_id, source_ids = all_faces[0]['_id'], [f['_id'] for f in all_faces[1:]]
        print(f"Comparing face ID {target_id} against {len(source_ids)} other faces.")
        config_to_set = {"mode": "compare", "target_face_id": target_id, "source_face_ids": source_ids,
                         "comparison_threshold": COMPARISON_THRESHOLD}

    elif SELECTED_MODE == "batch_compare":
        face_ids_a, face_ids_b = get_face_ids_from_path(db_manager_for_setup, PATH_FOLDER_A), get_face_ids_from_path(
            db_manager_for_setup, PATH_FOLDER_B)
        if not face_ids_a or not face_ids_b:
            print("ERROR: One or both folders have no indexed faces. Run 'add_faces' and 'index' first.")
            exit()
        config_to_set = {"mode": "batch_compare", "source_face_ids_a": face_ids_a, "source_face_ids_b": face_ids_b,
                         "comparison_threshold": COMPARISON_THRESHOLD}

    elif SELECTED_MODE == "find_duplicates":
        config_to_set = {"mode": "find_duplicates", "comparison_threshold": DUPLICATE_THRESHOLD}

    elif SELECTED_MODE == "merge_persons":
        config_to_set = {"mode": "merge_persons", "source_person_name": PERSON_TO_MERGE_SOURCE,
                         "destination_person_name": PERSON_TO_MERGE_DESTINATION}

    else:
        print(f"ERROR: Invalid mode '{SELECTED_MODE}' selected.")
        exit()

    # Common Execution Logic
    if config_to_set:
        app_instance.config(**config_to_set)
        app_instance.run_operation()
        while app_instance.is_active():
            time.sleep(1)
        final_status = app_instance.get_status()
        print("\n--- Final Status from Core App ---")
        print(f"Status: {final_status['status']}, Details: {final_status['details']}")

    print("\nTest script finished.")
