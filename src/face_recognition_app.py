# src/face_recognition_app.py (With Enroll Mode)

import os
import time
import threading
import numpy as np
import torch
import logging
from typing import Optional, Dict, Any, List, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import shutil

from src.base_app import _BaseApp
from src.feature_extractor import FeatureExtractor
from src.similarity_searcher import SimilaritySearcher
from src.result_manager import ResultManager
from src.face_db_manager import FaceDBManager
from src.utils import calculate_file_hash
from config import *

logger = logging.getLogger("FaceRecAppLogger")


class FaceRecognitionApp(_BaseApp):
    """
    Main application class with a mode-based architecture.
    Modes: enroll, index, compare, sort, folder_to_folder.
    """

    def __init__(self, use_gpu: bool = True):
        super().__init__()

        self.app_config: Dict[str, Any] = {
            "mode": "enroll",  # Default mode
            "person_name": "",  # For enroll mode
            "source_folder_path": "",  # Used for enroll, index, sort, etc.
            # ... other config keys from previous versions ...
            "update_db": False, "target_path": "", "target_folder_path": "",
            "sorting_output_path": SORTING_OUTPUT_FOLDER, "comparison_threshold": 0.7,
            "sorting_threshold": DEFAULT_SORTING_THRESHOLD, "min_faces_per_group": DEFAULT_MIN_FACES_PER_GROUP,
            "max_display_results": DEFAULT_MAX_DISPLAY_RESULTS, "batch_size": DEFAULT_BATCH_SIZE
        }

        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.db_manager = FaceDBManager(db_path=FACE_DATABASE_FILE)
        # Note the change here: db_manager is no longer passed to FeatureExtractor
        self.feature_extractor = FeatureExtractor(device=self.device)
        self.similarity_searcher = SimilaritySearcher(db_manager=self.db_manager)
        self.result_manager = ResultManager(result_folder_path=RESULTS_FOLDER)
        self.logger.info(f"FaceRecognitionApp initialized on device: {self.device}")

    def config(self, **kwargs) -> bool:
        # This method remains the same
        for key, value in kwargs.items():
            if key in self.app_config: self.app_config[key] = value
        os.makedirs(self.app_config["sorting_output_path"], exist_ok=True)
        os.makedirs(self.result_manager.result_folder_path, exist_ok=True)
        return True

    def run(self) -> bool:
        if self.processing_active:
            self.logger.warning("A process is already active.")
            return False

        mode = self.app_config.get("mode")
        self._update_status(STATUS_WAITING, f"Initializing '{mode}' mode...", 0)
        self.processing_active = True

        thread_target = None
        if mode == 'enroll':
            thread_target = self._execute_enrollment

        elif mode == 'identify':  # Add new mode
            thread_target = self._execute_identify

        # --- Other modes will be added back later ---
        # elif mode == 'index':
        #     thread_target = self._execute_indexing
        # ...

        if not thread_target:
            self._update_status(STATUS_ERROR, f"Mode '{mode}' is not yet implemented or is invalid.", 0)
            self.processing_active = False
            return False

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()
        return True

    # --- New Execution Method for Enrollment ---
    def _execute_enrollment(self):
        try:
            person_name = self.app_config.get("person_name")
            folder_path = self.app_config.get("source_folder_path")

            if not person_name or not folder_path or not os.path.isdir(folder_path):
                raise ValueError("For enroll mode, 'person_name' and a valid 'source_folder_path' are required.")

            self._update_status("PROCESSING", f"Checking if person '{person_name}' exists...", 10)
            if self.db_manager.get_person_by_name(person_name):
                raise ValueError(f"Person '{person_name}' already exists in the database.")

            self._update_status("PROCESSING", f"Adding '{person_name}' to database...", 20)
            person_id = self.db_manager.add_person(name=person_name)
            if person_id is None:
                raise RuntimeError(f"Failed to create person '{person_name}' in the database.")

            self._update_status("PROCESSING", "Gathering images for enrollment...", 30)
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_paths:
                raise ValueError("Source folder for enrollment contains no images.")

            self._update_status("PROCESSING", f"Extracting features from {len(image_paths)} images...", 40)
            features_list = self.feature_extractor.process_images(image_paths, batch_size=self.app_config["batch_size"])
            if not features_list:
                raise RuntimeError("Feature extraction failed, no features were returned.")

            self._update_status("PROCESSING", f"Adding {len(features_list)} faces to '{person_name}'...", 70)
            self.db_manager.add_faces_to_person(person_id, features_list)

            self._update_status("PROCESSING", "Calculating and saving representative embedding...", 85)
            self.db_manager.update_person_embedding(person_id)

            self._update_status("PROCESSING", "Rebuilding search index...", 95)
            self.similarity_searcher.build_and_save_index_from_db()

            self._update_status(STATUS_COMPLETE,
                                f"Successfully enrolled '{person_name}' with {len(features_list)} photos.", 100)

        except Exception as e:
            self.logger.exception(f"Error during enrollment: {e}")
            self._update_status(STATUS_ERROR, str(e), self.status_progress)
        finally:
            self.processing_active = False

    def _execute_identify(self):
        try:
            target_path = self.app_config.get("target_path")
            if not target_path or not os.path.isfile(target_path):
                raise ValueError("For identify mode, a valid 'target_path' is required.")

            self._update_status("PROCESSING", "Extracting features from target image...", 10)
            target_features = self.feature_extractor.process_images([target_path])
            if not target_features:
                raise ValueError("Could not extract features from target image.")

            target_embedding = target_features[0]['embedding']

            self._update_status("PROCESSING", "Identifying person from database...", 50)

            # This method now returns a ranked list of (person_id, avg_similarity)
            person_results = self.similarity_searcher.search_similar_persons(target_embedding)

            if not person_results:
                self._update_status(STATUS_COMPLETE,
                                    "Identification complete. No similar person found in the database.", 100)
                return

            # Get the top result
            top_person_id, top_similarity = person_results[0]

            # Get person's name from their ID
            person_info = self.db_manager.get_person_by_id(
                top_person_id)  # You'll need to add this small method to DBManager
            person_name = person_info['name'] if person_info else "Unknown"

            threshold = self.app_config.get("comparison_threshold", 0.7)
            if top_similarity > threshold:
                result_details = f"Person identified as '{person_name}' with {top_similarity:.2%} similarity."
            else:
                result_details = f"Best match is '{person_name}' ({top_similarity:.2%}), but it is below the threshold."

            self.logger.info(result_details)
            self._update_status(STATUS_COMPLETE, result_details, 100)

        except Exception as e:
            self.logger.exception(f"Error during identification: {e}")
            self._update_status(STATUS_ERROR, str(e), self.status_progress)
        finally:
            self.processing_active = False