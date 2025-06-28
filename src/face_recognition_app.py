# src/face_recognition_app.py (Optimized Batch Compare using Annoy)

import os
import time
import threading
import numpy as np
import torch
import logging
from typing import Dict, Any, List
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from bson import ObjectId

from src.base_app import _BaseApp
from src.signals import status_updated, operation_complete, operation_error
from src.feature_extractor import FeatureExtractor
from src.similarity_searcher import SimilaritySearcher
from src.face_db_manager import FaceDBManager
from config import *

logger = logging.getLogger("FaceRecAppLogger")


class FaceRecognitionApp(_BaseApp):
    def __init__(self, use_gpu: bool = True):
        super().__init__()

        self.app_config: Dict[str, Any] = {
            "mode": "index", "person_name": "", "source_folder_path": "", "target_path": "",
            "target_face_id": None, "source_face_ids": [], "source_face_ids_a": [],
            "source_face_ids_b": [], "sorting_face_ids": [], "source_person_name": "",
            "destination_person_name": "", "comparison_threshold": DEFAULT_THRESHOLD,
            "sorting_threshold": DEFAULT_SORTING_THRESHOLD, "min_faces_per_group": DEFAULT_MIN_FACES_PER_GROUP,
            "batch_size": DEFAULT_BATCH_SIZE
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

        try:
            self.db_manager = FaceDBManager()
            self.feature_extractor = FeatureExtractor(device=self.device)
            self.similarity_searcher = SimilaritySearcher(db_manager=self.db_manager)
            logger.info(f"FaceRecognitionApp core initialized on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize a core component: {e}", exc_info=True)
            raise

    def config(self, **kwargs) -> bool:
        for key, value in kwargs.items():
            if "_id" in key and value is not None and isinstance(value, str):
                self.app_config[key] = ObjectId(value)
            elif "_ids" in key and isinstance(value, list):
                self.app_config[key] = [ObjectId(item) for item in value if isinstance(item, str)]
            elif key in self.app_config:
                self.app_config[key] = value
        logger.info(f"Configuration applied: {kwargs}")
        return True

    def run(self):
        if self.processing_active: return
        mode = self.app_config.get("mode")
        self._update_status(STATUS_WAITING, f"Initializing '{mode}' mode...", 0)

        target_map = {
            "index": self._execute_indexing,
            "enroll": self._execute_enrollment,
            "identify": self._execute_identify,
            "sort": self._execute_sorting,
            "compare": self._execute_comparison,
            "batch_compare": self._execute_batch_comparison,
            "find_duplicates": self._execute_find_duplicates,
            "merge_persons": self._execute_merge_persons,
        }
        thread_target = target_map.get(mode)

        if not thread_target:
            self._update_status_and_emit_error(f"Unknown mode: '{mode}'")
            return

        self.processing_active = True
        self.cancel_requested = False
        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()

    def _update_status(self, status: str, details: str = "", progress: int = None):
        super()._update_status(status, details, progress)
        status_updated.send(self, progress=self.status_progress, details=self.status_details,
                            status=self.process_status)

    def _update_status_and_emit_error(self, error_message: str):
        logger.error(error_message)
        self._update_status(STATUS_ERROR, error_message, self.status_progress)
        operation_error.send(self, message=error_message)

    # --- Execution Methods ---

    def _execute_indexing(self):
        try:
            self._update_status(STATUS_PROCESSING, "Fetching unindexed faces...", 10)
            unindexed_faces = self.db_manager.get_unindexed_faces()
            if not unindexed_faces:
                # Even if no new faces, maybe the index needs a rebuild
                self._update_status(STATUS_PROCESSING, "No new faces. Verifying search index...", 90)
                self.similarity_searcher.build_and_save_index_from_db()
                operation_complete.send(self, message="Database is already up-to-date.")
                return

            paths_to_process = [face['file_path'] for face in unindexed_faces]
            face_id_map = {face['file_path']: face['_id'] for face in unindexed_faces}
            features_list = self.feature_extractor.process_images(paths_to_process, self.app_config["batch_size"])

            self._update_status(STATUS_PROCESSING, f"Saving {len(features_list)} new embeddings to DB...", 70)
            updates = [(face_id_map.get(f['file_path']), f['embedding']) for f in features_list if
                       face_id_map.get(f['file_path'])]
            self.db_manager.set_face_embeddings(updates)

            # --- FIX: Rebuild the search index after adding new embeddings ---
            self._update_status(STATUS_PROCESSING, "Rebuilding search index with new data...", 90)
            self.similarity_searcher.build_and_save_index_from_db()

            operation_complete.send(self, message=f"Indexing complete. Added {len(updates)} embeddings.")
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_enrollment(self):
        try:
            person_name = self.app_config.get("person_name")
            folder_path = self.app_config.get("source_folder_path")
            if not person_name or not folder_path or not os.path.isdir(folder_path): raise ValueError(
                "`person_name` and `source_folder_path` are required.")

            person = self.db_manager.get_person_by_name(person_name)
            person_id = person['_id'] if person else self.db_manager.add_person(name=person_name)
            if not person_id: raise RuntimeError(f"Could not create/find person '{person_name}'.")

            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not image_paths: raise ValueError("Source folder is empty.")

            features_list = self.feature_extractor.process_images(image_paths, self.app_config["batch_size"])
            if not features_list: raise RuntimeError("Feature extraction failed.")

            self.db_manager.add_faces_to_person(person_id, features_list)
            self.db_manager.update_person_embedding(person_id)
            operation_complete.send(self, message=f"Successfully enrolled/updated '{person_name}'.")
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_identify(self):
        try:
            target_path = self.app_config.get("target_path")
            if not target_path or not os.path.isfile(target_path): raise ValueError(
                "A valid 'target_path' is required.")

            target_features = self.feature_extractor.process_images([target_path])
            if not target_features: raise ValueError("Could not extract features from target.")
            target_embedding = target_features[0]['embedding']

            known_persons = self.db_manager.get_all_person_embeddings()
            if not known_persons: raise ValueError("No persons enrolled in database.")

            best_match_name, best_match_similarity = "Unknown", -1.0
            for _, name, mean_embedding in known_persons:
                similarity = cosine_similarity(target_embedding.reshape(1, -1), mean_embedding.reshape(1, -1))[0][0]
                if similarity > best_match_similarity: best_match_similarity, best_match_name = similarity, name

            threshold = self.app_config["comparison_threshold"]
            result = f"Person identified as '{best_match_name}' with {best_match_similarity:.2%} similarity." if best_match_similarity > threshold else f"Best match '{best_match_name}' ({best_match_similarity:.2%}) is below threshold."
            operation_complete.send(self, message=result,
                                    results={"name": best_match_name, "similarity": best_match_similarity})
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_sorting(self):
        try:
            face_ids = self.app_config.get("sorting_face_ids", [])
            if not face_ids: raise ValueError("No face IDs provided for sorting.")

            embeddings_map = self.db_manager.get_embeddings_by_ids(face_ids)
            if not embeddings_map: raise ValueError("Could not retrieve embeddings for the given IDs.")

            face_id_list, embedding_matrix = list(embeddings_map.keys()), np.array(list(embeddings_map.values()))

            similarity_matrix = cosine_similarity(embedding_matrix)
            np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
            distance_matrix = 1.0 - similarity_matrix
            clusters = DBSCAN(eps=(1 - self.app_config["sorting_threshold"]),
                              min_samples=self.app_config["min_faces_per_group"], metric='precomputed').fit_predict(
                distance_matrix)

            cluster_map = {}
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_map: cluster_map[cluster_id] = []
                cluster_map[cluster_id].append(face_id_list[i])

            new_persons_count = 0
            for cluster_id, clustered_face_ids in cluster_map.items():
                if cluster_id == -1: continue  # Handle noise/unclassified later if needed
                person_name = f"Person_{int(time.time())}_{cluster_id}"
                person_id = self.db_manager.add_person(name=person_name)
                if person_id:
                    self.db_manager.assign_faces_to_person(clustered_face_ids, person_id)
                    self.db_manager.update_person_embedding(person_id)
                    new_persons_count += 1

            operation_complete.send(self, message=f"Sorting complete. Created {new_persons_count} new person groups.")
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_comparison(self):  # This is the 'compare_one_vs_many' mode
        try:
            target_id = self.app_config.get("target_face_id")
            source_ids = self.app_config.get("source_face_ids", [])
            if not target_id: raise ValueError("Target Face ID is required.")

            self._update_status(STATUS_PROCESSING, "Fetching target embedding...", 10)
            target_embedding_map = self.db_manager.get_embeddings_by_ids([target_id])
            if not target_embedding_map: raise ValueError("Target face not found or not indexed.")
            target_embedding = target_embedding_map[target_id]

            self._update_status(STATUS_PROCESSING, "Searching for similar faces using index...", 50)

            # Use the fast Annoy search to find top candidates from the entire database
            # We search for a bit more than we need to ensure good coverage.
            num_to_search = self.similarity_searcher.get_index_item_count()
            all_matches = self.similarity_searcher.search_similar(target_embedding, n_results=num_to_search)

            self._update_status(STATUS_PROCESSING, "Filtering and ranking results...", 90)

            # Filter the results to include only those from the user-provided source_ids list
            source_ids_set = {str(id) for id in source_ids}  # For fast lookup

            final_matches = []
            for face_id_str, similarity in all_matches:
                if face_id_str in source_ids_set and similarity > self.app_config["comparison_threshold"]:
                    final_matches.append({"face_id": face_id_str, "similarity": float(similarity)})

            final_matches.sort(key=lambda x: x["similarity"], reverse=True)

            operation_complete.send(self, message=f"Comparison complete. Found {len(final_matches)} matches.",
                                    results=final_matches)
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_batch_comparison(self):
        """Compares each face in group A against all faces in group B using the Annoy index."""
        try:
            ids_a = self.app_config.get("source_face_ids_a", [])
            ids_b = self.app_config.get("source_face_ids_b", [])
            if not ids_a or not ids_b:
                raise ValueError("Two lists of face IDs (A and B) are required.")

            self._update_status(STATUS_PROCESSING, "Fetching embeddings for Group A...", 10)
            embeddings_a_map = self.db_manager.get_embeddings_by_ids(ids_a)
            if not embeddings_a_map:
                raise ValueError("Could not retrieve any embeddings for Group A.")

            self._update_status(STATUS_PROCESSING, f"Comparing {len(embeddings_a_map)} faces from Group A...", 30)

            all_results = {}
            total_faces_a = len(embeddings_a_map)
            threshold = self.app_config["comparison_threshold"]
            ids_b_set = {str(id) for id in ids_b}  # Use a set for fast lookups

            for i, (target_id, target_embedding) in enumerate(embeddings_a_map.items()):
                if self.cancel_requested: raise InterruptedError("Operation cancelled.")

                progress = 30 + int((i / total_faces_a) * 65) if total_faces_a > 0 else 95
                self._update_status(STATUS_PROCESSING, f"Processing face {i + 1}/{total_faces_a} from Group A",
                                    progress)

                # Use the fast Annoy search to get top candidates from the ENTIRE database
                num_to_search = self.similarity_searcher.get_index_item_count()
                search_candidates = self.similarity_searcher.search_similar(target_embedding, n_results=num_to_search)

                # Filter the candidates to keep only those that are in Group B and above the threshold
                matches = [
                    {"face_id": face_id_str, "similarity": float(similarity)}
                    for face_id_str, similarity in search_candidates
                    if face_id_str in ids_b_set and similarity > threshold
                ]

                if matches:
                    matches.sort(key=lambda x: x["similarity"], reverse=True)
                    all_results[str(target_id)] = matches

            result_message = f"Batch comparison complete. Found matches for {len(all_results)} faces."
            self._update_status(STATUS_COMPLETE, result_message, 100)
            operation_complete.send(self, message=result_message, results=all_results)

        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_find_duplicates(self):
        try:
            known_persons = self.db_manager.get_all_person_embeddings()
            if len(known_persons) < 2: raise ValueError("Need at least two enrolled persons to find duplicates.")

            person_data = [{"id": p[0], "name": p[1]} for p in known_persons]
            embeddings = np.array([p[2] for p in known_persons])
            similarity_matrix = cosine_similarity(embeddings)

            potential_duplicates = []
            threshold = self.app_config["comparison_threshold"]
            for i in range(len(person_data)):
                for j in range(i + 1, len(person_data)):
                    if similarity_matrix[i, j] > threshold:
                        potential_duplicates.append(
                            {"person_1_name": person_data[i]["name"], "person_2_name": person_data[j]["name"],
                             "similarity": float(similarity_matrix[i, j])})

            potential_duplicates.sort(key=lambda x: x["similarity"], reverse=True)
            operation_complete.send(self, message=f"Found {len(potential_duplicates)} potential duplicate pairs.",
                                    results=potential_duplicates)
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False

    def _execute_merge_persons(self):
        try:
            source_name, dest_name = self.app_config.get("source_person_name"), self.app_config.get(
                "destination_person_name")
            if not source_name or not dest_name: raise ValueError("Source and destination person names are required.")

            success = self.db_manager.merge_persons(source_name, dest_name)
            if not success: raise RuntimeError("Database operation to merge persons failed.")

            operation_complete.send(self, message=f"Successfully merged '{source_name}' into '{dest_name}'.")
        except Exception as e:
            self._update_status_and_emit_error(str(e))
        finally:
            self.processing_active = False
