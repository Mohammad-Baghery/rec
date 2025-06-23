# src/similarity_searcher.py (Updated to store person IDs)

import os
import json
import numpy as np
from annoy import AnnoyIndex
import logging
from typing import Optional, List, Callable, Tuple

from src.face_db_manager import FaceDBManager
from config import EMBEDDING_DIMENSION, ANNOY_INDEX_FILE, ANNOY_NUM_TREES

logger = logging.getLogger("FaceRecAppLogger")


class SimilaritySearcher:
    def __init__(self, db_manager: FaceDBManager, **kwargs):
        self.db_manager = db_manager
        self.embedding_dimension = EMBEDDING_DIMENSION
        self.annoy_index_file = ANNOY_INDEX_FILE
        self.annoy_num_trees = ANNOY_NUM_TREES
        self.annoy_index: Optional[AnnoyIndex] = None
        # This map now stores a tuple: (file_path, person_id)
        self.annoy_id_to_metadata: List[Tuple[str, int]] = []
        self._load_annoy_index()

    def _load_annoy_index(self):
        # The map file will now store metadata
        metadata_map_file = self.annoy_index_file + ".metadata.json"
        if os.path.exists(self.annoy_index_file) and os.path.exists(metadata_map_file):
            try:
                self.annoy_index = AnnoyIndex(self.embedding_dimension, 'angular')
                self.annoy_index.load(self.annoy_index_file)
                with open(metadata_map_file, 'r', encoding='utf-8') as f:
                    self.annoy_id_to_metadata = [tuple(item) for item in json.load(f)]
                logger.info(f"Loaded Annoy index with {self.get_index_item_count()} items.")
            except Exception as e:
                logger.error(f"Error loading Annoy index: {e}", exc_info=True)
                self.annoy_index = None
        else:
            logger.info("No Annoy index found.")
            self.annoy_index = None

    def build_and_save_index_from_db(self, progress_callback=None, progress_range=(0, 100)):
        """
        Builds Annoy index from all individual face embeddings in the database.
        It now also stores the person_id for each face.
        """
        if progress_callback: progress_callback("PROCESSING", "Fetching face embeddings from DB...", progress_range[0])

        # This method in DBManager needs to be updated to return person_id as well
        all_faces = self.db_manager.get_all_face_data_for_index()
        if not all_faces:
            logger.warning("No embeddings in DB to build Annoy index.")
            self.annoy_index = None
            return

        logger.info(f"Building Annoy index from {len(all_faces)} faces...")
        self.annoy_index = AnnoyIndex(self.embedding_dimension, 'angular')
        self.annoy_id_to_metadata = []

        all_faces.sort(key=lambda item: item[0])  # Sort by file_path for consistency

        for i, (path, person_id, embedding) in enumerate(all_faces):
            self.annoy_index.add_item(i, embedding)
            self.annoy_id_to_metadata.append((path, person_id))

        if progress_callback: progress_callback("PROCESSING", f"Building {self.annoy_num_trees} trees...",
                                                progress_range[0] + int((progress_range[1] - progress_range[0]) * 0.5))
        self.annoy_index.build(self.annoy_num_trees)

        try:
            self.annoy_index.save(self.annoy_index_file)
            # Save the new metadata map
            with open(self.annoy_index_file + ".metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.annoy_id_to_metadata, f)
            logger.info(f"Annoy index built and saved with {self.get_index_item_count()} items.")
            if progress_callback: progress_callback("PROCESSING", "Annoy index saved.", progress_range[1])
        except Exception as e:
            logger.error(f"Error saving Annoy index: {e}", exc_info=True)
            self.annoy_index = None

    def search_similar_persons(self, query_embedding: np.ndarray) -> List[Tuple[int, float]]:
        """
        Searches for similar faces and returns the most likely person IDs and their average similarity.
        """
        if not self.annoy_index or self.get_index_item_count() == 0: return []

        # Search for more results than needed to get a good sample
        num_to_search = min(100, self.get_index_item_count())
        annoy_ids, annoy_distances = self.annoy_index.get_nns_by_vector(query_embedding, num_to_search,
                                                                        include_distances=True)

        person_similarities = {}
        for i, annoy_id in enumerate(annoy_ids):
            sim = 1 - (max(0, annoy_distances[i] ** 2) / 2)
            if sim < 0.5: continue  # Ignore very dissimilar faces

            _, person_id = self.annoy_id_to_metadata[annoy_id]
            if person_id not in person_similarities:
                person_similarities[person_id] = []
            person_similarities[person_id].append(sim)

        # Average the similarities for each person found
        avg_similarities = []
        for person_id, sims in person_similarities.items():
            # Consider top 5 similarities for a more robust average
            top_sims = sorted(sims, reverse=True)[:5]
            avg_sim = np.mean(top_sims)
            avg_similarities.append((person_id, avg_sim))

        avg_similarities.sort(key=lambda x: x[1], reverse=True)
        return avg_similarities

    def get_index_item_count(self) -> int:
        return self.annoy_index.get_n_items() if self.annoy_index else 0