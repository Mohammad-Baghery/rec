# src/similarity_searcher.py (Re-introduced and Optimized)

import os
import json
import numpy as np
from annoy import AnnoyIndex
import logging
from typing import Optional, List, Tuple

from src.face_db_manager import FaceDBManager
from config import EMBEDDING_DIMENSION, ANNOY_INDEX_FILE, ANNOY_NUM_TREES

logger = logging.getLogger("FaceRecAppLogger")


class SimilaritySearcher:
    """Manages the Annoy index for ultra-fast approximate nearest neighbor search."""

    def __init__(self, db_manager: FaceDBManager):
        self.db_manager = db_manager
        self.embedding_dimension = EMBEDDING_DIMENSION
        self.annoy_index_file = ANNOY_INDEX_FILE
        self.annoy_num_trees = ANNOY_NUM_TREES
        self.annoy_index: Optional[AnnoyIndex] = None
        # This map now stores the database _id for each face as a string
        self.annoy_id_to_face_id: List[str] = []
        self._load_annoy_index()

    def _load_annoy_index(self):
        """Loads a pre-built Annoy index and its metadata map from disk."""
        metadata_map_file = self.annoy_index_file + ".metadata.json"
        if os.path.exists(self.annoy_index_file) and os.path.exists(metadata_map_file):
            try:
                self.annoy_index = AnnoyIndex(self.embedding_dimension, 'angular')
                self.annoy_index.load(self.annoy_index_file)
                with open(metadata_map_file, 'r', encoding='utf-8') as f:
                    self.annoy_id_to_face_id = json.load(f)
                logger.info(f"Loaded Annoy index with {self.get_index_item_count()} items.")
            except Exception as e:
                logger.error(f"Error loading Annoy index: {e}", exc_info=True)
                self.annoy_index = None
        else:
            logger.info("No Annoy index found. It will be built after the 'index' mode is run.")

    def build_and_save_index_from_db(self):
        """Builds an Annoy index from all individual face embeddings in the database."""
        # This method now fetches all faces that have an embedding
        all_faces = self.db_manager.get_all_face_data_for_index()
        if not all_faces:
            logger.warning("No indexed faces in DB to build Annoy index.")
            self.annoy_index = None
            return

        logger.info(f"Building Annoy index from {len(all_faces)} faces...")
        self.annoy_index = AnnoyIndex(self.embedding_dimension, 'angular')
        self.annoy_id_to_face_id = []

        # Sort by face ID string to ensure consistent index building
        all_faces.sort(key=lambda item: str(item[0]))

        for i, (face_id, _, embedding) in enumerate(all_faces):
            self.annoy_index.add_item(i, embedding)
            self.annoy_id_to_face_id.append(str(face_id))

        self.annoy_index.build(self.annoy_num_trees)

        try:
            os.makedirs(os.path.dirname(self.annoy_index_file), exist_ok=True)
            self.annoy_index.save(self.annoy_index_file)
            with open(self.annoy_index_file + ".metadata.json", 'w', encoding='utf-8') as f:
                json.dump(self.annoy_id_to_face_id, f)
            logger.info(f"Annoy index built and saved with {self.get_index_item_count()} items.")
        except Exception as e:
            logger.error(f"Error saving Annoy index: {e}", exc_info=True)
            self.annoy_index = None

    def search_similar(self, query_embedding: np.ndarray, n_results: int) -> List[Tuple[str, float]]:
        """
        Searches for similar faces and returns a list of (face_id_str, similarity) tuples.
        """
        if not self.annoy_index or self.get_index_item_count() == 0:
            logger.warning("Annoy index is not built or is empty.")
            return []

        # Get neighbor IDs and distances from Annoy
        annoy_ids, annoy_distances = self.annoy_index.get_nns_by_vector(query_embedding, n_results,
                                                                        include_distances=True)

        results = []
        for i, annoy_id in enumerate(annoy_ids):
            # Convert Annoy's angular distance to cosine similarity
            similarity = 1 - (annoy_distances[i] ** 2 / 2)

            if annoy_id < len(self.annoy_id_to_face_id):
                face_id_str = self.annoy_id_to_face_id[annoy_id]
                results.append((face_id_str, similarity))
        return results

    def get_index_item_count(self) -> int:
        return self.annoy_index.get_n_items() if self.annoy_index else 0
