# src/face_db_manager.py (New Data-Centric Logic)

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from pymongo import MongoClient, UpdateOne
from pymongo.errors import ConnectionFailure, BulkWriteError
from bson import ObjectId, Binary
import time

from config import MONGO_URI, MONGO_DATABASE_NAME
from src.utils import calculate_file_hash

logger = logging.getLogger("FaceRecAppLogger")


class FaceDBManager:
    """Manages all database interactions with MongoDB in a data-centric workflow."""

    def __init__(self):
        try:
            self.client = MongoClient(MONGO_URI)
            self.client.admin.command('ismaster')
            self.db = self.client[MONGO_DATABASE_NAME]
            self._create_indexes()
            logger.info(f"Successfully connected to MongoDB: {MONGO_DATABASE_NAME}")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Ensures necessary indexes exist."""
        self.db.persons.create_index("name", unique=True)
        self.db.faces.create_index("file_hash", unique=True)
        self.db.faces.create_index("person_id")
        self.db.faces.create_index("embedding")  # To find unindexed faces
        logger.info("MongoDB indexes ensured.")

    def add_face_paths(self, file_paths: List[str]) -> int:
        """
        (UI Task) Adds new face documents from file paths.
        Uses upsert to avoid creating duplicates based on file_hash.
        """
        if not file_paths: return 0

        operations = []
        for path in file_paths:
            if not os.path.isfile(path): continue
            file_hash = calculate_file_hash(path)
            if file_hash:
                new_face_doc = {
                    "file_path": path, "file_hash": file_hash,
                    "person_id": None, "embedding": None, "created_at": time.time()
                }
                op = UpdateOne({"file_hash": file_hash}, {"$setOnInsert": new_face_doc}, upsert=True)
                operations.append(op)

        if not operations: return 0

        try:
            result = self.db.faces.bulk_write(operations, ordered=False)
            logger.info(f"Added {result.upserted_count} new unique face paths to the database.")
            return result.upserted_count
        except BulkWriteError as bwe:
            logger.warning(
                f"Batch insert completed with some duplicate errors, which is expected. Added: {bwe.details.get('nUpserted', 0)}")
            return bwe.details.get('nUpserted', 0)
        except Exception as e:
            logger.error(f"An error occurred during bulk face insert: {e}", exc_info=True)
            return 0

    def get_unindexed_faces(self) -> List[Dict[str, Any]]:
        """(Backend Task) Returns faces that have not been processed yet."""
        cursor = self.db.faces.find({"embedding": None}, {"_id": 1, "file_path": 1})
        return list(cursor)

    def set_face_embeddings(self, updates: List[Tuple[ObjectId, np.ndarray]]):
        """(Backend Task) Updates faces with their calculated embeddings."""
        if not updates: return

        bulk_ops = [
            UpdateOne({"_id": face_id}, {"$set": {"embedding": Binary(embedding.tobytes())}})
            for face_id, embedding in updates
        ]

        try:
            self.db.faces.bulk_write(bulk_ops, ordered=False)
            logger.info(f"Successfully saved embeddings for {len(updates)} faces.")
        except Exception as e:
            logger.error(f"An error occurred during bulk embedding update: {e}", exc_info=True)

    def add_person(self, name: str) -> Optional[ObjectId]:
        """(Backend Task) Adds a new person and returns their ID."""
        try:
            result = self.db.persons.insert_one({"name": name, "created_at": time.time()})
            return result.inserted_id
        except DuplicateKeyError:
            person = self.db.persons.find_one({"name": name})
            return person['_id'] if person else None

    def assign_faces_to_person(self, face_ids: List[ObjectId], person_id: Optional[ObjectId]):
        """(Backend Task) Assigns a list of faces to a person (or unassigns if person_id is None)."""
        self.db.faces.update_many(
            {"_id": {"$in": face_ids}},
            {"$set": {"person_id": person_id}}
        )
        logger.info(f"Assigned {len(face_ids)} faces to person ID {person_id}.")

    def update_person_embedding(self, person_id: ObjectId):
        """(Backend Task) Calculates and updates the mean_embedding for a person."""
        faces = list(self.db.faces.find({"person_id": person_id, "embedding": {"$ne": None}}))
        if not faces: return
        embeddings = [np.frombuffer(face['embedding'], dtype=np.float32) for face in faces]
        mean_embedding = np.mean(embeddings, axis=0)
        self.db.persons.update_one(
            {"_id": person_id},
            {"$set": {"mean_embedding": Binary(mean_embedding.tobytes())}}
        )

    def get_embeddings_by_ids(self, face_ids: List[ObjectId]) -> Dict[ObjectId, np.ndarray]:
        """(Backend Task) Fetches embeddings for a specific list of face IDs."""
        cursor = self.db.faces.find({"_id": {"$in": face_ids}, "embedding": {"$ne": None}})
        return {doc['_id']: np.frombuffer(doc['embedding'], dtype=np.float32) for doc in cursor}

    def close(self):
        if self.client:
            self.client.close()
