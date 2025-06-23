# src/face_db_manager.py (Final Corrected Version)

import sqlite3
import numpy as np
import json
import logging
import os
import time
from typing import Optional, Dict, Any, List, Tuple

from config import FACE_DATABASE_FILE

logger = logging.getLogger("FaceRecAppLogger")


class FaceDBManager:
    """
    Manages a two-table database for persons and their associated faces.
    """

    def __init__(self, db_path: str = FACE_DATABASE_FILE):
        self.db_path = db_path
        self._conn = None
        self._connect()
        self._create_tables()
        logger.info(f"Initialized FaceDBManager with new schema: {self.db_path}")

    def _connect(self):
        try:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.row_factory = sqlite3.Row
            logger.debug("SQLite database connected successfully.")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self):
        cursor = self._conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    mean_embedding BLOB,
                    notes TEXT,
                    created_at TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS faces (
                    face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (person_id) ON DELETE CASCADE
                )
            ''')
            self._conn.commit()
            logger.debug("Tables 'persons' and 'faces' ensured to exist.")
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {e}")
            self._conn.rollback()

    def add_person(self, name: str, notes: str = "") -> Optional[int]:
        if not self._conn: return None
        cursor = self._conn.cursor()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            cursor.execute("INSERT INTO persons (name, notes, created_at) VALUES (?, ?, ?)", (name, notes, timestamp))
            self._conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            logger.warning(f"Person with name '{name}' already exists.")
            return None

    def get_person_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not self._conn: return None
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def add_faces_to_person(self, person_id: int, features_list: List[Dict[str, Any]]):
        if not features_list or not self._conn: return
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        face_data = [(person_id, f.get('file_hash'), f.get('file_path'), f.get('embedding').tobytes(), timestamp) for f
                     in features_list if f.get('embedding') is not None]
        cursor = self._conn.cursor()
        try:
            cursor.execute("BEGIN TRANSACTION;")
            cursor.executemany(
                "INSERT OR IGNORE INTO faces (person_id, file_hash, file_path, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                face_data)
            self._conn.commit()
            logger.info(f"Added {len(face_data)} new faces for person ID {person_id}.")
        except sqlite3.Error as e:
            logger.error(f"Failed to add faces for person ID {person_id}: {e}")
            self._conn.rollback()

    def update_person_embedding(self, person_id: int):
        if not self._conn: return
        cursor = self._conn.cursor()
        cursor.execute("SELECT embedding FROM faces WHERE person_id = ?", (person_id,))
        rows = cursor.fetchall()
        if not rows: return
        embeddings = [np.frombuffer(row['embedding'], dtype=np.float32) for row in rows]
        mean_embedding = np.mean(embeddings, axis=0)
        try:
            cursor.execute("UPDATE persons SET mean_embedding = ? WHERE person_id = ?",
                           (mean_embedding.tobytes(), person_id))
            self._conn.commit()
            logger.info(f"Updated mean embedding for person ID {person_id}.")
        except sqlite3.Error as e:
            logger.error(f"Failed to update mean embedding for person ID {person_id}: {e}")
            self._conn.rollback()

    def get_all_person_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        if not self._conn: return []
        cursor = self._conn.cursor()
        cursor.execute("SELECT person_id, name, mean_embedding FROM persons WHERE mean_embedding IS NOT NULL")
        return [(row['person_id'], row['name'], np.frombuffer(row['mean_embedding'], dtype=np.float32)) for row in
                cursor.fetchall()]

    def get_all_face_embeddings_with_paths(self) -> List[Tuple[str, np.ndarray]]:
        """
        NEW METHOD: Retrieves file_path and embedding for every single face in the database.
        This is needed to build the Annoy index for fast lookups.
        """
        if not self._conn: return []
        cursor = self._conn.cursor()
        cursor.execute("SELECT file_path, embedding FROM faces WHERE embedding IS NOT NULL")
        return [(row['file_path'], np.frombuffer(row['embedding'], dtype=np.float32)) for row in cursor.fetchall()]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("SQLite database connection closed.")

    def get_all_face_data_for_index(self) -> List[Tuple[str, int, np.ndarray]]:
        """
        NEW METHOD: Retrieves file_path, person_id, and embedding for every single face.
        This is needed to build the rich Annoy index.
        """
        if not self._conn: return []
        cursor = self._conn.cursor()
        cursor.execute("SELECT file_path, person_id, embedding FROM faces WHERE embedding IS NOT NULL")
        return [
            (row['file_path'], row['person_id'], np.frombuffer(row['embedding'], dtype=np.float32))
            for row in cursor.fetchall()
        ]

    # Add this to face_db_manager.py
    def get_person_by_id(self, person_id: int) -> Optional[Dict[str, Any]]:
        if not self._conn: return None
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

