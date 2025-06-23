# src/utils.py

import hashlib
import logging
import os
from typing import Optional # <--- اضافه شده

logger = logging.getLogger("FaceRecAppLogger") # استفاده از نام logger که در base_app تعریف شده

def calculate_file_hash(filepath: str, hash_algo='sha256') -> Optional[str]:
    """Calculates the hash of a file's content."""
    h = hashlib.new(hash_algo)
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        logger.warning(f"File not found for hashing: {filepath}")
        return None
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return None