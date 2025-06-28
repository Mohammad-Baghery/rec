# src/feature_extractor.py (Final Corrected Version)

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import logging
from typing import Optional, Dict, Any, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils import calculate_file_hash
from config import DEFAULT_BATCH_SIZE

logger = logging.getLogger("FaceRecAppLogger")


class FeatureExtractor:
    """
    Optimized for Pre-Cropped Faces.
    This class's sole responsibility is to take image paths and return extracted features.
    It does not interact with the database directly.
    """

    def __init__(self, device: torch.device, **kwargs):
        self.device = device
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        logger.info("FeatureExtractor initialized in pre-cropped face mode.")

    def process_images(self, image_paths: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> List[Dict[str, Any]]:
        """
        Loads pre-cropped face images, and returns a list of feature dicts.
        Each dict contains path, hash, and embedding.
        """
        logger.info(f"Processing {len(image_paths)} images to extract features...")

        # Prepare data for processing
        images_to_process_data = []
        for img_path in tqdm(image_paths, desc="Loading and Hashing Images"):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"Could not read image, skipping: {img_path}")
                    continue

                file_hash = calculate_file_hash(img_path)
                if not file_hash:
                    logger.warning(f"Could not calculate hash, skipping: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_to_process_data.append((img_path, file_hash, image_rgb))
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")

        if not images_to_process_data:
            logger.warning("No valid images found to process.")
            return []

        # Transform all images
        all_tensors = [self.transform(Image.fromarray(data[2])) for data in images_to_process_data]
        metadata_list = [{'path': data[0], 'hash': data[1]} for data in images_to_process_data]

        if not all_tensors:
            return []

        # Run FaceNet in batches for maximum speed
        all_features = []
        try:
            dataset = torch.utils.data.TensorDataset(torch.stack(all_tensors))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            all_embeddings = []
            with torch.no_grad():
                for batch_tensors in tqdm(loader, desc="Extracting Embeddings"):
                    batch_on_device = batch_tensors[0].to(self.device)
                    embeddings = self.facenet(batch_on_device)
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    all_embeddings.append(embeddings.cpu())

            embeddings_batch_cpu = torch.cat(all_embeddings).numpy()

            # Combine metadata with the extracted embeddings
            for i, embedding in enumerate(embeddings_batch_cpu):
                meta = metadata_list[i]
                features = {
                    'file_path': meta['path'],
                    'file_hash': meta['hash'],
                    'embedding': embedding.astype(np.float32)
                }
                all_features.append(features)

        except Exception as e:
            logger.error(f"Error during batched FaceNet inference: {e}", exc_info=True)

        logger.info(f"Feature extraction complete. Extracted {len(all_features)} features.")
        return all_features
