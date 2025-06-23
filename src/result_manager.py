# src/result_manager.py (Final Corrected Version for Sorting)

import os
import json
import cv2
import numpy as np
import shutil
import logging
from typing import Optional, Dict, Any, List, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import RESULTS_FOLDER

logger = logging.getLogger("FaceRecAppLogger")


class ResultManager:
    """
    Handles saving comparison results and managing file operations like sorting.
    """

    def __init__(self, result_folder_path: str = RESULTS_FOLDER):
        self.result_folder_path = result_folder_path
        os.makedirs(self.result_folder_path, exist_ok=True)
        logger.info(f"ResultManager initialized. Results saved to: {self.result_folder_path}")

    def save_results_image(self, target_path: str, results: List[Tuple[str, float]],
                           output_filename: str, max_results: int = 10) -> Optional[str]:
        # This function remains unchanged.
        if not results:
            logger.warning("No results to display for image saving.")
            return None

        results_to_show = min(len(results), max_results)
        fig, axes = plt.subplots(1, results_to_show + 1, figsize=(15, 4))

        if not isinstance(axes, (np.ndarray, list)):
            axes = [axes]

        try:
            target_img = cv2.imread(target_path)
            if target_img is None:
                logger.warning(f"Failed to load target image for display: {target_path}")
                plt.close(fig)
                return None

            axes[0].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Target Image")
            axes[0].axis('off')

            for i in range(results_to_show):
                img_path, similarity = results[i]
                ax = axes[i + 1]
                img = cv2.imread(img_path)
                if img is None:
                    ax.text(0.5, 0.5, "Image load failed", ha='center', va='center', color='red')
                    ax.axis('off')
                    continue

                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(f"{os.path.basename(img_path)}\nSimilarity: {similarity * 100:.2f}%", fontsize=8)
                ax.axis('off')

            plt.tight_layout()
            result_image_path = os.path.join(self.result_folder_path, output_filename)
            plt.savefig(result_image_path)
            return result_image_path
        except Exception as e:
            logger.error(f"Error saving results image: {e}", exc_info=True)
            return None
        finally:
            plt.close(fig)

    def save_results_to_json(self, data: Dict[str, Any], output_filename: str) -> Optional[str]:
        # This function remains unchanged.
        output_path = os.path.join(self.result_folder_path, output_filename)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Results JSON saved to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving results JSON: {e}", exc_info=True)
            return None

    def copy_files_to_folders(self, groups: Dict[str, List[str]], output_folder: str) -> Dict[str, str]:
        """
        Copies files into their respective group folders and returns a
        mapping of {original_path: new_path}.
        """
        os.makedirs(output_folder, exist_ok=True)
        # --- FIX: This function now returns a direct path map for DB updates ---
        path_map = {}
        for group_name, image_paths in groups.items():
            group_folder = os.path.join(output_folder, group_name)
            os.makedirs(group_folder, exist_ok=True)

            for original_path in image_paths:
                filename = os.path.basename(original_path)
                destination_path = os.path.join(group_folder, filename)

                # Handle potential file name conflicts
                if os.path.exists(destination_path):
                    base, ext = os.path.splitext(filename)
                    k = 1
                    while os.path.exists(os.path.join(group_folder, f"{base}_{k}{ext}")):
                        k += 1
                    destination_path = os.path.join(group_folder, f"{base}_{k}{ext}")

                try:
                    shutil.copy2(original_path, destination_path)
                    # Map the original source path to its new destination path
                    path_map[original_path] = destination_path
                except Exception as e:
                    logger.error(f"Failed to copy {original_path} to {destination_path}: {e}", exc_info=True)

        return path_map