# app2.py (Final version with All Explicit & Fully Documented Config)

import logging
from typing import Dict, Any, List, Optional
from src.face_recognition_app import FaceRecognitionApp
from src.signals import status_updated, operation_complete, operation_error

logger = logging.getLogger("FaceRecAppLogger")


class App2:
    """
    Interface class (Bridge) that connects a UI to the core application.
    It provides clear, explicit methods and parameters for the UI developer
    and listens for signals from the backend to update the UI.
    """

    def __init__(self, use_gpu: bool = True):
        self.app_core = FaceRecognitionApp(use_gpu=use_gpu)
        self._connect_signals()
        logger.info("App2 initialized and connected to signals.")

    def _connect_signals(self):
        """Connects UI handler methods to the core app's signals."""
        status_updated.connect(self.on_status_update)
        operation_complete.connect(self.on_operation_complete)
        operation_error.connect(self.on_operation_error)

    # --- Signal Handlers (To be implemented in the final GUI) ---
    def on_status_update(self, sender, **kwargs): pass

    def on_operation_complete(self, sender, **kwargs): pass

    def on_operation_error(self, sender, **kwargs): pass

    # --- Public methods for the UI to control the app ---

    def config(
            self,
            mode: str,

            # --- Parameters for different modes ---
            # Note to UI Dev: Only provide the parameters relevant to the selected 'mode'.

            # Used by 'enroll'
            person_name: Optional[str] = None,
            source_face_ids: Optional[List[str]] = None,  # Previously enroll_from_ids

            # Used by 'identify'
            target_face_id: Optional[str] = None,

            # Used by 'compare_one_vs_many'
            # target_face_id is reused here
            # source_face_ids is reused here

            # Used by 'batch_compare'
            source_face_ids_a: Optional[List[str]] = None,
            source_face_ids_b: Optional[List[str]] = None,

            # Used by 'sort'
            sorting_face_ids: Optional[List[str]] = None,

            # Used by 'merge_persons'
            source_person_name: Optional[str] = None,
            destination_person_name: Optional[str] = None,

            # General optional parameters
            comparison_threshold: Optional[float] = None,
            sorting_threshold: Optional[float] = None,
            min_faces_per_group: Optional[int] = None
    ):
        """
        Configures the application for a specific task with explicit, documented parameters.
        The UI developer should call this method before `run_operation()`.

        --- EXAMPLES FOR EACH MODE ---

        1. To index all new faces in the database:
           (This mode takes no extra parameters)
           app_instance.config(mode="index")

        2. To enroll a new person from a LIST OF EXISTING FACE IDs:
           # The UI should first get these IDs, for example, from a user's selection
           # of unassigned faces.
           face_ids = ["62a1...", "62a2...", "62a3..."]
           app_instance.config(
               mode="enroll",
               person_name="New Employee Name",
               source_face_ids=face_ids
           )

        3. To identify a person FROM AN EXISTING FACE ID in the database:
           app_instance.config(
               mode="identify",
               target_face_id="62b1...",
               comparison_threshold=0.75
           )

        4. To sort a list of unassigned faces into new person groups:
           # The UI first needs to fetch these IDs from the DB.
           ids_to_sort = ["62c1...", "62c2...", "62c3..."]
           app_instance.config(
               mode="sort",
               sorting_face_ids=ids_to_sort,
               sorting_threshold=0.88,
               min_faces_per_group=3
           )

        5. To compare one specific face against many others (using their DB IDs):
           target_id = "62d1..."
           source_ids = ["62d2...", "62d3...", "62d4..."]
           app_instance.config(
               mode="compare_one_vs_many",
               target_face_id=target_id,
               source_face_ids=source_ids,
               comparison_threshold=0.7
           )

        6. To compare all faces in group A against all in group B (using their DB IDs):
           ids_a = ["62e1...", "62e2..."]
           ids_b = ["62f3...", "62f4..."]
           app_instance.config(
               mode="batch_compare",
               source_face_ids_a=ids_a,
               source_face_ids_b=ids_b,
               comparison_threshold=0.7
           )

        7. To find potential duplicate persons in the database:
           # This mode compares all known persons against each other.
           app_instance.config(
               mode="find_duplicates",
               comparison_threshold=0.9
           )

        8. To merge two persons:
           app_instance.config(
               mode="merge_persons",
               source_person_name="Ali M.",
               destination_person_name="Ali Mohammadi"
           )

        9. NOTE: The 'add_faces' mode is not part of the core engine.
           It's a helper function in the test_runner to simulate the UI
           adding file paths to the database. The UI should call
           `db_manager.add_face_paths(paths)` directly for this task.
        """
        config_dict = locals()
        config_dict.pop("self")  # Remove 'self' from the dictionary

        # Filter out keys with None values so we don't override core defaults.
        final_config = {k: v for k, v in config_dict.items() if v is not None}

        return self.app_core.config(**final_config)

    def run_operation(self):
        """Starts the configured operation."""
        self.app_core.run()

    def get_status(self) -> dict:
        """Gets the current status from the core application."""
        return self.app_core.status()

    def is_active(self) -> bool:
        """Checks if a process is currently active."""
        return self.app_core.processing_active

    def stop_processing(self):
        """Requests to stop the current background process."""
        self.app_core.stop()