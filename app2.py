# app2.py (Final Corrected Version)

from src.face_recognition_app import FaceRecognitionApp
from config import *


class App2:
    """
    This class acts as a clean interface or bridge between a GUI
    and the core FaceRecognitionApp.
    """

    def __init__(self, use_gpu: bool = True):
        # The core application logic resides in FaceRecognitionApp
        self.app_core = FaceRecognitionApp(use_gpu=use_gpu)
        print("App2 initialized. Ready for commands.")

    def get_health(self) -> dict:
        """Returns the health status of the application."""
        return self.app_core.health()

    def get_status(self) -> dict:
        """Returns the current processing status."""
        return self.app_core.status()

    def set_config(self, **kwargs) -> bool:
        """Sets configuration parameters by passing them to the core app."""
        # Directly pass all keyword arguments to the core's config method
        return self.app_core.config(**kwargs)

    def run_operation(self) -> bool:
        """
        Starts the configured operation (index, compare, sort) in the core app.
        This is the new main entry point for starting a task.
        """
        return self.app_core.run()

    # The methods below are kept for convenience and call the new system
    def start_comparison(self) -> bool:
        """Alias for run_operation to maintain compatibility with older runners."""
        return self.run_operation()

    def start_indexing(self) -> bool:
        """Alias for run_operation, assuming config is set to 'index' mode."""
        return self.run_operation()

    def start_sorting(self) -> bool:
        """Alias for run_operation, assuming config is set to 'sort' mode."""
        return self.run_operation()

    def stop_processing(self) -> bool:
        """Requests to stop the current processing operation."""
        return self.app_core.stop()

    def pause_processing(self) -> bool:
        """Requests to pause the current processing operation."""
        return self.app_core.pause()

    def resume_processing(self) -> bool:
        """Requests to resume a paused operation."""
        return self.app_core.resume()