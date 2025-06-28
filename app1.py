# app2.py (Blinker Signal Receiver with Control Methods)

import logging
from src.face_recognition_app import FaceRecognitionApp
from src.signals import status_updated, operation_complete, operation_error

logger = logging.getLogger("FaceRecAppLogger")


class App2:
    """
    Interface class that connects to the core app's signals
    and can act as a bridge to a user interface.
    It also provides methods to control the application's lifecycle.
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

    # --- Signal Handlers (These would update the GUI) ---

    def on_status_update(self, sender, **kwargs):
        """Receives status updates from the core app."""
        # In a real GUI, you would update elements here.
        # For console testing, we let the test runner print the status.
        pass

    def on_operation_complete(self, sender, **kwargs):
        """Receives signal when an operation is successfully completed."""
        # In a real GUI, this would show a success message.
        message = kwargs.get('message', 'Operation Done.')
        logger.info(f"Success: {message}")

    def on_operation_error(self, sender, **kwargs):
        """Receives signal when an error occurs."""
        # In a real GUI, this would show an error popup.
        message = kwargs.get('message', 'An unknown error occurred.')
        logger.error(f"Error: {message}")

    # --- Public methods to control the app ---

    def config(self, **kwargs) -> bool:
        """Sets configuration parameters."""
        return self.app_core.config(**kwargs)

    def run_operation(self):
        """Starts the configured operation."""
        self.app_core.run()

    def get_status(self) -> dict:
        """Gets the current status from the core application."""
        return self.app_core.status()

    def stop_processing(self):
        """Requests to stop the current background process."""
        self.app_core.stop()

    def is_active(self) -> bool:
        """Checks if a process is currently active."""
        return self.app_core.processing_active
