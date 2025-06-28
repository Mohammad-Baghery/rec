# src/base_app.py

import logging
import sys
import time
import threading
from typing import Optional, Dict, Any, List

# Import signals and configs
from src.signals import log_emitted
from config import *


# --- Custom Log Handler to Emit Signals ---
class SignalLogHandler(logging.Handler):
    """
    A custom logging handler that emits a blinker signal for each log record.
    """

    def __init__(self):
        super().__init__()

    def emit(self, record):
        """
        Overrides the default emit method.
        Formats the log record and sends it as a signal.
        """
        message = self.format(record)
        # Send the signal with the log level and the formatted message
        log_emitted.send(
            'log_handler',
            level=record.levelname,
            message=message
        )


# --- Configure Logging ---
# This setup block ensures that logging is configured only once.
logger = logging.getLogger("FaceRecAppLogger")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)

    # 1. Console Handler (for developers)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, CONSOLE_LOG_LEVEL.upper()))
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (for persistent logs)
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
    file_handler.setLevel(getattr(logging, FILE_LOG_LEVEL.upper()))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 3. Signal Handler (for the UI)
    signal_handler = SignalLogHandler()
    signal_handler.setLevel(logging.INFO)  # Set the level for what logs you want to see in the UI
    logger.addHandler(signal_handler)


class _BaseApp:
    """
    Base class providing common functionalities like status management, logging,
    and cancellation/pause control for background tasks.
    """

    def __init__(self):
        self.logger = logger
        self._process_status = STATUS_IDLE
        self.status_details = ""
        self.status_progress = 0
        self.status_history: List[Dict[str, Any]] = []
        self._processing_active = False
        self._cancel_requested = False
        self._paused_event = threading.Event()
        self._paused_event.set()  # Default to not paused (event is set)

    @property
    def processing_active(self) -> bool:
        """Returns True if a background process is currently running or paused."""
        return self._processing_active

    @processing_active.setter
    def processing_active(self, value: bool):
        self._processing_active = value
        self.logger.debug(f"Processing active set to: {value}")

    def _update_status(self, status: str, details: str = "", progress: Optional[int] = None):
        """Updates the internal state of the application."""
        self.process_status = status
        self.status_details = details
        if progress is not None:
            self.status_progress = min(100, max(0, progress))  # Clamp progress between 0 and 100

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status_history.append({
            "status": status,
            "details": details,
            "progress": self.status_progress,
            "timestamp": timestamp
        })
        # Keep history from getting too large
        if len(self.status_history) > 100:
            self.status_history.pop(0)

        # Log the main status update through the standard logger
        self.logger.info(f"Status: {status} - {details} ({self.status_progress}%)")

    def _check_for_cancellation_or_pause(self) -> bool:
        """
        Checks if a stop or pause has been requested.
        This should be called inside long-running loops in child classes.
        """
        if self._cancel_requested:
            self.logger.warning("Processing terminated due to cancellation request.")
            raise InterruptedError("Operation cancelled by user.")

        # Pausing logic can be added here if needed in the future
        # if not self._paused_event.is_set():
        #     ...

        return False

    def status(self) -> Dict[str, Any]:
        """Returns the current status of the application as a dictionary."""
        return {
            "status": self.process_status,
            "details": self.status_details,
            "progress": self.status_progress
        }

    def stop(self):
        """Sets the flag to stop the current background process."""
        if self.processing_active:
            self.cancel_requested = True
            self.logger.warning("Cancellation request received.")
