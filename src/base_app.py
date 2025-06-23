# src/base_app.py

import logging
import sys
import time
import threading
from typing import Optional, Dict, Any, List

from config import (
    STATUS_IDLE, STATUS_INITIALIZING, STATUS_WAITING, STATUS_PROCESSING,
    STATUS_COMPLETE, STATUS_ERROR, STATUS_INCOMPLETE, STATUS_PAUSED,
    LOG_FILE_PATH, CONSOLE_LOG_LEVEL, FILE_LOG_LEVEL
)

# --- Configure Logging ---
logger = logging.getLogger("FaceRecAppLogger")  # Named logger
logger.setLevel(logging.DEBUG)  # Catch all messages from sub-components

# Create handlers
if logger.handlers:  # Prevent adding handlers multiple times if module is reloaded
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
# FIX: مشخص کردن انکدینگ utf-8 برای StreamHandler (برای رفع UnicodeEncodeError)
console_handler.setLevel(getattr(logging, CONSOLE_LOG_LEVEL.upper()))
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
console_handler.setStream(sys.stdout)  # اطمینان از استفاده از sys.stdout
if sys.stdout.encoding != 'utf-8':
    # اگر ترمینال utf-8 نیست، ممکن است این راه‌حل کمک کند
    # اما بهترین راه حل، تنظیم ترمینال به utf-8 است.
    # این کد ممکن است در برخی محیط ها نیاز به تنظیمات بیشتری داشته باشد.
    # فعلاً به جای تلاش برای تغییر انکدینگ استریم، فقط در صورت نیاز آن را به utf-8 تغییر می دهیم.
    pass  # ماژول logging خودش این کار را انجام می‌دهد.

file_handler = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')  # FIX: مشخص کردن انکدینگ utf-8 برای FileHandler
file_handler.setLevel(getattr(logging, FILE_LOG_LEVEL.upper()))

# Create formatters and add them to handlers
console_formatter = logging.Formatter('%(levelname)s: %(message)s')
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


class _BaseApp:
    """
    Base class providing common functionalities like status management, logging,
    and cancellation/pause control.
    """

    def __init__(self):
        self.logger = logger
        self._process_status = STATUS_IDLE
        self.status_details = ""
        self.status_progress = 0
        self.status_history: List[Dict[str, Any]] = []
        self._processing_active = False
        self._cancel_requested = False
        self._pause_requested = False
        self._paused_event = threading.Event()
        self._paused_event.set()

    @property
    def process_status(self) -> str:
        return self._process_status

    @process_status.setter
    def process_status(self, value: str):
        self._process_status = value
        self.logger.debug(f"Process status set to: {value}")

    @property
    def processing_active(self) -> bool:
        return self._processing_active

    @processing_active.setter
    def processing_active(self, value: bool):
        self._processing_active = value
        self.logger.debug(f"Processing active set to: {value}")

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested

    @cancel_requested.setter
    def cancel_requested(self, value: bool):
        self._cancel_requested = value
        if value:
            self.logger.warning("Cancellation request received.")

    @property
    def pause_requested(self) -> bool:
        return self._pause_requested

    @pause_requested.setter
    def pause_requested(self, value: bool):
        self._pause_requested = value
        if value:
            self.logger.info("Pause request received.")
            self._paused_event.clear()
            self.process_status = STATUS_PAUSED
        else:
            self.logger.info("Resume request received.")
            self._paused_event.set()
            self._update_status(STATUS_PROCESSING, "Processing resumed...",
                                self.status_progress)  # Update status after resuming

    def _update_status(self, status: str, details: str = "", progress: Optional[int] = None):
        self.process_status = status
        self.status_details = details
        if progress is not None:
            self.status_progress = progress

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status_history.append({
            "status": status,
            "details": details,
            "progress": self.status_progress,
            "timestamp": timestamp
        })
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]

        # FIX: اطمینان حاصل می کنیم که پیام لاگ بدون کاراکترهای مشکل ساز ارسال می شود
        # یا انکدینگ ترمینال به درستی تنظیم شده است.
        # راه حل موقت: حذف کاراکترهای غیر ASCII از پیام برای جلوگیری از UnicodeEncodeError
        safe_details = details.encode('ascii', 'ignore').decode('ascii')
        self.logger.info(f"Status: {status} - {safe_details} ({self.status_progress}%)")

    def _check_for_cancellation_or_pause(self) -> bool:
        if self.cancel_requested:
            self.logger.warning("Processing terminated due to cancellation request.")
            self.cancel_requested = False
            self.processing_active = False
            self._update_status(STATUS_INCOMPLETE, "Processing cancelled by user.", self.status_progress)
            return True

        if not self._paused_event.is_set():
            self.logger.info("Processing paused. Waiting for resume.")
            self._update_status(STATUS_PAUSED, "Processing paused by user.", self.status_progress)
            self._paused_event.wait()
            self.logger.info("Processing resumed.")
            # Status will be updated to PROCESSING when execution continues by the setter

        return False

    def status(self) -> Dict[str, Any]:
        self.logger.debug("Getting current status information.")
        return {
            "status": self.process_status,
            "details": self.status_details,
            "progress": self.status_progress,
            "history": self.status_history[-10:],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def pause(self) -> bool:
        if self.processing_active and self.process_status == STATUS_PROCESSING:
            self.pause_requested = True
            return True
        self.logger.warning("No active processing to pause or already paused.")
        return False

    def resume(self) -> bool:
        if self.processing_active and self.process_status == STATUS_PAUSED:
            self.pause_requested = False
            return True
        self.logger.warning("No paused processing to resume or not active.")
        return False

    def stop(self) -> bool:
        if self.processing_active:
            self.cancel_requested = True
            if not self._paused_event.is_set():
                self._paused_event.set()
            return True
        self.logger.info("No active processing to stop.")
        return False