# src/signals.py
# This file defines all the signals that the application can emit.

from blinker import Signal

# A general signal for status updates, carrying progress and details.
# The UI will listen to this to update progress bars and status labels.
status_updated = Signal(
    doc="Sent when the status of a background operation changes. "
        "Provides progress, details, and the current status string."
)

# A signal that fires when any operation completes successfully.
# It can carry a final summary message and any relevant result data.
operation_complete = Signal(
    doc="Sent when a background task finishes without errors. "
        "Provides a summary message and optional results payload."
)

# A signal that fires when any operation fails with an error.
# It carries the error message for display to the user.
operation_error = Signal(
    doc="Sent when a background task fails due to an exception. "
        "Provides an error message."
)

# A specific signal for when a new person is successfully enrolled.
enrollment_succeeded = Signal(
    doc="Sent specifically when a new person is successfully enrolled. "
        "Provides the person's name and the number of faces added."
)

# A new signal specifically for broadcasting log messages to the UI.
# Payload will contain 'level' (e.g., INFO, ERROR) and 'message'.
log_emitted = Signal(doc="Sent whenever a log message is generated.")