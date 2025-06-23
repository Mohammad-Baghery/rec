# src/signals.py
# This file defines all the signals that the application can emit.

from blinker import Signal

# A general signal for status updates, carrying progress percentage and details.
# The UI will listen to this to update progress bars and status labels.
status_updated = Signal("status_updated")

# A signal that fires when any operation completes successfully.
# It can carry a final summary message.
operation_complete = Signal("operation_complete")

# A signal that fires when any operation fails with an error.
# It carries the error message.
operation_error = Signal("operation_error")

# You can also create more specific signals for different events.
# For example, when a new person is successfully enrolled.
enrollment_succeeded = Signal("enrollment_succeeded")