"""Single point where the Qt binding is chosen.

Every other GUI module imports Qt from here, never from PySide6 directly, so
swapping bindings is a one-file change.

Why this matters beyond tidiness: matplotlib picks its own Qt binding when
its backend is first imported, and loading two different Qt bindings into one
process crashes at the C++ level. PyQt6 is installed in this venv (matplotlib
pulls it in as a GUI backend), so without forcing QT_API here, matplotlib
would happily import PyQt6 alongside our PySide6 and take the app down. The
env vars below must be set before matplotlib is imported anywhere - importing
this module first is what guarantees that.
"""
import os

os.environ.setdefault("QT_API", "pyside6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PySide6.QtCore import Qt, Signal, Slot  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("QtAgg")

__all__ = ["QtCore", "QtGui", "QtWidgets", "Qt", "Signal", "Slot"]
