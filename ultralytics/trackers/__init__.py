# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Tracker exports with lazy imports.

Keeping these imports lazy avoids importing the optional ``lap`` dependency when
callers only need a tracker that uses SciPy (for example ``mask_aware``).
"""

from typing import Any

__all__ = "BOTSORT", "BYTETracker", "register_tracker"


def __getattr__(name: str) -> Any:
    if name == "BOTSORT":
        from .bot_sort import BOTSORT

        return BOTSORT
    if name == "BYTETracker":
        from .byte_tracker import BYTETracker

        return BYTETracker
    if name == "register_tracker":
        from .track import register_tracker

        return register_tracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
