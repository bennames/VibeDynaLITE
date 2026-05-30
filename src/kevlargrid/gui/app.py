"""Main DearPyGui application entry point.

Sets up the DearPyGui context, viewport, and primary window layout,
then starts the render loop.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


def launch() -> None:
    """Initialise and launch the KevlarGrid GUI application."""
    ...
