"""Configuration input panel.

Provides the :class:`ConfigPanel` widget for entering material, grid,
projectile, and simulation parameters through the GUI.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class ConfigPanel:
    """DearPyGui panel for simulation configuration input."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the panel's DearPyGui widgets."""
        ...

    def get_config(self) -> dict:
        """Read the current widget values and return a config dict."""
        raise NotImplementedError("Stub")
