"""Simulation controls widget.

Provides play / pause / step / reset controls and progress indicators
for the explicit solver loop.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class SimulationControls:
    """DearPyGui widget group for simulation run controls."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the control buttons and progress bar."""
        ...
