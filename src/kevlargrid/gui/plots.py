"""Real-time plot widgets.

Provides :class:`StrainPlot` and :class:`EnergyPlot` widgets that update
live during a running simulation to display peak strain and energy balance
histories.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class StrainPlot:
    """Live peak-strain time-history plot widget."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the DearPyGui plot."""
        ...

    def update(self, time: float, peak_strain: float) -> None:
        """Append a new data point to the plot."""
        ...


class EnergyPlot:
    """Live energy-balance time-history plot widget."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the DearPyGui plot."""
        ...

    def update(self, time: float, energy: dict) -> None:
        """Append new energy data to the plot."""
        ...
