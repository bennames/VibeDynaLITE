"""Results dashboard module.

Provides the :class:`ResultsDashboard` widget for displaying post-run
summary statistics, energy totals, and failure maps.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class ResultsDashboard:
    """Post-simulation results dashboard panel."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the dashboard layout."""
        ...

    def populate(self, results: dict) -> None:
        """Fill the dashboard with data from a completed simulation."""
        ...
