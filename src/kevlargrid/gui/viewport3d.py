"""3-D visualisation viewport.

Provides the :class:`Viewport3D` widget for rendering the deforming grid
and projectile in real time during a simulation run.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]

import numpy as np


class Viewport3D:
    """3-D viewport for grid + projectile visualisation."""

    def __init__(self) -> None: ...

    def build(self) -> None:
        """Construct the DearPyGui 3-D drawing layer."""
        ...

    def update(self, positions: np.ndarray, failed: np.ndarray) -> None:
        """Redraw the grid with the current deformed positions."""
        ...
