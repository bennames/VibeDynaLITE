"""3-D perspective rendering viewport widget.

Translates 3-D spring-mass nodes and projectile positions into projected
2-D coordinates drawn dynamically on an ImGui canvas drawlist.
Supports trackpad gestures, sliders, and color-coded strain representations.
"""

from __future__ import annotations

import math
from typing import Any

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]

import numpy as np

from kevlargrid.solver.grid import Grid


class Viewport3D:
    """3-D viewport for grid + projectile visual quality rendering."""

    def __init__(self) -> None:
        self.canvas_tag = "viewport_drawlist"
        self.group_tag = "viewport_group"
        self.slider_yaw = "viewport_yaw_slider"
        self.slider_pitch = "viewport_pitch_slider"
        self.slider_zoom = "viewport_zoom_slider"
        self.layer_group = "viewport_layer_group"

        # Camera parameter state (macOS trackpad defaults)
        self.yaw = 0.785  # ~45 degrees in radians
        self.pitch = 0.523  # ~30 degrees in radians
        self.distance = 0.3  # camera distance (meters)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.focal_length = 600.0

        # Canvas rendering centers
        self.center_x = 350.0
        self.center_y = 200.0

        # Active grid metadata
        self.grid: Grid | None = None
        self.n_plies = 1
        self.n_nodes_per_layer = 121
        self.grid_center = np.zeros(3)

        # Layer visibility flags
        self.layer_visibility: list[bool] = [True] * 10

        # Mouse Drag gesture trackers
        self.drag_start = [0.0, 0.0]
        self.is_dragging = False

    def build(self) -> None:
        """Construct the DearPyGui 3-D drawing layout, sliders, and handlers."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, border=True, height=430, width=-1):
            dpg.add_text("3D Dynamic Woven Mesh Viewport", color=[0, 191, 255])
            dpg.add_separator()

            # Canvas Drawing Canvas
            dpg.add_drawlist(tag=self.canvas_tag, width=700, height=310)

            # Bind Canvas Mouse Gestures Handlers
            self._setup_mouse_handlers()

            # Responsive Control Sliders Toolbar Row
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="Yaw",
                    tag=self.slider_yaw,
                    width=120,
                    min_value=-180.0,
                    max_value=180.0,
                    default_value=45.0,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Pitch",
                    tag=self.slider_pitch,
                    width=120,
                    min_value=-89.0,
                    max_value=89.0,
                    default_value=30.0,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Zoom",
                    tag=self.slider_zoom,
                    width=120,
                    min_value=0.05,
                    max_value=1.5,
                    default_value=0.3,
                    callback=self._on_slider_change,
                )

            # Per-Ply visibility checkbox row
            dpg.add_group(tag=self.layer_group, horizontal=True)

    def _setup_mouse_handlers(self) -> None:
        """Register trackpad-friendly keyboard/mouse coordinate listeners."""
        if dpg is None:  # pragma: no cover
            return

        handler_reg = "viewport_mouse_handlers"
        if dpg.does_item_exist(handler_reg):
            dpg.delete_item(handler_reg)

        with dpg.handler_registry(tag=handler_reg):
            dpg.add_mouse_click_handler(callback=self._on_mouse_click)
            dpg.add_mouse_drag_handler(callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(callback=self._on_mouse_release)

    def _on_mouse_release(self, sender: str, app_data: Any) -> None:
        """Track mouse release events to end dragging state."""
        self.is_dragging = False

    def _on_mouse_click(self, sender: str, app_data: Any) -> None:
        """Track mouse click start to manage orbit or panning translation coordinates."""
        if dpg is None:  # pragma: no cover
            return
        if dpg.is_item_hovered(self.canvas_tag):
            self.drag_start = dpg.get_mouse_pos(local=False)
            self.is_dragging = True

    def _on_mouse_drag(self, sender: str, app_data: Any) -> None:
        """Responsive mouse drag callback with macOS Trackpad keyboard modifiers."""
        if dpg is None or not self.is_dragging:  # pragma: no cover
            return

        mouse_pos = dpg.get_mouse_pos(local=False)
        dx = mouse_pos[0] - self.drag_start[0]
        dy = mouse_pos[1] - self.drag_start[1]
        self.drag_start = mouse_pos

        # Check Keyboard Modifiers for Trackpad Friendliness
        is_shift = dpg.is_key_down(16)
        is_option = dpg.is_key_down(18) or dpg.is_key_down(17)

        if is_shift:
            # 1. Shift + Left-drag = Pan camera
            self.pan_x += dx * 0.00015 * self.distance
            self.pan_y -= dy * 0.00015 * self.distance
        elif is_option:
            # 2. Option + Left-drag = Zoom camera
            self.distance = min(1.5, max(0.02, self.distance + dy * 0.001 * self.distance))
            dpg.set_value(self.slider_zoom, self.distance)
        else:
            # 3. Standard Left-drag = Orbit Yaw and Pitch
            self.yaw += dx * 0.004
            self.pitch = min(math.pi / 2 - 0.01, max(-math.pi / 2 + 0.01, self.pitch + dy * 0.004))

            # Sync to sliders UI
            dpg.set_value(self.slider_yaw, math.degrees(self.yaw))
            dpg.set_value(self.slider_pitch, math.degrees(self.pitch))

        self.redraw()

    def _on_slider_change(self, sender: str, app_data: float) -> None:
        """Update camera parameters directly from UI sliders."""
        if dpg is None:  # pragma: no cover
            return

        if sender == self.slider_yaw:
            self.yaw = math.radians(app_data)
        elif sender == self.slider_pitch:
            self.pitch = math.radians(app_data)
        elif sender == self.slider_zoom:
            self.distance = app_data

        self.redraw()

    def reset(self, grid: Grid, n_plies: int = 1, n_nodes_per_layer: int = 121) -> None:
        """Store grid model coordinates and regenerate layer visibility checkboxes.

        Parameters
        ----------
        grid : Grid
            Simulated Lumped-Mass spring network.
        n_plies : int
            Ply layers.
        n_nodes_per_layer : int
            Discrete node counts in a single ply.
        """
        self.grid = grid
        self.n_plies = n_plies
        self.n_nodes_per_layer = n_nodes_per_layer
        self.layer_visibility = [True] * n_plies

        # Find bounds of single grid center
        if len(grid.nodes) > 0:
            # Find center of nodes
            self.grid_center = np.mean(grid.nodes[:n_nodes_per_layer], axis=0)
        else:  # pragma: no cover
            self.grid_center = np.zeros(3)

        # Recenter camera panning
        self.pan_x = 0.0
        self.pan_y = 0.0

        if dpg is None:  # pragma: no cover
            return

        # Redraw layer check boxes dynamically
        if dpg.does_item_exist(self.layer_group):
            dpg.delete_item(self.layer_group, children_only=True)

            if n_plies > 1:
                dpg.add_text("Layer Visibility: ", parent=self.layer_group)
                for ply in range(n_plies):
                    dpg.add_checkbox(
                        label=f"Ply {ply}",
                        tag=f"viewport_chk_ply_{ply}",
                        default_value=True,
                        parent=self.layer_group,
                        callback=self._on_layer_toggle,
                        user_data=ply,
                    )

        self.redraw()

    def _on_layer_toggle(self, sender: str, app_data: bool, user_data: int) -> None:
        """Triggered when per-layer checkboxes are checked/unchecked."""
        ply_idx = user_data
        if ply_idx < len(self.layer_visibility):
            self.layer_visibility[ply_idx] = app_data
        self.redraw()

    def redraw(self) -> None:
        """Clear canvas drawlist and project coordinates to draw springs + projectile."""
        if dpg is None or self.grid is None:  # pragma: no cover
            return

        if not dpg.does_item_exist(self.canvas_tag):  # pragma: no cover
            return

        dpg.delete_item(self.canvas_tag, children_only=True)

        # Render nothing if node arrays are empty
        if len(self.grid.nodes) == 0:  # pragma: no cover
            return

        # 1. Calculate camera rotations and screen translations
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)

        # 3D Yaw-Pitch camera projection coordinates rotation matrix
        R = np.array([[cy, 0.0, -sy], [-sy * sp, cp, -cy * sp], [sy * cp, sp, cy * cp]])  # noqa: N806

        # 2. Translate nodes relative to center target point
        nodes_rel = self.grid.nodes - self.grid_center
        # Rotate coordinates
        nodes_rotated = nodes_rel @ R.T

        # Camera translations including horizontal/vertical pans
        cam_x = nodes_rotated[:, 0] + self.pan_x
        cam_y = nodes_rotated[:, 1] + self.pan_y
        cam_z = nodes_rotated[:, 2] + self.distance
        cam_z = np.maximum(cam_z, 1e-4)  # clip zero bounds

        # Project coordinates onto viewport canvas
        scr_x = self.center_x + (self.focal_length * cam_x / cam_z)
        scr_y = self.center_y - (self.focal_length * cam_y / cam_z)

        # 3. Draw springs with active layer filtering
        springs = self.grid.springs
        stiffnesses = self.grid.stiffnesses
        rest_lengths = self.grid.rest_lengths
        failed = self.grid.failed

        # Calculate live engineering strain for color-scale mapping
        p1 = self.grid.nodes[springs[:, 0]]
        p2 = self.grid.nodes[springs[:, 1]]
        lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
        strains = (lengths - rest_lengths) / rest_lengths

        # Color Spectrum mapping threshold (e.g. failure strain)
        fail_thresh = 0.036
        if len(stiffnesses) > 0:
            # Try to fetch from default active material failure strain limit
            fail_thresh = 0.036

        # Draw lines in drawlist
        for j in range(len(springs)):
            u, v = springs[j, 0], springs[j, 1]

            # Map spring to ply layer index
            ply_idx = u // self.n_nodes_per_layer
            if ply_idx < len(self.layer_visibility) and not self.layer_visibility[ply_idx]:
                continue  # Skip drawing hidden layers

            x_u, y_u = float(scr_x[u]), float(scr_y[u])
            x_v, y_v = float(scr_x[v]), float(scr_y[v])

            # Filter off-screen clippings
            if (x_u < 0 or x_u > 700 or y_u < 0 or y_u > 310) and (
                x_v < 0 or x_v > 700 or y_v < 0 or y_v > 310
            ):
                continue

            if failed[j]:
                # Deep brick red transparent lines for ruptured springs
                dpg.draw_line(
                    [x_u, y_u],
                    [x_v, y_v],
                    color=[139, 0, 0, 45],
                    thickness=1,
                    parent=self.canvas_tag,
                )
            else:
                # Active spring: Map strain to green/blue -> yellow -> red spectrum
                eps = strains[j]
                ratio = min(1.0, max(0.0, eps / fail_thresh))

                if ratio <= 0.5:
                    # Blue [0,191,255] to Yellow [255,215,0]
                    r = int(0 + (255 - 0) * (ratio / 0.5))
                    g = int(191 + (215 - 191) * (ratio / 0.5))
                    b = int(255 + (0 - 255) * (ratio / 0.5))
                else:
                    # Yellow [255,215,0] to Crimson [220,20,60]
                    r = int(255 + (220 - 255) * ((ratio - 0.5) / 0.5))
                    g = int(215 + (20 - 215) * ((ratio - 0.5) / 0.5))
                    b = int(0 + (60 - 0) * ((ratio - 0.5) / 0.5))

                dpg.draw_line(
                    [x_u, y_u],
                    [x_v, y_v],
                    color=[r, g, b, 230],
                    thickness=1.5,
                    parent=self.canvas_tag,
                )

    def update(self, positions: np.ndarray, failed: np.ndarray) -> None:
        """Update node coordinate positions dynamically and trigger a redraw.

        Parameters
        ----------
        positions : np.ndarray
            Current dynamic 3D node coordinates, shape (n_nodes, 3).
        failed : np.ndarray
            Current boolean spring failure statuses, shape (n_springs,).
        """
        if self.grid is not None:
            self.grid.nodes = positions
            self.grid.failed = failed
            self.redraw()

    def draw_projectile(
        self, position: np.ndarray, blade_width: float, edge_thickness: float
    ) -> None:
        """Render a 3D wireframe box representing the striking metal projectile.

        Parameters
        ----------
        position : np.ndarray
            3-D coordinate of projectile's center, shape (3,).
        blade_width : float
            Width along X axis (meters).
        edge_thickness : float
            Thickness along Y axis (meters).
        """
        if dpg is None or self.grid is None:  # pragma: no cover
            return

        # Build 3D bounding vertices relative to center position
        w_h = blade_width / 2.0
        t_h = edge_thickness / 2.0
        h_h = 0.005  # semi-height of bounding container

        # 8 box corners
        offsets = np.array(
            [
                [-w_h, -t_h, -h_h],
                [w_h, -t_h, -h_h],
                [w_h, t_h, -h_h],
                [-w_h, t_h, -h_h],
                [-w_h, -t_h, h_h],
                [w_h, -t_h, h_h],
                [w_h, t_h, h_h],
                [-w_h, t_h, h_h],
            ]
        )

        # World corners
        corners = position + offsets

        # Project using camera rotation
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        R = np.array([[cy, 0.0, -sy], [-sy * sp, cp, -cy * sp], [sy * cp, sp, cy * cp]])  # noqa: N806

        c_rel = corners - self.grid_center
        c_rot = c_rel @ R.T

        c_cam_x = c_rot[:, 0] + self.pan_x
        c_cam_y = c_rot[:, 1] + self.pan_y
        c_cam_z = c_rot[:, 2] + self.distance
        c_cam_z = np.maximum(c_cam_z, 1e-4)

        scr_x = self.center_x + (self.focal_length * c_cam_x / c_cam_z)
        scr_y = self.center_y - (self.focal_length * c_cam_y / c_cam_z)

        # Box edge indexes (12 edges total)
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom loop
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top loop
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical lines
        ]

        # Draw wireframe container lines (silver/white)
        for u, v in edges:
            x_u, y_u = float(scr_x[u]), float(scr_y[u])
            x_v, y_v = float(scr_x[v]), float(scr_y[v])
            dpg.draw_line(
                [x_u, y_u],
                [x_v, y_v],
                color=[230, 230, 250, 240],
                thickness=2.5,
                parent=self.canvas_tag,
            )
