"""3-D perspective rendering viewport widget.

Translates 3-D spring-mass nodes and projectile positions into projected
2-D coordinates drawn dynamically on an ImGui canvas drawlist.
Supports trackpad gestures, sliders, and color-coded strain representations.
"""

from __future__ import annotations

import math
import threading
from typing import Any

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]

import numpy as np

try:
    import pyvista as pv
    import vtk

    HAS_PYVISTA = True
except ImportError:
    pv = None  # type: ignore[assignment]
    vtk = None  # type: ignore[assignment]
    HAS_PYVISTA = False

from kevlargrid.solver.grid import Grid
from kevlargrid.utils import get_logger

logger = get_logger("gui.viewport3d")


class Viewport3D:
    """3-D viewport for grid + projectile visual quality rendering."""

    def __init__(self) -> None:
        self.render_lock = threading.RLock()
        self.canvas_tag = "viewport_drawlist"
        self.group_tag = "viewport_group"
        self.slider_yaw = "viewport_yaw_slider"
        self.slider_pitch = "viewport_pitch_slider"
        self.slider_zoom = "viewport_zoom_slider"
        self.slider_pan_x = "viewport_pan_x_slider"
        self.slider_pan_y = "viewport_pan_y_slider"
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
        self.drag_button: int | None = None

        # Projectile state cache for renderer
        self.proj_position = np.zeros(3)
        self.proj_blade_width = 0.02
        self.proj_edge_thickness = 0.005
        self.proj_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.proj_shape_type = "box"
        self.proj_radius = 0.005
        self.proj_length = 0.01
        self.proj_edge_radius = 0.0
        self.proj_ogive_multiplier = 2.0
        self.proj_span = 0.05
        self.proj_root_chord = 0.01
        self.proj_tip_chord = 0.005
        self.proj_twist = 15.0
        self.proj_thickness_ratio = 12.0
        self.proj_tip_radius = 0.002

        # PyVista Offscreen visualization objects
        self.plotter: pv.Plotter | None = None
        self.mesh: pv.PolyData | None = None
        self.actor: Any = None
        self.proj_actor: Any = None
        self.has_pyvista = False

        self.width = 700
        self.height = 310
        self._needs_plotter_resize = False
        self._target_plotter_width = 700
        self._target_plotter_height = 310
        self._texture_w = 700
        self._texture_h = 310
        self._needs_redraw = False

    def build(self) -> None:
        """Construct the DearPyGui 3-D drawing layout, sliders, and handlers."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, border=True, height=430, width=-1):
            dpg.add_text("3D Dynamic Woven Mesh Viewport", color=[0, 191, 255])
            dpg.add_separator()

            # Canvas Drawing Canvas
            dpg.add_drawlist(tag=self.canvas_tag, width=self.width, height=self.height)

            # Bind Canvas Mouse Gestures Handlers
            self._setup_mouse_handlers()

            # Responsive Control Sliders Toolbar Row
            with dpg.group(horizontal=True):
                dpg.add_slider_float(
                    label="Yaw",
                    tag=self.slider_yaw,
                    width=100,
                    min_value=-180.0,
                    max_value=180.0,
                    default_value=45.0,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Pitch",
                    tag=self.slider_pitch,
                    width=100,
                    min_value=-89.0,
                    max_value=89.0,
                    default_value=30.0,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Zoom",
                    tag=self.slider_zoom,
                    width=100,
                    min_value=0.05,
                    max_value=1.5,
                    default_value=0.3,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Pan X",
                    tag=self.slider_pan_x,
                    width=100,
                    min_value=-2.0,
                    max_value=2.0,
                    default_value=0.0,
                    callback=self._on_slider_change,
                )
                dpg.add_slider_float(
                    label="Pan Y",
                    tag=self.slider_pan_y,
                    width=100,
                    min_value=-2.0,
                    max_value=2.0,
                    default_value=0.0,
                    callback=self._on_slider_change,
                )

            # Per-Ply visibility checkbox row
            dpg.add_group(tag=self.layer_group, horizontal=True)

            # Setup dynamic texture registry for PyVista offscreen rendering
            if HAS_PYVISTA:
                texture_reg_tag = "viewport_texture_registry"
                if not dpg.does_item_exist(texture_reg_tag):
                    with dpg.texture_registry(tag=texture_reg_tag, show=False):
                        self._texture_tag = dpg.generate_uuid()
                        dpg.add_dynamic_texture(
                            width=self.width,
                            height=self.height,
                            default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
                            tag=self._texture_tag,
                        )
                        self._texture_w = self.width
                        self._texture_h = self.height

    def resize(self, width: int, height: int) -> None:
        """Resize the viewport drawing canvas and offscreen PyVista plotter dynamically."""
        with self.render_lock:
            self.width = width
            self.height = height
            self.center_x = width / 2.0
            self.center_y = height / 2.0

            if dpg is not None and dpg.does_item_exist(self.canvas_tag):
                dpg.configure_item(self.canvas_tag, width=width, height=height)

            if HAS_PYVISTA and self.has_pyvista:
                if self.plotter is None:
                    self._needs_plotter_resize = True
                    self._target_plotter_width = width
                    self._target_plotter_height = height
                else:
                    curr_w, curr_h = self.plotter.window_size
                    # Only request plotter recreation if dimensions differ by > 50 px
                    if abs(curr_w - width) > 50 or abs(curr_h - height) > 50:
                        self._needs_plotter_resize = True
                        self._target_plotter_width = width
                        self._target_plotter_height = height

            self.redraw()

    def _setup_mouse_handlers(self) -> None:
        """Register trackpad-friendly keyboard/mouse coordinate listeners."""
        if dpg is None:  # pragma: no cover
            return

        handler_reg = "viewport_mouse_handlers"
        if dpg.does_item_exist(handler_reg):
            dpg.delete_item(handler_reg)

        with dpg.handler_registry(tag=handler_reg):
            dpg.add_mouse_down_handler(button=-1, callback=self._on_mouse_down)
            dpg.add_mouse_drag_handler(button=-1, callback=self._on_mouse_drag)
            dpg.add_mouse_release_handler(button=-1, callback=self._on_mouse_release)

    def _on_mouse_release(self, sender: str, app_data: Any) -> None:
        """Track mouse release events to end dragging state."""
        self.is_dragging = False
        self.drag_button = None

    def _on_mouse_down(self, sender: str, app_data: Any) -> None:
        """Track mouse press start to manage orbit or panning translation coordinates."""
        if dpg is None:  # pragma: no cover
            return
        if self.is_dragging:
            return
        if dpg.is_item_hovered(self.canvas_tag):
            self.drag_start = dpg.get_mouse_pos(local=False)
            self.is_dragging = True
            button = (
                app_data[0]
                if isinstance(app_data, (list, tuple)) and len(app_data) > 0
                else app_data
            )
            self.drag_button = button

    def _on_mouse_drag(self, sender: str, app_data: Any) -> None:
        """Responsive mouse drag callback with macOS Trackpad keyboard modifiers."""
        if dpg is None or not self.is_dragging:  # pragma: no cover
            return

        mouse_pos = dpg.get_mouse_pos(local=False)
        dx = mouse_pos[0] - self.drag_start[0]
        dy = mouse_pos[1] - self.drag_start[1]
        self.drag_start = mouse_pos

        # Check Keyboard Modifiers for Trackpad Friendliness (Windows and macOS GLFW key codes)
        is_shift = dpg.is_key_down(16) or dpg.is_key_down(340) or dpg.is_key_down(344)
        is_option = dpg.is_key_down(18) or dpg.is_key_down(342) or dpg.is_key_down(346)
        is_control = dpg.is_key_down(17) or dpg.is_key_down(341) or dpg.is_key_down(345)
        is_command = dpg.is_key_down(343) or dpg.is_key_down(347)

        # Check Mouse Buttons (0=left, 1=right, 2=middle)
        is_right_drag = dpg.is_mouse_button_down(1)
        is_middle_drag = dpg.is_mouse_button_down(2)

        if is_shift or is_control or is_command or is_right_drag or is_middle_drag:
            # 1. Shift/Ctrl/Cmd/Right/Middle Drag = Pan camera
            self.pan_x += dx * 0.003 * self.distance
            self.pan_y -= dy * 0.003 * self.distance
            dpg.set_value(self.slider_pan_x, self.pan_x)
            dpg.set_value(self.slider_pan_y, self.pan_y)
        elif is_option:
            # 2. Option + Left-drag = Zoom camera
            self.distance = min(1.5, max(0.02, self.distance + dy * 0.002 * self.distance))
            dpg.set_value(self.slider_zoom, self.distance)
        else:
            # 3. Standard Left-drag = Orbit Yaw and Pitch
            self.yaw += dx * 0.008
            self.pitch = min(math.pi / 2 - 0.01, max(-math.pi / 2 + 0.01, self.pitch + dy * 0.008))

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
        elif sender == self.slider_pan_x:
            self.pan_x = app_data
        elif sender == self.slider_pan_y:
            self.pan_y = app_data

        self.redraw()

    def reset(
        self,
        grid: Grid,
        n_plies: int = 1,
        n_nodes_per_layer: int = 121,
        blade_width: float = 0.02,
        edge_thickness: float = 0.005,
        shape_type: str = "box",
        radius: float = 0.005,
        length: float = 0.01,
        edge_radius: float = 0.0,
        ogive_multiplier: float = 2.0,
        span: float = 0.05,
        root_chord: float = 0.01,
        tip_chord: float = 0.005,
        twist: float = 15.0,
        thickness_ratio: float = 12.0,
        tip_radius: float = 0.002,
        t_ply: float | None = None,
    ) -> None:
        """Store grid model coordinates and regenerate layer visibility checkboxes.

        Parameters
        ----------
        grid : Grid
            Simulated Lumped-Mass spring network.
        n_plies : int
            Ply layers.
        n_nodes_per_layer : int
            Discrete node counts in a single ply.
        blade_width : float
            Width of the projectile blade.
        edge_thickness : float
            Thickness of the projectile edge.
        shape_type : str
            Shape string mapping.
        radius : float
            Radius dimension.
        """
        with self.render_lock:
            if t_ply is None:
                if n_nodes_per_layer > 0:
                    n_plies = max(1, len(grid.nodes) // n_nodes_per_layer)
                else:
                    n_plies = 1
            
            self.grid = grid
            self.n_plies = n_plies
            self.n_nodes_per_layer = n_nodes_per_layer
            self.layer_visibility = [True] * n_plies
            
            # Cache projectile params
            self.proj_shape_type = shape_type
            self.proj_blade_width = blade_width
            self.proj_edge_thickness = edge_thickness
            self.proj_radius = radius
            self.proj_length = length
            self.proj_edge_radius = edge_radius
            self.proj_ogive_multiplier = ogive_multiplier
            self.proj_span = span
            self.proj_root_chord = root_chord
            self.proj_tip_chord = tip_chord
            self.proj_twist = twist
            self.proj_thickness_ratio = thickness_ratio
            self.proj_tip_radius = tip_radius
            
            if hasattr(self, "_last_mesh_params"):
                delattr(self, "_last_mesh_params")
            self.proj_mesh = None
            self.proj_actor = None

            # Find bounds of single grid center
            if len(grid.nodes) > 0:
                # Find center of nodes
                self.grid_center = np.mean(grid.nodes[:n_nodes_per_layer], axis=0)
            else:  # pragma: no cover
                self.grid_center = np.zeros(3)

            # Recenter camera panning
            self.pan_x = 0.0
            self.pan_y = 0.0

            # Try to initialize PyVista offscreen visualization plotter
            if HAS_PYVISTA and pv is not None:
                try:
                    # Clean up existing plotter to free GPU memory
                    if self.plotter is not None:
                        import contextlib

                        with contextlib.suppress(Exception):
                            self.plotter.close()

                    self.plotter = pv.Plotter(
                        off_screen=True, window_size=[self.width, self.height]
                    )
                    self.plotter.background_color = "black"  # type: ignore[assignment]

                    texture_reg_tag = "viewport_texture_registry"
                    if dpg is not None and dpg.does_item_exist(texture_reg_tag):
                        if not hasattr(self, "_texture_tag"):
                            self._texture_tag = dpg.generate_uuid()

                        texture_exists = dpg.does_item_exist(self._texture_tag)
                        texture_size_matches = (
                            getattr(self, "_texture_w", 0) == self.width
                            and getattr(self, "_texture_h", 0) == self.height
                        )

                        if not texture_exists or not texture_size_matches:
                            if texture_exists:
                                dpg.delete_item(self._texture_tag)

                            self._texture_tag = dpg.generate_uuid()
                            dpg.add_dynamic_texture(
                                width=self.width,
                                height=self.height,
                                default_value=np.zeros(
                                    self.width * self.height * 4, dtype=np.float32
                                ),
                                tag=self._texture_tag,
                                parent=texture_reg_tag,
                            )
                            self._texture_w = self.width
                            self._texture_h = self.height

                    # Pre-calculate VTK cell-connectivity connectivity line lists once
                    springs = grid.springs
                    n_springs = len(springs)
                    lines = np.empty(n_springs * 3, dtype=np.int32)
                    lines[0::3] = 2
                    lines[1::3] = springs[:, 0]
                    lines[2::3] = springs[:, 1]

                    self.mesh = pv.PolyData(grid.nodes, lines=lines)

                    # Initialize cell color array (RGBA uint8)
                    dummy_colors = np.zeros((n_springs, 4), dtype=np.uint8)
                    self.mesh.cell_data["colors"] = dummy_colors

                    # Add mesh to plotter
                    self.actor = self.plotter.add_mesh(
                        self.mesh,
                        scalars="colors",
                        rgba=True,
                        line_width=1.5,
                        show_scalar_bar=False,
                        lighting=False,
                    )

                    # Projectile mesh will be created by redraw() since _last_mesh_params was deleted
                    # Show to initialize offscreen rendering context window
                    self.plotter.show(auto_close=False, interactive=False, interactive_update=True)

                    self.has_pyvista = True
                except Exception as e:
                    logger.error(f"PyVista viewport reset failed: {e}", exc_info=True)
                    # Fall back to DearPyGui native draw_line loop
                    self.has_pyvista = False
            else:
                self.has_pyvista = False

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
        with self.render_lock:
            ply_idx = user_data
            if ply_idx < len(self.layer_visibility):
                self.layer_visibility[ply_idx] = app_data
            self.redraw()

    def redraw(self, force: bool = False) -> None:
        """Clear canvas drawlist and project coordinates to draw springs + projectile."""
        if not force:
            self._needs_redraw = True
            return

        with self.render_lock:
            if dpg is None or self.grid is None:  # pragma: no cover
                return

            if not dpg.does_item_exist(self.canvas_tag):  # pragma: no cover
                return

            # Deferred plotter resize if requested S7.10
            if HAS_PYVISTA and self.has_pyvista and getattr(self, "_needs_plotter_resize", False):
                try:
                    w = getattr(self, "_target_plotter_width", self.width)
                    h = getattr(self, "_target_plotter_height", self.height)
                    import contextlib

                    if self.plotter is not None:
                        with contextlib.suppress(Exception):
                            self.plotter.close()

                    self.plotter = pv.Plotter(off_screen=True, window_size=[w, h])
                    self.plotter.background_color = "black"

                    if self.mesh is not None:
                        self.actor = self.plotter.add_mesh(
                            self.mesh,
                            scalars="colors",
                            rgba=True,
                            line_width=1.5,
                            show_scalar_bar=False,
                            lighting=False,
                        )
                    if hasattr(self, "proj_mesh") and self.proj_mesh is not None:
                        self.proj_actor = self.plotter.add_mesh(
                            self.proj_mesh,
                            color=[230, 230, 250],
                            style="wireframe",
                            line_width=2.5,
                            lighting=False,
                        )
                    self.plotter.show(auto_close=False, interactive=False, interactive_update=True)

                    texture_reg_tag = "viewport_texture_registry"
                    if dpg.does_item_exist(texture_reg_tag):
                        if hasattr(self, "_texture_tag") and dpg.does_item_exist(self._texture_tag):
                            dpg.delete_item(self._texture_tag)

                        self._texture_tag = dpg.generate_uuid()
                        dpg.add_dynamic_texture(
                            width=w,
                            height=h,
                            default_value=np.zeros(w * h * 4, dtype=np.float32),
                            tag=self._texture_tag,
                            parent=texture_reg_tag,
                        )
                        self._texture_w = w
                        self._texture_h = h
                    self._needs_plotter_resize = False
                except Exception as e:
                    logger.error(f"PyVista deferred plotter resize failed: {e}", exc_info=True)
                    self.has_pyvista = False

            # 1. Calculate camera rotations and screen translations
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            cp, sp = math.cos(self.pitch), math.sin(self.pitch)

            # 3D Yaw-Pitch camera projection coordinates rotation matrix
            R = np.array([[cy, 0.0, -sy], [-sy * sp, cp, -cy * sp], [sy * cp, sp, cy * cp]])  # noqa: N806

            springs = self.grid.springs
            failed = self.grid.failed
            n_springs = len(springs)

            # Calculate live engineering strain for color-scale mapping
            p1 = self.grid.nodes[springs[:, 0]]
            p2 = self.grid.nodes[springs[:, 1]]
            lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
            strains = (lengths - self.grid.rest_lengths) / self.grid.rest_lengths
            fail_thresh = 0.036

            # --- PyVista offscreen hardware rendering path ---
            if (
                HAS_PYVISTA
                and self.has_pyvista
                and self.plotter is not None
                and self.mesh is not None
            ):
                try:
                    # Update point coordinate positions in-place
                    self.mesh.points = self.grid.nodes

                    # Compute RGBA cell colors vectorised
                    colors = np.zeros((n_springs, 4), dtype=np.uint8)

                    # Failed springs: Deep brick red transparent lines [139, 0, 0, 45]
                    colors[failed] = [139, 0, 0, 45]

                    # Active springs: interpolate Blue -> Yellow -> Crimson
                    active = ~failed
                    ratios = np.clip(strains[active] / fail_thresh, 0.0, 1.0)

                    mask1 = ratios <= 0.5
                    ratio1 = ratios[mask1] / 0.5

                    mask2 = ratios > 0.5
                    ratio2 = (ratios[mask2] - 0.5) / 0.5

                    c_active = np.zeros((np.sum(active), 4), dtype=np.uint8)

                    # Write mask1 colors (Blue -> Yellow)
                    c_active[mask1, 0] = (0 + 255 * ratio1).astype(np.uint8)
                    c_active[mask1, 1] = (191 + (215 - 191) * ratio1).astype(np.uint8)
                    c_active[mask1, 2] = (255 - 255 * ratio1).astype(np.uint8)
                    c_active[mask1, 3] = 230

                    # Write mask2 colors (Yellow -> Crimson)
                    c_active[mask2, 0] = (255 - (255 - 220) * ratio2).astype(np.uint8)
                    c_active[mask2, 1] = (215 - (215 - 20) * ratio2).astype(np.uint8)
                    c_active[mask2, 2] = (0 + 60 * ratio2).astype(np.uint8)
                    c_active[mask2, 3] = 230

                    colors[active] = c_active

                    # Set alpha to 0 for springs in hidden layers
                    ply_indices = springs[:, 0] // self.n_nodes_per_layer
                    for ply_idx, visible in enumerate(self.layer_visibility):
                        if not visible:
                            colors[ply_indices == ply_idx, 3] = 0

                    self.mesh.cell_data["colors"] = colors

                    # Check if shape parameters changed, rebuild the mesh
                    mesh_params = (
                        self.proj_shape_type,
                        self.proj_radius,
                        self.proj_length,
                        self.proj_edge_radius,
                        self.proj_ogive_multiplier,
                        self.proj_span,
                        self.proj_root_chord,
                        self.proj_tip_chord,
                        self.proj_twist,
                        self.proj_thickness_ratio,
                        self.proj_tip_radius,
                        self.proj_blade_width,
                        self.proj_edge_thickness,
                    )
                    if (not hasattr(self, "_last_mesh_params") or
                        self._last_mesh_params != mesh_params or
                        not hasattr(self, "proj_mesh") or
                        self.proj_mesh is None):
                        
                        self._last_mesh_params = mesh_params
                        shape = self.proj_shape_type
                        if shape == "sphere":
                            self.proj_mesh = pv.Sphere(radius=self.proj_radius, theta_resolution=16, phi_resolution=16)
                        elif shape == "cylinder":
                            self.proj_mesh = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=self.proj_radius, height=self.proj_length, resolution=16)
                        elif shape == "bullet":
                            # Compute z_com
                            R0 = self.proj_radius
                            R_og = R0 * self.proj_ogive_multiplier
                            L_nose = math.sqrt(max(0.0, 2.0 * R_og * R0 - R0 ** 2))
                            L_body = max(0.0, self.proj_length - L_nose)
                            N = 100
                            zs = np.linspace(-L_body, L_nose, N)
                            dz = (L_body + L_nose) / N
                            dV_sum = 0.0
                            z_dV_sum = 0.0
                            for z in zs:
                                r = R0 if z < 0 else R0 - R_og + math.sqrt(max(0.0, R_og**2 - z**2))
                                dV = math.pi * (r ** 2) * dz
                                dV_sum += dV
                                z_dV_sum += z * dV
                            z_com = z_dV_sum / dV_sum if dV_sum > 0 else 0.0
                            self.proj_mesh = _make_bullet_mesh(self.proj_radius, self.proj_length, self.proj_ogive_multiplier, z_com)
                        elif shape == "propeller":
                            S = self.proj_span
                            c_r = self.proj_root_chord
                            c_t = self.proj_tip_chord
                            tau = self.proj_thickness_ratio / 100.0
                            N = 100
                            ys = np.linspace(0.0, S, N)
                            dy = S / N
                            dV_sum = 0.0
                            y_dV_sum = 0.0
                            for y in ys:
                                c = c_r + (y / S) * (c_t - c_r)
                                area = 0.60 * (c ** 2) * tau
                                dV = area * dy
                                dV_sum += dV
                                y_dV_sum += y * dV
                            y_com = y_dV_sum / dV_sum if dV_sum > 0 else 0.0
                            self.proj_mesh = _make_propeller_mesh(self.proj_span, self.proj_root_chord, self.proj_tip_chord, self.proj_twist, self.proj_thickness_ratio, self.proj_tip_radius, y_com)
                        else:  # box
                            w_h = self.proj_blade_width / 2.0
                            t_h = self.proj_edge_thickness / 2.0
                            h_h = 0.005
                            self.proj_mesh = pv.Box(bounds=[-w_h, w_h, -t_h, t_h, -h_h, h_h])
                        
                        if self.proj_actor is not None:
                            try:
                                self.plotter.remove_actor(self.proj_actor)
                            except Exception:
                                pass
                        self.proj_actor = self.plotter.add_mesh(
                            self.proj_mesh,
                            color=[230, 230, 250],
                            style="wireframe",
                            line_width=2.5,
                            lighting=False,
                        )

                    # Update projectile position and orientation in actor using 4x4 matrix
                    self.proj_actor.position = (0.0, 0.0, 0.0)
                    w, x, y, z = self.proj_quat
                    r00 = 1.0 - 2.0 * (y*y + z*z)
                    r01 = 2.0 * (x*y - w*z)
                    r02 = 2.0 * (x*z + w*y)
                    
                    r10 = 2.0 * (x*y + w*z)
                    r11 = 1.0 - 2.0 * (x*x + z*z)
                    r12 = 2.0 * (y*z - w*x)
                    
                    r20 = 2.0 * (x*z - w*y)
                    r21 = 2.0 * (y*z + w*x)
                    r22 = 1.0 - 2.0 * (x*x + y*y)
                    
                    T = np.array([
                        [r00, r01, r02, self.proj_position[0]],
                        [r10, r11, r12, self.proj_position[1]],
                        [r20, r21, r22, self.proj_position[2]],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
                    self.proj_actor.user_matrix = T

                    # Set camera look-at vectors using rotation matrix R
                    local_x = R[0, :]
                    local_y = R[1, :]
                    local_z = R[2, :]

                    focal_point = self.grid_center - self.pan_x * local_x - self.pan_y * local_y
                    camera_position = focal_point + self.distance * local_z

                    self.plotter.camera.position = camera_position
                    self.plotter.camera.focal_point = focal_point
                    self.plotter.camera.up = local_y

                    # Offscreen render
                    self.plotter.render()

                    # Read buffer image
                    img = self.plotter.image
                    img_h, img_w = img.shape[0], img.shape[1]
                    texture_reg_tag = "viewport_texture_registry"
                    if not hasattr(self, "_texture_tag"):
                        self._texture_tag = dpg.generate_uuid()

                    if (
                        getattr(self, "_texture_w", 0) != img_w
                        or getattr(self, "_texture_h", 0) != img_h
                        or not dpg.does_item_exist(self._texture_tag)
                    ):
                        if dpg.does_item_exist(texture_reg_tag):
                            if dpg.does_item_exist(self._texture_tag):
                                dpg.delete_item(self._texture_tag)

                            self._texture_tag = dpg.generate_uuid()
                            dpg.add_dynamic_texture(
                                width=img_w,
                                height=img_h,
                                default_value=np.zeros(img_w * img_h * 4, dtype=np.float32),
                                tag=self._texture_tag,
                                parent=texture_reg_tag,
                            )
                            self._texture_w = img_w
                            self._texture_h = img_h

                    if img.shape[2] == 3:
                        rgba = np.empty((img_h, img_w, 4), dtype=np.float32)
                        rgba[:, :, :3] = img.astype(np.float32) / 255.0
                        rgba[:, :, 3] = 1.0
                    else:
                        rgba = img.astype(np.float32) / 255.0

                    # Update DearPyGui dynamic texture
                    dpg.set_value(self._texture_tag, rgba.ravel())

                    # Draw the dynamic texture image onto DPG canvas
                    dpg.delete_item(self.canvas_tag, children_only=True)
                    dpg.draw_image(
                        texture_tag=self._texture_tag,
                        pmin=[0, 0],
                        pmax=[self.width, self.height],
                        parent=self.canvas_tag,
                    )
                    return
                except Exception as e:
                    logger.error(f"PyVista viewport redraw failed: {e}", exc_info=True)

            # --- Native DearPyGui Fallback Loop ---
            dpg.delete_item(self.canvas_tag, children_only=True)

            # Rotate coordinates
            nodes_rel = self.grid.nodes - self.grid_center
            nodes_rotated = nodes_rel @ R.T

            # Camera translations including horizontal/vertical pans
            cam_x = nodes_rotated[:, 0] + self.pan_x
            cam_y = nodes_rotated[:, 1] + self.pan_y
            cam_z = np.maximum(nodes_rotated[:, 2] + self.distance, 1e-4)

            # Project coordinates onto viewport canvas
            scr_x = self.center_x + (self.focal_length * cam_x / cam_z)
            scr_y = self.center_y - (self.focal_length * cam_y / cam_z)

            for j in range(n_springs):
                u, v = springs[j, 0], springs[j, 1]

                # Map spring to ply layer index
                ply_idx = u // self.n_nodes_per_layer
                if ply_idx < len(self.layer_visibility) and not self.layer_visibility[ply_idx]:
                    continue  # Skip drawing hidden layers

                x_u, y_u = float(scr_x[u]), float(scr_y[u])
                x_v, y_v = float(scr_x[v]), float(scr_y[v])

                # Filter off-screen clippings
                if (x_u < 0 or x_u > self.width or y_u < 0 or y_u > self.height) and (
                    x_v < 0 or x_v > self.width or y_v < 0 or y_v > self.height
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
                        r = int(0 + (255 - 0) * (ratio / 0.5))
                        g = int(191 + (215 - 191) * (ratio / 0.5))
                        b = int(255 + (0 - 255) * (ratio / 0.5))
                    else:
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

            # Draw projectile wireframe in DPG fallback path
            try:
                lines = self._get_shape_wireframe_lines()
                for pt1_loc, pt2_loc in lines:
                    pt1_rot = q_rotate_vector(self.proj_quat, pt1_loc)
                    pt2_rot = q_rotate_vector(self.proj_quat, pt2_loc)
                    
                    pt1_world = pt1_rot + self.proj_position
                    pt2_world = pt2_rot + self.proj_position
                    
                    pt1_rel = pt1_world - self.grid_center
                    pt2_rel = pt2_world - self.grid_center
                    
                    pt1_cam = pt1_rel @ R.T
                    pt2_cam = pt2_rel @ R.T
                    
                    cam1_z = max(pt1_cam[2] + self.distance, 1e-4)
                    cam2_z = max(pt2_cam[2] + self.distance, 1e-4)
                    
                    scr1_x = self.center_x + (self.focal_length * pt1_cam[0] / cam1_z)
                    scr1_y = self.center_y - (self.focal_length * pt1_cam[1] / cam1_z)
                    
                    scr2_x = self.center_x + (self.focal_length * pt2_cam[0] / cam2_z)
                    scr2_y = self.center_y - (self.focal_length * pt2_cam[1] / cam2_z)
                    
                    dpg.draw_line(
                        [float(scr1_x), float(scr1_y)],
                        [float(scr2_x), float(scr2_y)],
                        color=[230, 230, 250, 255],
                        thickness=2.5,
                        parent=self.canvas_tag,
                    )
            except Exception as e:
                logger.error(f"Fallback projectile draw failed: {e}")

    def _get_shape_wireframe_lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        shape = self.proj_shape_type.lower()
        lines = []
        
        if shape == "sphere":
            R = self.proj_radius
            N = 16
            # XY ring
            pts_xy = [np.array([R * math.cos(2*math.pi*i/N), R * math.sin(2*math.pi*i/N), 0.0]) for i in range(N)]
            # YZ ring
            pts_yz = [np.array([0.0, R * math.cos(2*math.pi*i/N), R * math.sin(2*math.pi*i/N)]) for i in range(N)]
            # ZX ring
            pts_zx = [np.array([R * math.sin(2*math.pi*i/N), 0.0, R * math.cos(2*math.pi*i/N)]) for i in range(N)]
            
            for j in range(N):
                lines.append((pts_xy[j], pts_xy[(j+1)%N]))
                lines.append((pts_yz[j], pts_yz[(j+1)%N]))
                lines.append((pts_zx[j], pts_zx[(j+1)%N]))
                
        elif shape == "cylinder":
            R = self.proj_radius
            L = self.proj_length
            N = 16
            # Bottom cap at z = -L/2
            pts_bot = [np.array([R * math.cos(2*math.pi*i/N), R * math.sin(2*math.pi*i/N), -L/2.0]) for i in range(N)]
            # Top cap at z = L/2
            pts_top = [np.array([R * math.cos(2*math.pi*i/N), R * math.sin(2*math.pi*i/N), L/2.0]) for i in range(N)]
            
            for j in range(N):
                lines.append((pts_bot[j], pts_bot[(j+1)%N]))
                lines.append((pts_top[j], pts_top[(j+1)%N]))
                
            # 4 axial lines
            for idx in [0, N//4, N//2, (3*N)//4]:
                lines.append((pts_bot[idx], pts_top[idx]))
                
        elif shape == "bullet":
            R0 = self.proj_radius
            R_og = R0 * self.proj_ogive_multiplier
            L_nose = math.sqrt(max(0.0, 2.0 * R_og * R0 - R0 ** 2))
            L_body = max(0.0, self.proj_length - L_nose)
            
            # Compute z_com
            N_steps = 100
            zs_com = np.linspace(-L_body, L_nose, N_steps)
            dz = (L_body + L_nose) / N_steps
            dV_sum = 0.0
            z_dV_sum = 0.0
            for z_val in zs_com:
                r = R0 if z_val < 0.0 else R0 - R_og + math.sqrt(max(0.0, R_og**2 - z_val**2))
                dV = math.pi * (r ** 2) * dz
                dV_sum += dV
                z_dV_sum += z_val * dV
            z_com = z_dV_sum / dV_sum if dV_sum > 0 else 0.0
            
            N = 16
            z_base = -L_body - z_com
            z_mid = -z_com
            z_tip = L_nose - z_com
            
            # Bottom cap
            pts_bot = [np.array([R0 * math.cos(2*math.pi*i/N), R0 * math.sin(2*math.pi*i/N), z_base]) for i in range(N)]
            # Mid cap (interface)
            pts_mid = [np.array([R0 * math.cos(2*math.pi*i/N), R0 * math.sin(2*math.pi*i/N), z_mid]) for i in range(N)]
            
            for j in range(N):
                lines.append((pts_bot[j], pts_bot[(j+1)%N]))
                lines.append((pts_mid[j], pts_mid[(j+1)%N]))
                
            # 4 body lines
            quad_indices = [0, N//4, N//2, (3*N)//4]
            for idx in quad_indices:
                lines.append((pts_bot[idx], pts_mid[idx]))
                
            # Ogive profiles from mid cap to tip
            for idx in quad_indices:
                angle = 2 * math.pi * idx / N
                c = math.cos(angle)
                s = math.sin(angle)
                prev_pt = pts_mid[idx]
                n_profile = 5
                for step in range(1, n_profile + 1):
                    z_geom = (step / n_profile) * L_nose
                    z_loc = z_geom - z_com
                    r_val = R0 - R_og + math.sqrt(max(0.0, R_og**2 - z_geom**2)) if step < n_profile else 0.0
                    curr_pt = np.array([r_val * c, r_val * s, z_loc])
                    lines.append((prev_pt, curr_pt))
                    prev_pt = curr_pt
                    
        elif shape == "propeller":
            S = self.proj_span
            c_r = self.proj_root_chord
            c_t = self.proj_tip_chord
            tau = self.proj_thickness_ratio / 100.0
            
            # Compute y_com
            N_steps = 100
            ys_com = np.linspace(0.0, S, N_steps)
            dy = S / N_steps
            dV_sum = 0.0
            y_dV_sum = 0.0
            for y_val in ys_com:
                c = c_r + (y_val / S) * (c_t - c_r)
                area = 0.60 * (c ** 2) * tau
                dV = area * dy
                dV_sum += dV
                y_dV_sum += y_val * dV
            y_com = y_dV_sum / dV_sum if dV_sum > 0 else 0.0
            
            N_stations = 10
            chord_pts_le = []
            chord_pts_te = []
            
            for k in range(N_stations):
                y_geom = (k / (N_stations - 1)) * S
                y_loc = y_geom - y_com
                c = c_r + (y_geom / S) * (c_t - c_r)
                theta = math.radians(self.proj_twist) * (y_geom / S)
                
                le_loc = np.array([-c/2.0 * math.cos(theta), y_loc, c/2.0 * math.sin(theta)])
                te_loc = np.array([c/2.0 * math.cos(theta), y_loc, -c/2.0 * math.sin(theta)])
                
                chord_pts_le.append(le_loc)
                chord_pts_te.append(te_loc)
                
                lines.append((le_loc, te_loc))
                
            for k in range(N_stations - 1):
                lines.append((chord_pts_le[k], chord_pts_le[k+1]))
                lines.append((chord_pts_te[k], chord_pts_te[k+1]))
                
        else:  # box
            w_h = self.proj_blade_width / 2.0
            t_h = self.proj_edge_thickness / 2.0
            h_h = 0.005
            corners = [
                np.array([-w_h, -t_h, -h_h]),
                np.array([w_h, -t_h, -h_h]),
                np.array([w_h, t_h, -h_h]),
                np.array([-w_h, t_h, -h_h]),
                np.array([-w_h, -t_h, h_h]),
                np.array([w_h, -t_h, h_h]),
                np.array([w_h, t_h, h_h]),
                np.array([-w_h, t_h, h_h]),
            ]
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for u, v in edges:
                lines.append((corners[u], corners[v]))
                
        return lines

    def _get_local_bounds(self) -> tuple[float, float, float, float, float, float]:
        shape = self.proj_shape_type
        if shape == "sphere":
            R = self.proj_radius
            return -R, R, -R, R, -R, R
        elif shape == "cylinder":
            R = self.proj_radius
            L = self.proj_length
            return -R, R, -R, R, -L/2.0, L/2.0
        elif shape == "bullet":
            R0 = self.proj_radius
            R_og = R0 * self.proj_ogive_multiplier
            L_nose = math.sqrt(max(0.0, 2.0 * R_og * R0 - R0 ** 2))
            L_body = max(0.0, self.proj_length - L_nose)
            # Compute z_com
            N = 100
            zs = np.linspace(-L_body, L_nose, N)
            dz = (L_body + L_nose) / N
            dV_sum = 0.0
            z_dV_sum = 0.0
            for z in zs:
                r = R0 if z < 0 else R0 - R_og + math.sqrt(max(0.0, R_og**2 - z**2))
                dV = math.pi * (r ** 2) * dz
                dV_sum += dV
                z_dV_sum += z * dV
            z_com = z_dV_sum / dV_sum if dV_sum > 0 else 0.0
            return -R0, R0, -R0, R0, -L_body - z_com, L_nose - z_com
        elif shape == "propeller":
            S = self.proj_span
            c_r = self.proj_root_chord
            c_t = self.proj_tip_chord
            tau = self.proj_thickness_ratio / 100.0
            # Compute y_com
            N = 100
            ys = np.linspace(0.0, S, N)
            dy = S / N
            dV_sum = 0.0
            y_dV_sum = 0.0
            for y in ys:
                c = c_r + (y / S) * (c_t - c_r)
                area = 0.60 * (c ** 2) * tau
                dV = area * dy
                dV_sum += dV
                y_dV_sum += y * dV
            y_com = y_dV_sum / dV_sum if dV_sum > 0 else 0.0
            t_max = c_r * tau
            return -c_r/2.0, c_r/2.0, -y_com, S - y_com, -t_max/2.0, t_max/2.0
        else: # box
            w_h = self.proj_blade_width / 2.0
            t_h = self.proj_edge_thickness / 2.0
            h_h = 0.005
            return -w_h, w_h, -t_h, t_h, -h_h, h_h

    def update(self, positions: np.ndarray, failed: np.ndarray) -> None:
        """Update node coordinate positions dynamically."""
        with self.render_lock:
            if self.grid is not None:
                self.grid.nodes = np.asarray(positions)
                self.grid.failed = np.asarray(failed)
                # We do NOT call self.redraw() here; it will be called by draw_projectile()
                # to render the complete synchronized frame containing the projectile.

    def draw_projectile(
        self,
        position: np.ndarray,
        blade_width: float,
        edge_thickness: float,
        quat: np.ndarray | None = None,
        shape_type: str = "box",
        radius: float = 0.005,
        length: float = 0.01,
        edge_radius: float = 0.0,
        ogive_multiplier: float = 2.0,
        span: float = 0.05,
        root_chord: float = 0.01,
        tip_chord: float = 0.005,
        twist: float = 15.0,
        thickness_ratio: float = 12.0,
        tip_radius: float = 0.002,
    ) -> None:
        """Render the striking projectile, caching details and triggering a redraw.

        Parameters
        ----------
        position : np.ndarray
            3-D coordinate of projectile's center, shape (3,).
        blade_width : float
            Width along X axis (meters).
        edge_thickness : float
            Thickness along Y axis (meters).
        """
        with self.render_lock:
            self.proj_position = np.asarray(position)
            self.proj_blade_width = blade_width
            self.proj_edge_thickness = edge_thickness
            self.proj_quat = np.asarray(quat) if quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
            self.proj_shape_type = shape_type
            self.proj_radius = radius
            self.proj_length = length
            self.proj_edge_radius = edge_radius
            self.proj_ogive_multiplier = ogive_multiplier
            self.proj_span = span
            self.proj_root_chord = root_chord
            self.proj_tip_chord = tip_chord
            self.proj_twist = twist
            self.proj_thickness_ratio = thickness_ratio
            self.proj_tip_radius = tip_radius

            if dpg is None or self.grid is None:  # pragma: no cover
                return

            # Redraw handles projectile actor rendering internally
            self.redraw()


def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[0], v[1], v[2]
    iw = -x*vx - y*vy - z*vz
    ix =  w*vx + y*vz - z*vy
    iy =  w*vy - x*vz + z*vx
    iz =  w*vz + x*vy - y*vx
    rx = ix*w + iw*(-x) + iy*(-z) - iz*(-y)
    ry = iy*w + iw*(-y) + iz*(-x) - ix*(-z)
    rz = iz*w + iw*(-z) + ix*(-y) - iy*(-x)
    return np.array([rx, ry, rz])


def _make_bullet_mesh(radius: float, length: float, ogive_multiplier: float, z_com: float) -> pv.PolyData:
    if pv is None:
        return None
    R0 = radius
    R_og = R0 * ogive_multiplier
    L_nose = math.sqrt(max(0.0, 2.0 * R_og * R0 - R0 ** 2))
    L_body = max(0.0, length - L_nose)
    
    nz = 30
    n_theta = 20
    zs = np.linspace(-L_body, L_nose, nz)
    rs = np.zeros(nz)
    for i, z in enumerate(zs):
        if z < 0:
            rs[i] = R0
        else:
            rs[i] = R0 - R_og + math.sqrt(max(0.0, R_og**2 - z**2))
    
    points = []
    points.append([0.0, 0.0, -L_body - z_com])
    
    for z, r in zip(zs, rs):
        for theta_idx in range(n_theta):
            theta = 2.0 * math.pi * theta_idx / n_theta
            points.append([r * math.cos(theta), r * math.sin(theta), z - z_com])
            
    points.append([0.0, 0.0, L_nose - z_com])
    points = np.array(points, dtype=np.float32)
    
    faces = []
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        faces.extend([3, 0, 1 + j_next, 1 + j])
        
    for i in range(nz - 1):
        r1_start = 1 + i * n_theta
        r2_start = 1 + (i + 1) * n_theta
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            faces.extend([4, r1_start + j, r1_start + j_next, r2_start + j_next, r2_start + j])
            
    last_ring_start = 1 + (nz - 1) * n_theta
    top_center_idx = len(points) - 1
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        faces.extend([3, top_center_idx, last_ring_start + j, last_ring_start + j_next])
        
    return pv.PolyData(points, faces=np.array(faces, dtype=np.int32))


def _make_propeller_mesh(span: float, root_chord: float, tip_chord: float, twist: float, thickness_ratio: float, tip_radius: float, y_com: float) -> pv.PolyData:
    if pv is None:
        return None
    ny = 25
    nx = 15
    S = span
    c_r = root_chord
    c_t = tip_chord
    twist_deg = twist
    tau = thickness_ratio / 100.0
    R_tip = tip_radius
    
    points = []
    points.append([0.0, -y_com, 0.0])
    
    ys_geom = np.linspace(0.0, S - R_tip, ny)
    for y_geom in ys_geom:
        c = c_r + (y_geom / S) * (c_t - c_r)
        theta = math.radians(twist_deg) * (y_geom / S)
        
        slice_pts = []
        us = np.linspace(0.0, 1.0, nx)
        for u in us:
            t = 5.0 * tau * (
                0.2969 * math.sqrt(u)
                - 0.1260 * u
                - 0.3516 * (u**2)
                + 0.2843 * (u**3)
                - 0.1015 * (u**4)
            ) * c
            z_val = max(t / 2.0, R_tip)
            x_val = (u - 0.5) * c
            
            xr = x_val * math.cos(theta) - z_val * math.sin(theta)
            zr = x_val * math.sin(theta) + z_val * math.cos(theta)
            slice_pts.append([xr, y_geom - y_com, zr])
            
        for u in reversed(us[1:-1]):
            t = 5.0 * tau * (
                0.2969 * math.sqrt(u)
                - 0.1260 * u
                - 0.3516 * (u**2)
                + 0.2843 * (u**3)
                - 0.1015 * (u**4)
            ) * c
            z_val = -max(t / 2.0, R_tip)
            x_val = (u - 0.5) * c
            
            xr = x_val * math.cos(theta) - z_val * math.sin(theta)
            zr = x_val * math.sin(theta) + z_val * math.cos(theta)
            slice_pts.append([xr, y_geom - y_com, zr])
            
        points.extend(slice_pts)
        
    points.append([0.0, S - y_com, 0.0])
    points = np.array(points, dtype=np.float32)
    
    faces = []
    n_per_slice = 2 * nx - 2
    
    for j in range(n_per_slice):
        j_next = (j + 1) % n_per_slice
        faces.extend([3, 0, 1 + j, 1 + j_next])
        
    for i in range(ny - 1):
        s1 = 1 + i * n_per_slice
        s2 = 1 + (i + 1) * n_per_slice
        for j in range(n_per_slice):
            j_next = (j + 1) % n_per_slice
            faces.extend([4, s1 + j, s2 + j, s2 + j_next, s1 + j_next])
            
    last_slice_start = 1 + (ny - 1) * n_per_slice
    tip_center_idx = len(points) - 1
    for j in range(n_per_slice):
        j_next = (j + 1) % n_per_slice
        faces.extend([3, tip_center_idx, last_slice_start + j_next, last_slice_start + j])
        
    return pv.PolyData(points, faces=np.array(faces, dtype=np.int32))

