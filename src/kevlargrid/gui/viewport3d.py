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
            button = app_data[0] if isinstance(app_data, (list, tuple)) and len(app_data) > 0 else app_data
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
        """
        with self.render_lock:
            self.grid = grid
            self.n_plies = n_plies
            self.n_nodes_per_layer = n_nodes_per_layer
            self.layer_visibility = [True] * n_plies
            self.proj_blade_width = blade_width
            self.proj_edge_thickness = edge_thickness

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

                    self.plotter = pv.Plotter(off_screen=True, window_size=[self.width, self.height])
                    self.plotter.background_color = "black"  # type: ignore[assignment]

                    texture_reg_tag = "viewport_texture_registry"
                    if dpg is not None and dpg.does_item_exist(texture_reg_tag):
                        if not hasattr(self, "_texture_tag"):
                            self._texture_tag = dpg.generate_uuid()
                        
                        texture_exists = dpg.does_item_exist(self._texture_tag)
                        texture_size_matches = getattr(self, "_texture_w", 0) == self.width and getattr(self, "_texture_h", 0) == self.height
                        
                        if not texture_exists or not texture_size_matches:
                            if texture_exists:
                                dpg.delete_item(self._texture_tag)
                            
                            self._texture_tag = dpg.generate_uuid()
                            dpg.add_dynamic_texture(
                                width=self.width,
                                height=self.height,
                                default_value=np.zeros(self.width * self.height * 4, dtype=np.float32),
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

                    # Add wireframe Box representing the projectile
                    w_h = self.proj_blade_width / 2.0
                    t_h = self.proj_edge_thickness / 2.0
                    h_h = 0.005
                    self.proj_mesh = pv.Box(bounds=[-w_h, w_h, -t_h, t_h, -h_h, h_h])
                    self.proj_actor = self.plotter.add_mesh(
                        self.proj_mesh,
                        color=[230, 230, 250],
                        style="wireframe",
                        line_width=2.5,
                        lighting=False,
                    )
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
            if HAS_PYVISTA and self.has_pyvista and self.plotter is not None and self.mesh is not None:
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

                    # Update projectile position in actor
                    self.proj_actor.position = self.proj_position

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
                        
                    if getattr(self, "_texture_w", 0) != img_w or getattr(self, "_texture_h", 0) != img_h or not dpg.does_item_exist(self._texture_tag):
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

    def update(self, positions: np.ndarray, failed: np.ndarray) -> None:
        """Update node coordinate positions dynamically."""
        with self.render_lock:
            if self.grid is not None:
                self.grid.nodes = np.asarray(positions)
                self.grid.failed = np.asarray(failed)
                # We do NOT call self.redraw() here; it will be called by draw_projectile()
                # to render the complete synchronized frame containing the projectile.

    def draw_projectile(
        self, position: np.ndarray, blade_width: float, edge_thickness: float
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

            if dpg is None or self.grid is None:  # pragma: no cover
                return

            # If PyVista is active, redraw handles projectile actor rendering internally
            if HAS_PYVISTA and self.has_pyvista:
                self.redraw()
                return

            # --- Native DearPyGui Projectile Fallback ---
            self.redraw()

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
            c_cam_z = np.maximum(c_rot[:, 2] + self.distance, 1e-4)

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
