"""Headless off-screen 3D mesh animation compiler.

Renders mass-spring trajectories into high-fidelity slowed-down MP4 or GIF animations
representing the impact event, utilizing standard matplotlib Agg backend.
"""

from __future__ import annotations

import math
import os
from typing import Any

# Force headless matplotlib rendering
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    pv = None
    HAS_PYVISTA = False


class VideoExporter:
    """Headless 3D line-wireframe video compiler using matplotlib.animation."""

    def __init__(
        self,
        config: dict[str, Any],
        history: list[dict[str, Any]],
        nx: int,
        ny: int,
        n_plies: int = 1,
        n_nodes_per_layer: int = 121,
    ) -> None:
        self.config = config
        self.history = history
        self.nx = nx
        self.ny = ny
        self.n_plies = n_plies
        self.n_nodes_per_layer = n_nodes_per_layer

    def compile(
        self,
        filepath: str,
        yaw: float = 45.0,
        pitch: float = 30.0,
        fps: int = 30,
        dpi: int = 100,
        distance: float | None = None,
        pan_x: float | None = None,
        pan_y: float | None = None,
    ) -> None:
        """Render trajectory history frames and compile an MP4 or GIF file.

        Parameters
        ----------
        filepath : str
            Output file path (ends with .mp4 or .gif).
        yaw : float
            Camera azimuthal rotation angle in degrees.
        pitch : float
            Camera elevation angle in degrees.
        fps : int
            Frames per second of the exported video.
        dpi : int
            DPI resolution scale of the video canvas.
        distance : float, optional
            Camera zoom/distance.
        pan_x : float, optional
            Horizontal camera panning translation.
        pan_y : float, optional
            Vertical camera panning translation.
        """
        if not self.history:
            raise ValueError("No history frames loaded to export video.")

        # Detect the actual number of renderable plies from the history data.
        actual_n_nodes = len(self.history[0]["nodes"])
        render_plies = max(1, actual_n_nodes // self.n_nodes_per_layer)

        # Build spring indexes to draw lines
        springs_list = []
        for ply in range(render_plies):
            offset = ply * self.n_nodes_per_layer
            for i in range(self.nx):
                for j in range(self.ny):
                    idx = offset + i * self.ny + j
                    # Orthogonal: warp (x)
                    if i < self.nx - 1:
                        springs_list.append((idx, idx + self.ny))
                    # Orthogonal: weft (y)
                    if j < self.ny - 1:
                        springs_list.append((idx, idx + 1))
                    # Diagonal: +45 deg
                    if i < self.nx - 1 and j < self.ny - 1:
                        springs_list.append((idx, idx + self.ny + 1))
                    # Diagonal: -45 deg
                    if i < self.nx - 1 and j > 0:
                        springs_list.append((idx, idx + self.ny - 1))

        springs = np.array(springs_list, dtype=np.int32)

        # Compute exact rest lengths from the initial reference coordinates in frame 0
        init_nodes = self.history[0]["nodes"]
        p1 = init_nodes[springs[:, 0]]
        p2 = init_nodes[springs[:, 1]]
        rest_lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
        # Guard against zero rest lengths to prevent division by zero in unphysical/dummy test meshes
        rest_lengths = np.where(rest_lengths == 0.0, 1.0, rest_lengths)

        # Determine spatial bounds across all history for tight viewport box (used by both paths)
        all_nodes = np.vstack([f["nodes"] for f in self.history])
        x_min, x_max = np.min(all_nodes[:, 0]), np.max(all_nodes[:, 0])
        y_min, y_max = np.min(all_nodes[:, 1]), np.max(all_nodes[:, 1])
        z_min, z_max = np.min(all_nodes[:, 2]) - 0.01, np.max(all_nodes[:, 2]) + 0.01

        fail_thresh = 0.036

        # Try PyVista Off-screen GPU Renderer path first
        use_pyvista = False
        if HAS_PYVISTA and pv is not None:
            try:
                plotter = pv.Plotter(off_screen=True, window_size=[700, 500])
                plotter.background_color = "black"

                # Build PolyData mesh
                lines = np.empty(len(springs) * 3, dtype=np.int32)
                lines[0::3] = 2
                lines[1::3] = springs[:, 0]
                lines[2::3] = springs[:, 1]
                mesh = pv.PolyData(self.history[0]["nodes"], lines=lines)

                # Initialize cell colors (RGBA uint8)
                mesh_colors = np.zeros((len(springs), 4), dtype=np.uint8)
                mesh.cell_data["colors"] = mesh_colors

                # Add mesh to plotter
                plotter.add_mesh(
                    mesh,
                    scalars="colors",
                    rgba=True,
                    line_width=1.5,
                    show_scalar_bar=False,
                    lighting=False,
                )

                # Add projectile wireframe box
                pw = self.config.get("projectile", {}).get("blade_width", 0.02)
                pt = self.config.get("projectile", {}).get("edge_thickness", 0.005)
                ph = 0.005
                proj_mesh = pv.Box(bounds=[-pw / 2, pw / 2, -pt / 2, pt / 2, -ph, ph])
                proj_actor = plotter.add_mesh(
                    proj_mesh,
                    color=[230, 230, 250],
                    style="wireframe",
                    line_width=2.5,
                    lighting=False,
                )

                # Position camera precisely
                yaw_rad = math.radians(yaw)
                pitch_rad = math.radians(pitch)
                cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
                cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
                R = np.array([
                    [cy, 0.0, -sy],
                    [-sy * sp, cp, -cy * sp],
                    [sy * cp, sp, cy * cp]
                ])
                local_x = R[0, :]
                local_y = R[1, :]
                local_z = R[2, :]

                grid_center = np.mean(self.history[0]["nodes"][:self.n_nodes_per_layer], axis=0)

                # Apply custom camera distance / pan if provided
                if distance is not None:
                    px = pan_x if pan_x is not None else 0.0
                    py = pan_y if pan_y is not None else 0.0
                    focal_point = grid_center - px * local_x - py * local_y
                    camera_position = focal_point + distance * local_z
                else:
                    x_span = x_max - x_min
                    y_span = y_max - y_min
                    distance_calc = max(x_span, y_span) * 2.5
                    focal_point = grid_center
                    camera_position = focal_point + distance_calc * local_z

                plotter.camera.position = camera_position
                plotter.camera.focal_point = focal_point
                plotter.camera.up = local_y

                plotter.show(auto_close=False, interactive=False, interactive_update=True)
                use_pyvista = True
            except Exception:
                # Silently fall back to optimized Matplotlib path if anything fails
                use_pyvista = False

        if use_pyvista:
            # RENDER USING PYVISTA + MATPLOTLIB 2D IMSHOW
            fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            # Draw initial frame
            plotter.render()
            img = plotter.image
            im = ax.imshow(img)

            def init() -> list[Any]:
                return [im]

            def update(frame_idx: int) -> list[Any]:
                frame = self.history[frame_idx]
                nodes = frame["nodes"]
                failed = frame["failed"]
                p_pos = frame["projectile_pos"]

                # Update mesh positions
                mesh.points = nodes

                # Update projectile position
                proj_actor.position = p_pos

                # Calculate strains
                p1 = nodes[springs[:, 0]]
                p2 = nodes[springs[:, 1]]
                lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
                strains = (lengths - rest_lengths) / rest_lengths

                # Update cell colors vectorised
                colors = np.zeros((len(springs), 4), dtype=np.uint8)
                n_failed = len(failed)
                failed_mask = np.zeros(len(springs), dtype=bool)
                failed_mask[:n_failed] = failed

                # Ruptured springs: transparent brick red
                colors[failed_mask] = [139, 0, 0, 45]

                # Active springs: Blue -> Yellow -> Crimson
                active = ~failed_mask
                ratios = np.clip(strains[active] / fail_thresh, 0.0, 1.0)

                mask1 = ratios <= 0.5
                ratio1 = ratios[mask1] / 0.5
                mask2 = ratios > 0.5
                ratio2 = (ratios[mask2] - 0.5) / 0.5

                c_active = np.zeros((np.sum(active), 4), dtype=np.uint8)
                c_active[mask1, 0] = (0 + 255 * ratio1).astype(np.uint8)
                c_active[mask1, 1] = (191 + (215 - 191) * ratio1).astype(np.uint8)
                c_active[mask1, 2] = (255 - 255 * ratio1).astype(np.uint8)
                c_active[mask1, 3] = 230

                c_active[mask2, 0] = (255 - (255 - 220) * ratio2).astype(np.uint8)
                c_active[mask2, 1] = (215 - (215 - 20) * ratio2).astype(np.uint8)
                c_active[mask2, 2] = (0 + 60 * ratio2).astype(np.uint8)
                c_active[mask2, 3] = 230

                colors[active] = c_active
                mesh.cell_data["colors"] = colors

                # Render and update imshow artist
                plotter.render()
                im.set_array(plotter.image)
                return [im]

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(self.history),
                init_func=init,
                blit=True,
            )

        else:
            # FALLBACK PATH: OPTIMIZED MATPLOTLIB 3D LINE3DCOLLECTION
            fig = plt.figure(figsize=(7, 5), dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")

            # Style Viewport
            if distance is not None:
                x_span = x_max - x_min
                y_span = y_max - y_min
                max_span = max(x_span, y_span)
                default_dist = max_span * 2.5
                zoom = default_dist / distance

                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                cz = (z_min + z_max) / 2.0

                px = pan_x if pan_x is not None else 0.0
                py = pan_y if pan_y is not None else 0.0

                dx_lim = (x_span / 2.0) / zoom
                dy_lim = (y_span / 2.0) / zoom
                dz_lim = ((z_max - z_min) / 2.0) / zoom

                cx_shifted = cx - px
                cy_shifted = cy - py

                ax.set_xlim3d(cx_shifted - dx_lim, cx_shifted + dx_lim)
                ax.set_ylim3d(cy_shifted - dy_lim, cy_shifted + dy_lim)
                ax.set_zlim3d(cz - dz_lim, cz + dz_lim)
            else:
                ax.set_xlim3d(x_min, x_max)
                ax.set_ylim3d(y_min, y_max)
                ax.set_zlim3d(z_min, z_max)

            ax.set_title("KevlarGrid 3D Woven Fabric Impact Telemetry")
            ax.view_init(elev=pitch, azim=yaw)

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_zlabel("Z (m)")

            # Line3DCollection is 100x faster than plotting each spring as a separate Line3D
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            lines_collection = Line3DCollection([], linewidths=0.8)
            ax.add_collection3d(lines_collection)

            # Projectile wireframe container (kept as single Line3D)
            (proj_line,) = ax.plot([], [], [], color="red", linewidth=2.0)

            def init() -> list[Any]:
                lines_collection.set_segments([])
                proj_line.set_data_3d([], [], [])
                return [lines_collection, proj_line]

            def update(frame_idx: int) -> list[Any]:
                frame = self.history[frame_idx]
                nodes = frame["nodes"]
                failed = frame["failed"]
                p_pos = frame["projectile_pos"]

                p1 = nodes[springs[:, 0]]
                p2 = nodes[springs[:, 1]]
                lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))
                strains = (lengths - rest_lengths) / rest_lengths

                # Update segment geometries in-place
                segments = np.stack([p1, p2], axis=1)
                lines_collection.set_segments(segments)

                # Vectorized color mapping
                colors = np.zeros((len(springs), 4))
                n_failed = len(failed)
                failed_mask = np.zeros(len(springs), dtype=bool)
                failed_mask[:n_failed] = failed

                # Transparent brick red for ruptured springs
                colors[failed_mask] = [0.5, 0.0, 0.0, 0.15]

                # Active springs color interpolation
                active = ~failed_mask
                ratios = np.clip(strains[active] / fail_thresh, 0.0, 1.0)
                colors[active, 0] = ratios
                colors[active, 1] = 1.0 - ratios
                colors[active, 2] = 0.5 * (1.0 - ratios)
                colors[active, 3] = 0.8

                lines_collection.set_color(colors)

                # Update projectile wireframe
                pw = 0.02
                pt = 0.005
                ph = 0.005
                p_offsets = np.array([
                    [-pw / 2, -pt / 2, -ph / 2],
                    [pw / 2, -pt / 2, -ph / 2],
                    [pw / 2, pt / 2, -ph / 2],
                    [-pw / 2, pt / 2, -ph / 2],
                    [-pw / 2, -pt / 2, ph / 2],
                    [pw / 2, -pt / 2, ph / 2],
                    [pw / 2, pt / 2, ph / 2],
                    [-pw / 2, pt / 2, ph / 2],
                ])
                corners = p_pos + p_offsets
                loop = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
                proj_line.set_data_3d(corners[loop, 0], corners[loop, 1], corners[loop, 2])

                return [lines_collection, proj_line]

            anim = animation.FuncAnimation(
                fig,
                update,
                frames=len(self.history),
                init_func=init,
                blit=True,
            )

        # Compile and save
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".gif":
            if animation.ImageMagickWriter.isAvailable():
                writer = animation.ImageMagickWriter(fps=fps)
                anim.save(filepath, writer=writer)
            else:
                anim.save(filepath, writer="pillow", fps=fps)
        else:
            if animation.FFMpegWriter.isAvailable():
                writer = animation.FFMpegWriter(
                    fps=fps,
                    codec="h264",
                    bitrate=1800,
                    extra_args=["-pix_fmt", "yuv420p"]
                )
                anim.save(filepath, writer=writer)
            else:
                anim.save(
                    filepath,
                    writer="ffmpeg",
                    fps=fps,
                    codec="h264",
                    extra_args=["-pix_fmt", "yuv420p"]
                )

        plt.close(fig)
        if use_pyvista:
            plotter.close()

        # Validate output: an empty/corrupt MP4 is typically < 1 KB
        file_size = os.path.getsize(filepath)
        if file_size < 1024:
            raise RuntimeError(
                f"Video export failed: output file '{filepath}' is only "
                f"{file_size} bytes (expected at least several KB). "
                f"This usually means frame rendering encountered an error."
            )
