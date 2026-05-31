"""Headless off-screen 3D mesh animation compiler.

Renders mass-spring trajectories into high-fidelity slowed-down MP4 or GIF animations
representing the impact event, utilizing standard matplotlib Agg backend.
"""

from __future__ import annotations

import os
from typing import Any

# Force headless matplotlib rendering
import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


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
        """
        if not self.history:
            raise ValueError("No history frames loaded to export video.")

        # Build spring indexes to draw lines
        springs_list = []
        for ply in range(self.n_plies):
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
        n_springs_per_ply = len(springs_list) // self.n_plies

        # 1. Setup Matplotlib 3D Canvas
        fig = plt.figure(figsize=(7, 5), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # Determine strict spatial bounds across all history for tight viewport box
        all_nodes = np.vstack([f["nodes"] for f in self.history])
        x_min, x_max = np.min(all_nodes[:, 0]), np.max(all_nodes[:, 0])
        y_min, y_max = np.min(all_nodes[:, 1]), np.max(all_nodes[:, 1])
        z_min, z_max = np.min(all_nodes[:, 2]) - 0.01, np.max(all_nodes[:, 2]) + 0.01

        # Render 3D scene elements list
        spring_lines: list[Any] = []
        for _ in range(len(springs)):
            (line,) = ax.plot([], [], [], color="blue", linewidth=0.5)
            spring_lines.append(line)

        # Projectile wireframe container lines
        (proj_line,) = ax.plot([], [], [], color="red", linewidth=2.0)

        # Style Viewport
        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(z_min, z_max)
        ax.set_title("KevlarGrid 3D Woven Fabric Impact Telemetry")
        ax.view_init(elev=pitch, azim=yaw)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        def init() -> list[Any]:
            for line in spring_lines:
                line.set_data_3d([], [], [])
            proj_line.set_data_3d([], [], [])
            return [*spring_lines, proj_line]

        def update(frame_idx: int) -> list[Any]:
            frame = self.history[frame_idx]
            nodes = frame["nodes"]
            failed = frame["failed"]
            p_pos = frame["projectile_pos"]

            # Draw Springs
            # Live strain coloring limit
            fail_thresh = 0.036

            p1 = nodes[springs[:, 0]]
            p2 = nodes[springs[:, 1]]
            lengths = np.sqrt(np.sum((p2 - p1) ** 2, axis=1))

            # Simple orthogonal and diagonal rest length arrays
            rest_lengths = np.zeros(len(springs))
            for k in range(self.n_plies):
                offset = k * n_springs_per_ply
                for j in range(n_springs_per_ply):
                    idx_s = offset + j
                    # Orthogonal connects idx, idx+ny or idx, idx+1. Distances are dx (0.01 or similar)
                    rest_lengths[idx_s] = 0.01 if j < 2 * n_springs_per_ply / 3 else 0.01414

            strains = (lengths - rest_lengths) / rest_lengths

            for j in range(len(springs)):
                u, v = springs[j, 0], springs[j, 1]
                x_pts = [nodes[u, 0], nodes[v, 0]]
                y_pts = [nodes[u, 1], nodes[v, 1]]
                z_pts = [nodes[u, 2], nodes[v, 2]]
                line = spring_lines[j]

                line.set_data_3d(x_pts, y_pts, z_pts)

                if failed[j]:
                    line.set_color([0.5, 0.0, 0.0, 0.15])  # Faded brick red
                    line.set_linewidth(0.4)
                else:
                    ratio = min(1.0, max(0.0, strains[j] / fail_thresh))
                    # Blue -> Green -> Red interpolation
                    r = ratio
                    g = 1.0 - ratio
                    b = 0.5 * (1.0 - ratio)
                    line.set_color((r, g, b, 0.8))
                    line.set_linewidth(0.8)

            # Draw Projectile Wireframe
            pw = 0.02
            pt = 0.005
            ph = 0.005
            p_offsets = np.array(
                [
                    [-pw / 2, -pt / 2, -ph / 2],
                    [pw / 2, -pt / 2, -ph / 2],
                    [pw / 2, pt / 2, -ph / 2],
                    [-pw / 2, pt / 2, -ph / 2],
                    [-pw / 2, -pt / 2, ph / 2],
                    [pw / 2, -pt / 2, ph / 2],
                    [pw / 2, pt / 2, ph / 2],
                    [-pw / 2, pt / 2, ph / 2],
                ]
            )
            corners = p_pos + p_offsets
            # Connect loop corners
            loop = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3]
            proj_line.set_data_3d(corners[loop, 0], corners[loop, 1], corners[loop, 2])

            return [*spring_lines, proj_line]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            init_func=init,
            blit=False,
        )

        # Compile and save
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".gif":
            gif_writer = (
                animation.ImageMagickWriter(fps=fps)
                if animation.ImageMagickWriter.isAvailable()
                else "pillow"
            )
            anim.save(filepath, writer=gif_writer, fps=fps)
        else:
            # Enforce MP4 via FFMpegWriter or default fallback
            mp4_writer = (
                animation.FFMpegWriter(fps=fps, bitrate=1800)
                if animation.FFMpegWriter.isAvailable()
                else "ffmpeg"
            )
            anim.save(filepath, writer=mp4_writer, fps=fps)

        plt.close(fig)
