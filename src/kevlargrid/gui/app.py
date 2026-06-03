"""Main DearPyGui application launch and background thread simulation coordinator.

Provides multi-threaded execution management for the explicit solver
and handles UI state rendering, dynamic plots, interactive 3D viewport,
post-simulation playback scrubbing, and crash recovery.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]

import numpy as np

from kevlargrid.gui.config_panel import ConfigPanel
from kevlargrid.gui.controls import SimulationControls
from kevlargrid.gui.dashboard import ResultsDashboard
from kevlargrid.gui.plots import EnergyPlot, StrainPlot
from kevlargrid.gui.viewport3d import Viewport3D
from kevlargrid.io.config import ValidationError, load_config, save_config, validate_config
from kevlargrid.solver import (
    backend,
    compute_cfl_timestep,
    compute_interply_contact_forces,
    compute_kinetic_energy,
    compute_strain_energy,
    generate_rectangular_grid,
)
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.projectile import (
    Projectile,
    distribute_contact_forces,
    update_contact_zone,
)
from kevlargrid.utils import get_logger

logger = get_logger("gui.app")


class SimRunner:
    """Thread-safe background simulation runner with history tracking."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # Telemetry metrics (protected by lock)
        self.state = "idle"  # idle, running, paused, completed, error
        self.elapsed_time = 0.0
        self.step = 0
        self.speed = 0.0
        self.eta = 0.0
        self.ke = 0.0
        self.se = 0.0
        self.damp_dissipated = 0.0
        self.peak_strain = 0.0
        self.proj_ke = 0.0
        self.failed_count = 0
        self.error_message = ""

        # History tracking for post-run playback
        self.history: list[dict[str, Any]] = []
        self.results_report: dict[str, Any] = {}

        # Cache of structural geometry
        self.grid_nodes = np.zeros((0, 3))
        self.grid_failed = np.zeros(0, dtype=bool)
        self.projectile_pos = np.zeros(3)

    def start(self, config: dict) -> None:
        """Launch the solver on a background worker thread."""
        self.reset()
        self.state = "running"
        self.stop_event.clear()
        self.pause_event.clear()
        logger.info("Initializing solver background worker thread...")
        self.active_thread = threading.Thread(target=self._run_loop, args=(config,), daemon=True)
        self.active_thread.start()
        logger.info("Solver worker thread started successfully.")

    def pause(self) -> None:
        """Pause worker thread execution."""
        with self.lock:
            if self.state == "running":
                self.state = "paused"
                self.pause_event.set()
                logger.info("Simulation execution paused by user.")

    def resume(self) -> None:
        """Resume worker thread execution."""
        with self.lock:
            if self.state == "paused":
                self.state = "running"
                self.pause_event.clear()
                logger.info("Simulation execution resumed by user.")

    def stop(self) -> None:
        """Forcibly terminate background execution."""
        logger.info("User requested simulation stop. Terminating solver thread...")
        self.stop_event.set()
        self.pause_event.clear()
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join(timeout=1.0)
        logger.info("Solver thread successfully terminated.")
        self.reset()

    def reset(self) -> None:
        """Restore initial telemetry and clear history tracking buffers."""
        with self.lock:
            self.state = "idle"
            self.elapsed_time = 0.0
            self.step = 0
            self.speed = 0.0
            self.eta = 0.0
            self.ke = 0.0
            self.se = 0.0
            self.damp_dissipated = 0.0
            self.peak_strain = 0.0
            self.proj_ke = 0.0
            self.failed_count = 0
            self.error_message = ""
            self.history.clear()
            self.results_report.clear()
            self.grid_nodes = np.zeros((0, 3))
            self.grid_failed = np.zeros(0, dtype=bool)
            self.projectile_pos = np.zeros(3)

    def get_telemetry(self) -> dict[str, Any]:
        """Fetch current thread-safe simulation telemetry data."""
        with self.lock:
            return {
                "state": self.state,
                "elapsed_time": self.elapsed_time,
                "step": self.step,
                "speed": self.speed,
                "eta": self.eta,
                "ke": self.ke,
                "se": self.se,
                "damp_dissipated": self.damp_dissipated,
                "peak_strain": self.peak_strain,
                "proj_ke": self.proj_ke,
                "failed_count": self.failed_count,
                "error_message": self.error_message,
                "history_length": len(self.history),
                "grid_nodes": self.grid_nodes.copy(),
                "grid_failed": self.grid_failed.copy(),
                "projectile_pos": self.projectile_pos.copy(),
                "results_report": self.results_report.copy(),
            }

    def _run_loop(self, config: dict) -> None:
        """Main solver dynamic time integration worker loop."""
        try:
            logger.info(
                "Initializing dynamic explicit solver. Compute Backend: %s, Hardware Device: %s",
                backend.get_backend_name().upper(),
                backend.get_active_device(),
            )
            # 1. Setup simulation components based on config
            mat = config["material"]
            grid_cfg = config["grid"]
            proj_cfg = config["projectile"]
            sim_cfg = config["simulation"]

            nx, ny, dx = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["dx"]
            n_plies = grid_cfg["n_plies"]
            t_ply = grid_cfg["t_ply"]

            # Build grid
            grid = generate_rectangular_grid(
                nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
            )

            # Build boundary mask S6.5.1
            boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
            n_nodes_per_layer = nx * ny
            # Only loop over physical layers in the grid to prevent IndexError in Mode A
            n_layers = n_plies if (t_ply is not None and n_plies > 1) else 1
            if grid_cfg["boundary_type"] == "fixed":
                for ply in range(n_layers):
                    offset = ply * n_nodes_per_layer
                    for i in range(nx):
                        for j in range(ny):
                            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                                boundary_mask[offset + i * ny + j] = True

            # Calculate safe CFL timestep
            k_penalty = 10.0 * np.mean(grid.stiffnesses)
            k_max_effective = max(np.max(grid.stiffnesses), k_penalty)
            dt = compute_cfl_timestep(
                np.array([k_max_effective]), grid.masses, dx, sim_cfg["cfl_factor"]
            )

            # Projectile setup
            proj = Projectile(
                mass=proj_cfg["mass"],
                velocity=proj_cfg["velocity"],
                position=proj_cfg["position"],
                blade_width=proj_cfg["blade_width"],
                edge_thickness=proj_cfg["edge_thickness"],
            )

            positions = grid.nodes.copy()
            velocities = np.zeros_like(positions)

            # Viscous damping parameter
            damping_coeff = sim_cfg["damping_coefficient"]
            duration = sim_cfg["duration"]

            # Loop tracking
            step = 0
            t_sim = 0.0
            last_time = time.time()
            last_step = 0

            # Store baseline snapshot at t=0
            initial_ke = compute_kinetic_energy(velocities, grid.masses)
            # Projectile initial kinetic energy S6.5.5
            initial_proj_ke = 0.5 * proj.mass * np.sum(proj.velocity**2)
            initial_se = 0.0
            initial_damp = 0.0
            self.history.append(
                {
                    "time": 0.0,
                    "nodes": positions.copy(),
                    "failed": grid.failed.copy(),
                    "projectile_pos": proj.position.copy(),
                    "ke": initial_ke,
                    "se": initial_se,
                    "damped": initial_damp,
                    "contact": 0.0,
                    "total": initial_ke + initial_proj_ke,
                    "failed_count": 0,
                    "peak_strain": 0.0,
                    "proj_ke": initial_proj_ke,
                }
            )

            # Peak deceleration recording
            accel_history = []
            damp_dissipated = 0.0

            # Fused chunking loop S7
            n_chunk = 100
            save_interval = 10
            jit_warmed = False

            while t_sim < duration:
                # Check stop signal
                if self.stop_event.is_set():
                    logger.info("Simulation loop aborted via stop event.")
                    break

                # Check pause signal
                if self.pause_event.is_set():
                    self.pause_event.wait(timeout=0.1)
                    last_time = time.time()  # reset speed calc
                    continue

                # How many steps to run in this chunk?
                steps_remaining = int(np.ceil((duration - t_sim) / dt))
                current_steps = min(n_chunk, steps_remaining)

                if current_steps <= 0:
                    break

                # JIT-Fused loop dispatching!
                if not jit_warmed:
                    logger.info(
                        "Executing initial integrator chunk; JIT compilation warm-up triggered..."
                    )
                    t_start_warm = time.time()

                if backend.get_backend_name() == "taichi":
                    from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop

                    (
                        positions,
                        velocities,
                        grid.failed,
                        proj.position,
                        proj.velocity,
                        damp_dissipated,
                        t_sim,
                        hist_pos,
                        hist_failed,
                        hist_proj_pos,
                        hist_time,
                        hist_ke,
                        hist_se,
                        hist_proj_ke,
                    ) = taichi_leapfrog_loop(
                        positions,
                        velocities,
                        grid.springs,
                        grid.stiffnesses,
                        grid.rest_lengths,
                        grid.failed,
                        grid.masses,
                        grid.tension_only,
                        boundary_mask,
                        proj.position,
                        proj.velocity,
                        proj.mass,
                        proj.blade_width,
                        proj.edge_thickness,
                        n_layers,
                        n_nodes_per_layer,
                        t_ply if t_ply is not None else 0.002,
                        dx,
                        k_penalty,
                        damping_coeff,
                        mat["failure_strain"],
                        dt,
                        current_steps,
                        save_interval,
                        damp_dissipated,
                        t_sim,
                    )
                else:
                    (
                        positions,
                        velocities,
                        grid.failed,
                        proj.position,
                        proj.velocity,
                        damp_dissipated,
                        t_sim,
                        hist_pos,
                        hist_failed,
                        hist_proj_pos,
                        hist_time,
                        hist_ke,
                        hist_se,
                        hist_proj_ke,
                    ) = fused_leapfrog_loop(
                        positions,
                        velocities,
                        grid.springs,
                        grid.stiffnesses,
                        grid.rest_lengths,
                        grid.failed,
                        grid.masses,
                        grid.tension_only,
                        boundary_mask,
                        proj.position,
                        proj.velocity,
                        proj.mass,
                        proj.blade_width,
                        proj.edge_thickness,
                        n_layers,
                        n_nodes_per_layer,
                        t_ply if t_ply is not None else 0.002,
                        dx,
                        k_penalty,
                        damping_coeff,
                        mat["failure_strain"],
                        dt,
                        current_steps,
                        save_interval,
                        damp_dissipated,
                        t_sim,
                    )

                if not jit_warmed:
                    logger.info(
                        "JIT compilation warm-up complete in %.2f seconds.",
                        time.time() - t_start_warm,
                    )
                    jit_warmed = True

                step += current_steps

                # Check for numerical instability (NaN/inf values) S6.5.11
                if np.isnan(positions[0, 0]) or np.any(np.isnan(positions)):
                    logger.error(
                        "Numerical instability detected: coordinates have diverged to NaN."
                    )
                    raise ValueError(
                        "Numerical instability detected: coordinates have diverged to NaN.\n"
                        "Please reduce your CFL safety factor or increase spacing dx."
                    )

                # Store live metrics at chunk end
                ke = compute_kinetic_energy(velocities, grid.masses)
                diff = positions[grid.springs[:, 1]] - positions[grid.springs[:, 0]]
                lengths = np.sqrt(np.sum(diff**2, axis=1))
                strains = (lengths - grid.rest_lengths) / grid.rest_lengths
                se = compute_strain_energy(
                    strains, grid.stiffnesses, grid.rest_lengths, grid.failed
                )
                proj_ke = 0.5 * proj.mass * np.sum(proj.velocity**2)
                failed_count = int(np.sum(grid.failed))
                peak_strain = float(np.max(strains)) if len(strains) > 0 else 0.0

                # Compute deceleration from contact force at the end of chunk
                update_contact_zone(proj, grid, proximity_threshold=dx * 2.0, positions=positions)
                proj_forces = distribute_contact_forces(
                    proj, grid, positions=positions, k_contact=k_penalty
                )
                proj_reaction_force = -np.sum(proj_forces, axis=0)
                proj_accel = proj_reaction_force / proj.mass
                accel_g = np.linalg.norm(proj_accel) / 9.81
                accel_history.append(accel_g)

                # Compute interply energy for mode B contact
                interply_energy = 0.0
                if n_plies > 1 and t_ply is not None:
                    _, interply_energy = compute_interply_contact_forces(
                        positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
                    )

                # Append JIT pre-allocated frames into our history list!
                m_actual = current_steps // save_interval
                for frame_idx in range(m_actual):
                    p_pos = hist_proj_pos[frame_idx]
                    f_ke = hist_ke[frame_idx]
                    f_se = hist_se[frame_idx]
                    f_p_ke = hist_proj_ke[frame_idx]
                    f_failed_c = int(np.sum(hist_failed[frame_idx]))

                    self.history.append(
                        {
                            "time": hist_time[frame_idx],
                            "nodes": hist_pos[frame_idx].copy(),
                            "failed": hist_failed[frame_idx].copy(),
                            "projectile_pos": p_pos.copy(),
                            "ke": f_ke,
                            "se": f_se,
                            "damped": damp_dissipated,
                            "contact": interply_energy,
                            "total": f_ke + f_se + damp_dissipated + interply_energy + f_p_ke,
                            "failed_count": f_failed_c,
                            "peak_strain": 0.0,
                            "proj_ke": f_p_ke,
                        }
                    )

                # Update live telemetry state protected by lock
                now = time.time()
                elapsed_real = now - last_time
                if elapsed_real >= 0.05:
                    steps_per_sec = (step - last_step) / elapsed_real
                    last_time = now
                    last_step = step
                else:
                    steps_per_sec = self.speed

                # Estimate ETA
                rem_steps = int((duration - t_sim) / dt)
                eta = rem_steps / steps_per_sec if steps_per_sec > 0.0 else 0.0

                with self.lock:
                    self.elapsed_time = t_sim
                    self.step = step
                    self.speed = steps_per_sec
                    self.eta = eta
                    self.ke = ke
                    self.se = se
                    self.damp_dissipated = damp_dissipated
                    self.peak_strain = peak_strain
                    self.proj_ke = proj_ke
                    self.failed_count = failed_count
                    self.grid_nodes = positions.copy()
                    self.grid_failed = grid.failed.copy()
                    self.projectile_pos = proj.position.copy()

                # Projectile arrest termination check
                if proj.velocity[2] <= 0.0:
                    break

            # Simulation Completed: Generate post-run summary impact report
            initial_v = np.linalg.norm(proj_cfg["velocity"])
            final_v = np.linalg.norm(proj.velocity)

            # Arrest check (did the projectile turn around before reaching panel boundary limits?)
            arrested = proj.velocity[2] <= 0.0

            peak_decel = max(accel_history) if accel_history else 0.0
            tot_springs = len(grid.springs)
            rupture_pct = (failed_count / tot_springs) * 100.0 if tot_springs > 0 else 0.0

            # Find maximum layer perforated S6.5.1
            max_ply_perf = -1
            is_mode_b = t_ply is not None and n_plies > 1
            if is_mode_b:
                spring_layers = grid.springs[:, 0] // n_nodes_per_layer
                for ply in range(n_plies):
                    ply_springs = spring_layers == ply
                    ply_failed = grid.failed[ply_springs]
                    if (
                        len(ply_failed) > 0 and np.mean(ply_failed) > 0.70
                    ):  # 70% threshold is complete yarn failure
                        max_ply_perf = ply

            logger.info(
                "Simulation completed successfully in %d steps. Projectile arrested: %s. "
                "Peak deceleration: %.2f g. Rupture: %.2f%%. Max layer perforated: %d",
                step,
                arrested,
                peak_decel,
                rupture_pct,
                max_ply_perf,
            )

            with self.lock:
                self.state = "completed"
                self.results_report = {
                    "arrested": arrested,
                    "peak_deceleration_g": peak_decel,
                    "yarn_rupture_percentage": rupture_pct,
                    "residual_velocity_ms": float(final_v) if not arrested else 0.0,
                    "energy_dissipation_efficiency": float(
                        (initial_v**2 - final_v**2) / (initial_v**2)
                    )
                    if initial_v > 0
                    else 0.0,
                    "max_layer_perforated": max_ply_perf,
                }
                # Store final trajectory step
                self.grid_nodes = positions.copy()
                self.grid_failed = grid.failed.copy()
                self.projectile_pos = proj.position.copy()

        except Exception as e:
            logger.exception("Simulation execution crashed due to an unhandled exception: %s", e)
            with self.lock:
                self.state = "error"
                self.error_message = str(e)


# Global instances
runner = SimRunner()
config_panel = ConfigPanel()
controls = SimulationControls()
strain_plot = StrainPlot()
energy_plot = EnergyPlot()
dashboard = ResultsDashboard()
dashboard.set_references(runner, config_panel)
viewport3d = Viewport3D()

# Playback controls state
playback_active = False
playback_frame_index = 0
playback_speed = 1.0  # multiplier (frames per update tick)
playback_last_tick = 0.0

last_autosave_time = 0.0
AUTOSAVE_DIR = ".autosave"
AUTOSAVE_PATH = f"{AUTOSAVE_DIR}/session.json"


def launch() -> None:
    """Initialise and launch the KevlarGrid GUI application."""
    if dpg is None:
        print("Error: DearPyGui is not installed. GUI cannot be launched.")
        return

    dpg.create_context()
    dpg.create_viewport(title="KevlarGrid Explicit Dynamic Solver v2.0", width=1280, height=830)

    # File Menu callbacks
    def _menu_save_config():
        config = config_panel.get_config()
        save_config(config, "configs/saved_configuration.json")
        _show_modal_message(
            "Config Saved", "Configuration successfully saved to:\nconfigs/saved_configuration.json"
        )

    def _menu_load_config():
        if os.path.exists("configs/saved_configuration.json"):
            config = load_config("configs/saved_configuration.json")
            config_panel.set_config(config)
            _show_modal_message(
                "Config Loaded", "Configuration loaded from:\nconfigs/saved_configuration.json"
            )
        else:
            _show_modal_message(
                "Error", "No saved configuration file found in 'configs/saved_configuration.json'."
            )

    # Main window viewport setup
    with dpg.viewport_menu_bar(), dpg.menu(label="File"):
        dpg.add_menu_item(label="Load Configuration", callback=_menu_load_config)
        dpg.add_menu_item(label="Save Configuration", callback=_menu_save_config)
        dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

    with (
        dpg.window(
            label="KevlarGrid Workspace",
            width=1260,
            height=760,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
        ),
        dpg.group(horizontal=True),
    ):
        # Left Panel: Sidebar Config inputs
        config_panel.build()

        dpg.add_spacer(width=10)

        # Right Panel: Simulation controls, progress metrics, and live tab lists
        with dpg.group():
            controls.build()

            dpg.add_spacer(height=5)

            # Tab layout split panel (Viewport, Plots, Dashboard Summary)
            with dpg.tab_bar():
                # Tab 1: 3D Perspective Viewport Mesh
                with dpg.tab(label="3D Viewport Visualization", tag="tab_viewport"):
                    viewport3d.build()

                    # Post-simulation playback widgets row toolbar
                    with dpg.group(tag="playback_group", show=False, horizontal=True):
                        dpg.add_button(
                            label="Play",
                            tag="pb_play_btn",
                            width=65,
                            callback=lambda: _toggle_playback(True),
                        )
                        dpg.add_button(
                            label="Pause",
                            tag="pb_pause_btn",
                            width=65,
                            callback=lambda: _toggle_playback(False),
                        )
                        dpg.add_button(
                            label="Step <<", width=65, callback=lambda: _step_playback(-1)
                        )
                        dpg.add_button(
                            label="Step >>", width=65, callback=lambda: _step_playback(1)
                        )
                        dpg.add_slider_int(
                            label="Timeline",
                            tag="pb_slider",
                            width=220,
                            min_value=0,
                            max_value=100,
                            default_value=0,
                            callback=_on_playback_slider_drag,
                        )
                        dpg.add_combo(
                            label="Speed",
                            tag="pb_speed_combo",
                            width=80,
                            items=["0.25x", "0.5x", "1.0x", "2.0x", "5.0x"],
                            default_value="1.0x",
                            callback=_on_playback_speed_change,
                        )

                # Tab 2: Dynamic 2D Telemetry Plots
                with (
                    dpg.tab(label="Dynamic Analytics Traces", tag="tab_plots"),
                    dpg.group(horizontal=True),
                ):
                    with dpg.group(width=410):
                        strain_plot.build()
                    with dpg.group(width=410):
                        energy_plot.build()

                # Tab 3: Dynamic Results Pass/Fail Dashboard
                with dpg.tab(label="Impact Results Summary", tag="tab_dashboard"):
                    dashboard.build()

    # Core Thread-Safe Control Callbacks
    def _on_start_btn():
        cfg = config_panel.get_config()
        try:
            validate_config(cfg)

            # Reset and initialize viewport coordinate mappings
            dummy_grid = generate_rectangular_grid(
                nx=cfg["grid"]["nx"],
                ny=cfg["grid"]["ny"],
                dx=cfg["grid"]["dx"],
                material=cfg["material"],
                n_plies=cfg["grid"]["n_plies"],
                t_ply=cfg["grid"]["t_ply"],
            )
            viewport3d.reset(
                grid=dummy_grid,
                n_plies=cfg["grid"]["n_plies"],
                n_nodes_per_layer=cfg["grid"]["nx"] * cfg["grid"]["ny"],
            )

            # Reset plots
            strain_plot.reset(threshold=cfg["material"]["failure_strain"])
            energy_plot.reset()
            _toggle_playback(False)
            dpg.configure_item("playback_group", show=False)

            runner.start(cfg)
            controls.set_button_states("running")
        except ValidationError as e:
            _show_modal_message("Validation Error", str(e))

    def _on_pause_btn():
        runner.pause()
        controls.set_button_states("paused")

    def _on_resume_btn():
        runner.resume()
        controls.set_button_states("running")

    def _on_stop_btn():
        runner.stop()
        controls.set_button_states("idle")
        _toggle_playback(False)
        dpg.configure_item("playback_group", show=False)

    def _on_reset_btn():
        runner.reset()
        controls.set_button_states("idle")
        _toggle_playback(False)
        dpg.configure_item("playback_group", show=False)

        # Reset telemetry widgets
        controls.update_telemetry(0.0, 0.001, 0, 0.0, 0.0, 0, 0.0, 0.0)
        strain_plot.reset()
        energy_plot.reset()

        # Redraw blank viewport S6.5.3
        reset_cfg = config_panel.get_config()
        blank_grid = generate_rectangular_grid(
            nx=reset_cfg["grid"]["nx"],
            ny=reset_cfg["grid"]["ny"],
            dx=reset_cfg["grid"]["dx"],
            material=reset_cfg["material"],
            n_plies=reset_cfg["grid"]["n_plies"],
            t_ply=reset_cfg["grid"]["t_ply"],
        )
        viewport3d.reset(blank_grid)

    # Bind controls callbacks
    controls.start_callback = lambda: (
        _on_resume_btn() if runner.get_telemetry()["state"] == "paused" else _on_start_btn()
    )
    controls.pause_callback = _on_pause_btn
    controls.stop_callback = _on_stop_btn
    controls.reset_callback = _on_reset_btn

    # Bind sidebar config load/save callbacks S6.5.9
    config_panel.load_callback = _menu_load_config
    config_panel.save_callback = _menu_save_config

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # --- S4.9 Session Auto-Recovery Check ---
    if os.path.exists(AUTOSAVE_PATH):
        try:
            cached_config = load_config(AUTOSAVE_PATH)
            _spawn_recovery_modal(cached_config)
        except Exception:
            pass

    # Custom frames rendering loop to manage background synchronization
    global last_autosave_time
    last_autosave_time = time.time()

    # Create dynamic initial grid for screen renders S6.5.3
    initial_cfg = config_panel.get_config()
    init_grid = generate_rectangular_grid(
        nx=initial_cfg["grid"]["nx"],
        ny=initial_cfg["grid"]["ny"],
        dx=initial_cfg["grid"]["dx"],
        material=initial_cfg["material"],
        n_plies=initial_cfg["grid"]["n_plies"],
        t_ply=initial_cfg["grid"]["t_ply"],
    )
    viewport3d.reset(init_grid)

    last_grid_key = None

    while dpg.is_dearpygui_running():
        # Update metrics from solver thread
        tel = runner.get_telemetry()
        state = tel["state"]
        cfg = config_panel.get_config()

        # Check for grid parameter updates in real time (idle state) S6.5.3
        if state == "idle":
            current_grid_key = (
                cfg["grid"]["nx"],
                cfg["grid"]["ny"],
                cfg["grid"]["dx"],
                cfg["grid"]["n_plies"],
                cfg["grid"]["t_ply"],
                cfg["grid"]["boundary_type"],
                cfg["material"].get("material_name", ""),
            )
            if current_grid_key != last_grid_key:
                last_grid_key = current_grid_key
                try:
                    preview_grid = generate_rectangular_grid(
                        nx=cfg["grid"]["nx"],
                        ny=cfg["grid"]["ny"],
                        dx=cfg["grid"]["dx"],
                        material=cfg["material"],
                        n_plies=cfg["grid"]["n_plies"],
                        t_ply=cfg["grid"]["t_ply"],
                    )
                    viewport3d.reset(preview_grid)
                except Exception:
                    pass

        if state == "running":
            controls.update_telemetry(
                elapsed_time=tel["elapsed_time"],
                duration=cfg["simulation"]["duration"],
                step=tel["step"],
                speed=tel["speed"],
                eta=tel["eta"],
                failed_count=tel["failed_count"],
                ke=tel["ke"],
                se=tel["se"],
            )
            # Live Plotting Traces Update S6.5.5
            strain_plot.update(tel["elapsed_time"], tel["peak_strain"])
            energy_plot.update(
                tel["elapsed_time"],
                {
                    "kinetic": tel["ke"],
                    "proj_ke": tel["proj_ke"],
                    "strain": tel["se"],
                    "damped": tel["damp_dissipated"],
                    "contact": 0.0,
                    "total": tel["ke"] + tel["se"] + tel["damp_dissipated"] + tel["proj_ke"],
                },
            )

            # Live 3D Mesh redrawing
            if len(tel["grid_nodes"]) > 0:
                viewport3d.update(tel["grid_nodes"], tel["grid_failed"])
                viewport3d.draw_projectile(
                    tel["projectile_pos"],
                    cfg["projectile"]["blade_width"],
                    cfg["projectile"]["edge_thickness"],
                )

        elif state == "completed":
            controls.set_button_states("completed")
            controls.update_telemetry(
                elapsed_time=cfg["simulation"]["duration"],
                duration=cfg["simulation"]["duration"],
                step=tel["step"],
                speed=0.0,
                eta=0.0,
                failed_count=tel["failed_count"],
                ke=tel["ke"],
                se=tel["se"],
            )
            # Populate Pass/Fail dynamic results summary
            if len(tel["results_report"]) > 0:
                dashboard.populate(tel["results_report"])

            # Enable post-simulation playback timeline scrubbing S5.6
            h_len = tel["history_length"]
            if h_len > 1:
                dpg.configure_item("playback_group", show=True)
                dpg.configure_item("pb_slider", max_value=h_len - 1)

                # Active Playback Loop S5.6
                global playback_active, playback_frame_index, playback_last_tick
                if playback_active:
                    now_tick = time.time()
                    # Regulate frame rate speed
                    interval = 0.05 / playback_speed
                    if now_tick - playback_last_tick >= interval:
                        playback_last_tick = now_tick
                        playback_frame_index = (playback_frame_index + 1) % h_len
                        dpg.set_value("pb_slider", playback_frame_index)
                        _render_playback_frame(playback_frame_index)

        elif state == "error":
            controls.set_button_states("idle")
            _show_modal_message("Solver Error", tel["error_message"])
            runner.reset()

        # --- S4.9 Background Auto-Save 60s Timer ---
        now = time.time()
        if now - last_autosave_time >= 60.0:
            last_autosave_time = now
            try:
                active_cfg = config_panel.get_config()
                os.makedirs(AUTOSAVE_DIR, exist_ok=True)
                save_config(active_cfg, AUTOSAVE_PATH)
            except Exception:
                pass

        dpg.render_dearpygui_frame()

    # Cleanup active solver thread on exit
    runner.stop()
    dpg.destroy_context()


# --- Playback Callbacks & Renderers ---


def _toggle_playback(play: bool) -> None:
    """Start or suspend post-run timeline playback animations."""
    global playback_active, playback_last_tick
    playback_active = play
    playback_last_tick = time.time()


def _step_playback(steps: int) -> None:
    """Manual step frame index forward/backward timeline controls."""
    global playback_frame_index, playback_active
    playback_active = False  # Pause auto play
    tel = runner.get_telemetry()
    h_len = tel["history_length"]
    if h_len > 1:
        playback_frame_index = max(0, min(h_len - 1, playback_frame_index + steps))
        dpg.set_value("pb_slider", playback_frame_index)
        _render_playback_frame(playback_frame_index)


def _on_playback_slider_drag(sender: str, app_data: int) -> None:
    """Timeline slider scrubbing handler."""
    global playback_frame_index, playback_active
    playback_active = False  # Pause auto play
    playback_frame_index = app_data
    _render_playback_frame(playback_frame_index)


def _on_playback_speed_change(sender: str, app_data: str) -> None:
    """Timeline frame rate playback speed modifier."""
    global playback_speed
    try:
        val = app_data.replace("x", "")
        playback_speed = float(val)
    except Exception:
        playback_speed = 1.0


def _render_playback_frame(frame_idx: int) -> None:
    """Sync viewport dynamic meshes, lines, and graphs to selected timeline frame."""
    history = runner.history
    if frame_idx >= len(history):
        return

    frame = history[frame_idx]
    cfg = config_panel.get_config()

    # 1. Update 3D viewport
    viewport3d.update(frame["nodes"], frame["failed"])
    viewport3d.draw_projectile(
        frame["projectile_pos"],
        cfg["projectile"]["blade_width"],
        cfg["projectile"]["edge_thickness"],
    )

    # 2. Update telemetry text values on panel controls
    controls.update_telemetry(
        elapsed_time=frame["time"],
        duration=cfg["simulation"]["duration"],
        step=frame["time"] // 1e-6,  # estimate step
        speed=0.0,
        eta=0.0,
        failed_count=frame["failed_count"],
        ke=frame["ke"],
        se=frame["se"],
    )

    # 3. Update plot timeline markers S6.5.4
    strain_plot.set_playback_marker(frame["time"], frame.get("peak_strain", 0.0))
    max_energy = max(frame["total"], 9000.0)
    energy_plot.set_playback_marker(frame["time"], max_energy)


def _show_modal_message(title: str, message: str) -> None:
    """Utility wrapper for spawning an overlay dialogue box."""
    if dpg is None:  # pragma: no cover
        return

    modal_tag = "popup_modal_message"
    if dpg.does_item_exist(modal_tag):
        dpg.delete_item(modal_tag)

    with dpg.window(
        label=title,
        tag=modal_tag,
        modal=True,
        show=True,
        width=320,
        height=180,
        no_resize=True,
        no_move=True,
    ):
        dpg.add_text(message)
        dpg.add_spacer(height=10)
        dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item(modal_tag))


def _spawn_recovery_modal(config: dict) -> None:
    """Prompt the user to restore their previous session."""
    if dpg is None:  # pragma: no cover
        return

    recovery_tag = "popup_modal_recovery"

    def _restore_session():
        config_panel.set_config(config)
        dpg.delete_item(recovery_tag)

    def _ignore_session():
        try:
            if os.path.exists(AUTOSAVE_PATH):
                os.remove(AUTOSAVE_PATH)
        except Exception:
            pass
        dpg.delete_item(recovery_tag)

    with dpg.window(
        label="Session Recovery",
        tag=recovery_tag,
        modal=True,
        show=True,
        width=380,
        height=180,
        no_resize=True,
        no_move=True,
    ):
        dpg.add_text(
            "We detected an autosaved configuration from a previous run.\nWould you like to recover your session parameters?"
        )
        dpg.add_spacer(height=15)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Yes, Restore", width=120, callback=_restore_session)
            dpg.add_button(label="No, Start Fresh", width=120, callback=_ignore_session)
