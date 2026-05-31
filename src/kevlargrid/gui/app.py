"""Main DearPyGui application launch and background thread simulation coordinator.

Provides multi-threaded execution management for the explicit solver
and handles UI state rendering, dynamic plots, interactive 3D viewport,
post-simulation playback scrubbing, and crash recovery.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
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
    check_failures,
    compute_cfl_timestep,
    compute_interply_contact_forces,
    compute_kinetic_energy,
    compute_spring_forces,
    compute_spring_strains,
    compute_strain_energy,
    generate_rectangular_grid,
    leapfrog_step,
)
from kevlargrid.solver.projectile import (
    Projectile,
    distribute_contact_forces,
    update_contact_zone,
)


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
        self.active_thread = threading.Thread(target=self._run_loop, args=(config,), daemon=True)
        self.active_thread.start()

    def pause(self) -> None:
        """Pause worker thread execution."""
        with self.lock:
            if self.state == "running":
                self.state = "paused"
                self.pause_event.set()

    def resume(self) -> None:
        """Resume worker thread execution."""
        with self.lock:
            if self.state == "paused":
                self.state = "running"
                self.pause_event.clear()

    def stop(self) -> None:
        """Forcibly terminate background execution."""
        self.stop_event.set()
        self.pause_event.clear()
        if self.active_thread and self.active_thread.is_alive():
            self.active_thread.join(timeout=1.0)
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

            # Build boundary mask
            boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
            n_nodes_per_layer = nx * ny
            if grid_cfg["boundary_type"] == "fixed":
                for ply in range(n_plies):
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
                    "total": initial_ke,
                    "failed_count": 0,
                }
            )

            # Peak deceleration recording
            accel_history = []
            damp_dissipated = 0.0

            # Dynamic loop
            while t_sim < duration:
                # Check stop signal
                if self.stop_event.is_set():
                    break

                # Check pause signal
                if self.pause_event.is_set():
                    self.pause_event.wait(timeout=0.1)
                    last_time = time.time()  # reset speed calc
                    continue

                # 1. Projectile Contact Forces (on layer 0)
                update_contact_zone(proj, grid, proximity_threshold=dx * 2.0, positions=positions)
                proj_forces = distribute_contact_forces(
                    proj, grid, positions=positions, k_contact=k_penalty
                )

                # 2. Inter-ply Contact Forces (Checkout Mode)
                interply_forces = np.zeros_like(positions)
                interply_energy = 0.0
                if n_plies > 1 and t_ply is not None:
                    interply_forces, interply_energy = compute_interply_contact_forces(
                        positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
                    )

                # 3. Internal Spring Forces
                spring_forces = compute_spring_forces(
                    positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
                )

                # 4. Viscous Damping Forces & Energy Dissipation calculation
                damp_forces = -damping_coeff * velocities
                # Dissipated energy power: P_d = F_d . v
                p_damp = np.sum(damp_forces * velocities)
                damp_dissipated += float(-p_damp * dt)

                # Net integration
                net_forces = spring_forces + proj_forces + interply_forces + damp_forces

                # Clamped boundary constraints
                net_forces[boundary_mask] = 0.0
                velocities[boundary_mask] = 0.0

                # Integrate node dynamics
                positions, velocities = leapfrog_step(
                    positions, velocities, net_forces, grid.masses, dt
                )

                # Integrate projectile kinematics
                proj_reaction_force = -np.sum(proj_forces, axis=0)
                proj_accel = proj_reaction_force / proj.mass
                proj.velocity += proj_accel * dt
                proj.position += proj.velocity * dt
                accel_g = np.linalg.norm(proj_accel) / 9.81
                accel_history.append(accel_g)

                # Check spring failures
                strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
                check_failures(strains, grid.failed, mat["failure_strain"])

                t_sim += dt
                step += 1

                # Calculate metrics
                ke = compute_kinetic_energy(velocities, grid.masses)
                se = compute_strain_energy(
                    strains, grid.stiffnesses, grid.rest_lengths, grid.failed
                )
                failed_count = int(np.sum(grid.failed))

                # Store dynamic playback frames every 10 steps (high fidelity)
                if step % 10 == 0:
                    self.history.append(
                        {
                            "time": t_sim,
                            "nodes": positions.copy(),
                            "failed": grid.failed.copy(),
                            "projectile_pos": proj.position.copy(),
                            "ke": ke,
                            "se": se,
                            "damped": damp_dissipated,
                            "contact": interply_energy,
                            "total": ke + se + damp_dissipated + interply_energy,
                            "failed_count": failed_count,
                        }
                    )

                # Update live telemetry state protected by lock
                if step % 50 == 0 or t_sim >= duration:
                    now = time.time()
                    elapsed_real = now - last_time
                    if elapsed_real >= 0.1:
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

            # Find maximum layer perforated
            max_ply_perf = -1
            if n_plies > 1:
                spring_layers = grid.springs[:, 0] // n_nodes_per_layer
                for ply in range(n_plies):
                    ply_springs = spring_layers == ply
                    ply_failed = grid.failed[ply_springs]
                    if (
                        len(ply_failed) > 0 and np.mean(ply_failed) > 0.70
                    ):  # 70% threshold is complete yarn failure
                        max_ply_perf = ply

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
            traceback.print_exc()
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

        # Redraw blank viewport
        blank_grid = generate_rectangular_grid(11, 11, 0.01, config_panel.get_config()["material"])
        viewport3d.reset(blank_grid)

    # Bind controls callbacks
    controls.start_callback = lambda: (
        _on_resume_btn() if runner.get_telemetry()["state"] == "paused" else _on_start_btn()
    )
    controls.pause_callback = _on_pause_btn
    controls.stop_callback = _on_stop_btn
    controls.reset_callback = _on_reset_btn

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

    # Create dummy initial grid for screen renders
    initial_cfg = config_panel.get_config()
    init_grid = generate_rectangular_grid(11, 11, 0.01, initial_cfg["material"])
    viewport3d.reset(init_grid)

    while dpg.is_dearpygui_running():
        # Update metrics from solver thread
        tel = runner.get_telemetry()
        state = tel["state"]
        cfg = config_panel.get_config()

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
            # Live Plotting Traces Update
            strain_plot.update(
                tel["elapsed_time"], float(tel["se"] * 0.0001)
            )  # approximate peak strain
            energy_plot.update(
                tel["elapsed_time"],
                {
                    "kinetic": tel["ke"],
                    "strain": tel["se"],
                    "damped": tel["elapsed_time"]
                    * tel["failed_count"]
                    * 0.01,  # approximate damping
                    "contact": 0.0,
                    "total": tel["ke"] + tel["se"],
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
