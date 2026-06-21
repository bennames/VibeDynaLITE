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
    generate_rectangular_grid,
)
from kevlargrid.utils import get_logger

logger = get_logger("gui.app")


class SimRunner:
    """Thread-safe background simulation runner with history tracking."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_thread: threading.Thread | None = None
        self.active_process: Any | None = None
        self.control_pipe_parent: Any | None = None
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
        self.failure_dissipated = 0.0
        self.clamp_dissipated = 0.0
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
            self.failure_dissipated = 0.0
            self.clamp_dissipated = 0.0
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
                "failure_dissipated": self.failure_dissipated,
                "clamp_dissipated": self.clamp_dissipated,
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
        import multiprocessing
        import traceback

        from kevlargrid.solver.worker import run_solver_process

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        parent_conn, child_conn = ctx.Pipe()

        self.control_pipe_parent = parent_conn

        # Pre-initialize grid structure in parent process for GUI placeholders
        try:
            mat = config["material"]
            grid_cfg = config["grid"]
            nx, ny, dx = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["dx"]
            n_plies = grid_cfg["n_plies"]
            t_ply = grid_cfg["t_ply"]
            # Build parent-side grid placeholder
            grid = generate_rectangular_grid(
                nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
            )
            with self.lock:
                self.grid_nodes = grid.nodes.copy()
                self.grid_failed = grid.failed.copy()
                self.projectile_pos = np.array(config["projectile"]["position"], dtype=np.float64)
        except Exception as e:
            with self.lock:
                self.state = "error"
                self.error_message = f"Grid generation error: {e}"
            return

        dt_sim = 1e-6

        # Start the solver process
        p = ctx.Process(target=run_solver_process, args=(config, queue, child_conn), daemon=True)
        self.active_process = p
        p.start()
        logger.info("Solver subprocess started with PID %s.", p.pid)

        last_time = time.time()
        last_step = 0
        t_sim_total = config["simulation"]["duration"]

        try:
            while p.is_alive() or not queue.empty():
                # Check if user requested stop via DPG button
                if self.stop_event.is_set():
                    parent_conn.send("stop")
                    p.terminate()
                    break

                # Check if user requested pause/resume
                if self.pause_event.is_set():
                    # Send pause if not already sent
                    with self.lock:
                        if self.state == "running":
                            self.state = "paused"
                            parent_conn.send("pause")
                else:
                    with self.lock:
                        if self.state == "paused":
                            self.state = "running"
                            parent_conn.send("resume")

                # Read update from queue
                try:
                    # Timeout so we check stop/pause signals frequently
                    msg = queue.get(timeout=0.05)
                except Exception:
                    continue

                if msg["type"] == "init":
                    logger.info(
                        "Solver subprocess initialized. Device: %s, Nodes: %d, Springs: %d",
                        msg["device"],
                        msg["n_nodes"],
                        msg["n_springs"],
                    )
                    dt_sim = msg["dt"]
                elif msg["type"] == "telemetry":
                    # Extract metric updates
                    with self.lock:
                        prev_damp = self.damp_dissipated
                        prev_failure = self.failure_dissipated
                        prev_clamp = self.clamp_dissipated

                        self.elapsed_time = msg["t_sim"]
                        self.step += msg["steps"]
                        self.ke = msg["ke"]
                        self.se = msg["se"]
                        self.damp_dissipated = msg["damp_dissipated"]
                        self.failure_dissipated = msg["failure_dissipated"]
                        self.clamp_dissipated = msg["clamp_dissipated"]
                        self.peak_strain = msg["peak_strain"]
                        self.proj_ke = msg["proj_ke"]
                        self.failed_count = msg["failed_count"]
                        self.grid_nodes = msg["positions"]
                        self.grid_failed = msg["failed"]
                        self.projectile_pos = msg["projectile_pos"]

                        # Append each frame in chunk history to self.history
                        hist_pos = msg["hist_pos"]
                        hist_failed = msg["hist_failed"]
                        hist_proj_pos = msg["hist_proj_pos"]
                        hist_time = msg["hist_time"]
                        hist_ke = msg["hist_ke"]
                        hist_se = msg["hist_se"]
                        hist_proj_ke = msg["hist_proj_ke"]
                        hist_peak_strain = msg.get("hist_peak_strain", None)

                        n_frames = len(hist_time)
                        for idx in range(n_frames):
                            # Interpolate cumulative dissipated energies linearly across the chunk S7.14
                            t_factor = (idx + 1) / n_frames if n_frames > 0 else 1.0
                            damp_val = prev_damp + (self.damp_dissipated - prev_damp) * t_factor
                            fail_val = prev_failure + (self.failure_dissipated - prev_failure) * t_factor
                            clamp_val = prev_clamp + (self.clamp_dissipated - prev_clamp) * t_factor

                            # Calculate conserved total energy including all components S7.14
                            tot_energy = (
                                hist_ke[idx]
                                + hist_se[idx]
                                + hist_proj_ke[idx]
                                + damp_val
                                + fail_val
                                + clamp_val
                            )

                            # Count failed springs at that step
                            failed_cnt_step = int(np.sum(hist_failed[idx]))

                            # Read accurate peak strain frame-by-frame
                            p_strain = float(hist_peak_strain[idx]) if hist_peak_strain is not None else float(self.peak_strain)

                            self.history.append(
                                {
                                    "time": float(hist_time[idx]),
                                    "nodes": hist_pos[idx],
                                    "failed": hist_failed[idx],
                                    "projectile_pos": hist_proj_pos[idx],
                                    "ke": float(hist_ke[idx]),
                                    "se": float(hist_se[idx]),
                                    "damped": float(damp_val),
                                    "failure_dissipated": float(fail_val),
                                    "clamp_dissipated": float(clamp_val),
                                    "contact": 0.0,
                                    "total": float(tot_energy),
                                    "failed_count": failed_cnt_step,
                                    "peak_strain": p_strain,
                                    "proj_ke": float(hist_proj_ke[idx]),
                                }
                            )

                        # Calculate running speed/eta metrics
                        now = time.time()
                        dt_real = now - last_time
                        if dt_real >= 1.0:
                            steps_done = self.step - last_step
                            self.speed = steps_done / dt_real
                            last_time = now
                            last_step = self.step
                            if self.speed > 0:
                                steps_left = (t_sim_total - self.elapsed_time) / dt_sim
                                self.eta = steps_left / self.speed
                            else:
                                self.eta = 0.0

                elif msg["type"] == "completed":
                    with self.lock:
                        self.state = "completed"
                        self.results_report = msg["report"]
                    logger.info("Solver subprocess completed successfully.")
                    break
                elif msg["type"] == "error":
                    with self.lock:
                        self.state = "error"
                        self.error_message = msg["message"]
                    logger.error(
                        "Solver subprocess crashed with error: %s\n%s",
                        msg["message"],
                        msg.get("traceback", ""),
                    )
                    break

        except Exception as e:
            logger.error("Error in runner listener thread: %s", traceback.format_exc())
            with self.lock:
                self.state = "error"
                self.error_message = f"Listener error: {e}"
        finally:
            # Cleanup process
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
            self.active_process = None
            self.control_pipe_parent = None


# Global instances
runner = SimRunner()
config_panel = ConfigPanel()
controls = SimulationControls()
strain_plot = StrainPlot()
energy_plot = EnergyPlot()
dashboard = ResultsDashboard()
viewport3d = Viewport3D()
dashboard.set_references(runner, config_panel, viewport3d)

# Playback controls state
playback_active = False
playback_frame_index = 0
playback_speed = 1.0  # multiplier (frames per update tick)
playback_last_tick = 0.0

last_autosave_time = 0.0
AUTOSAVE_DIR = ".autosave"
AUTOSAVE_PATH = f"{AUTOSAVE_DIR}/session.toml"


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
        save_config(config, "configs/saved_configuration.toml")
        _show_modal_message(
            "Config Saved", "Configuration successfully saved to:\nconfigs/saved_configuration.toml"
        )

    def _menu_load_config():
        if os.path.exists("configs/saved_configuration.toml"):
            config = load_config("configs/saved_configuration.toml")
            config_panel.set_config(config)
            _show_modal_message(
                "Config Loaded", "Configuration loaded from:\nconfigs/saved_configuration.toml"
            )
        elif os.path.exists("configs/saved_configuration.json"):
            # Backward compatibility check
            config = load_config("configs/saved_configuration.json")
            config_panel.set_config(config)
            _show_modal_message(
                "Config Loaded", "Configuration loaded from legacy JSON:\nconfigs/saved_configuration.json"
            )
        else:
            _show_modal_message(
                "Error", "No saved configuration file found in 'configs/saved_configuration.toml'."
            )

    # Main window viewport setup
    with dpg.viewport_menu_bar(), dpg.menu(label="File"):
        dpg.add_menu_item(label="Load Configuration", callback=_menu_load_config)
        dpg.add_menu_item(label="Save Configuration", callback=_menu_save_config)
        dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

    with (
        dpg.window(
            label="KevlarGrid Workspace",
            width=1300,
            height=760,
            no_title_bar=True,
            no_move=True,
            no_resize=False,
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

                # Tab 2: Dynamic 2D Telemetry Plots
                with (
                    dpg.tab(label="Dynamic Analytics Traces", tag="tab_plots"),
                    dpg.group(horizontal=True),
                ):
                    with dpg.group(width=410, tag="strain_plot_group"):
                        strain_plot.build()
                    with dpg.group(width=410, tag="energy_plot_group"):
                        energy_plot.build()

                # Tab 3: Dynamic Results Pass/Fail Dashboard
                with dpg.tab(label="Impact Results Summary", tag="tab_dashboard"):
                    dashboard.build()

            # Post-simulation playback widgets row toolbar (globally visible below the tabs)
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
                blade_width=cfg["projectile"]["blade_width"],
                edge_thickness=cfg["projectile"]["edge_thickness"],
            )
            viewport3d.draw_projectile(
                np.array(cfg["projectile"]["position"], dtype=np.float64),
                cfg["projectile"]["blade_width"],
                cfg["projectile"]["edge_thickness"],
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
        viewport3d.reset(
            blank_grid,
            blade_width=reset_cfg["projectile"]["blade_width"],
            edge_thickness=reset_cfg["projectile"]["edge_thickness"],
        )
        viewport3d.draw_projectile(
            np.array(reset_cfg["projectile"]["position"], dtype=np.float64),
            reset_cfg["projectile"]["blade_width"],
            reset_cfg["projectile"]["edge_thickness"],
        )

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

    _last_resize_size = (0, 0)
    _resize_pending = False
    _resize_time = 0.0
    _pending_right_width = 0
    _pending_tab_height = 0

    def _on_viewport_resize(sender, app_data):
        nonlocal _last_resize_size, _resize_pending, _resize_time, _pending_right_width, _pending_tab_height
        W_view, H_view = app_data[0], app_data[1]
        
        # Debounce identical resize calls S7.10
        if (W_view, H_view) == _last_resize_size:
            return
        _last_resize_size = (W_view, H_view)
        
        # Configure workspace window
        if dpg.does_item_exist("KevlarGrid Workspace"):
            dpg.configure_item("KevlarGrid Workspace", width=W_view, height=H_view - 40)
        
        left_width = int(W_view * 0.3)
        left_width = max(420, min(500, left_width))
        right_width = W_view - left_width - 30
        
        # Dynamically resize sidebar inputs, controls, tabs, and viewport
        if dpg.does_item_exist(config_panel.group_tag):
            dpg.configure_item(config_panel.group_tag, width=left_width, height=H_view - 60)
        if dpg.does_item_exist(controls.group_tag):
            dpg.configure_item(controls.group_tag, width=right_width)
        
        playback_visible = dpg.is_item_shown("playback_group") if dpg.does_item_exist("playback_group") else False
        playback_height = 40 if playback_visible else 0
        tab_height = H_view - 385 - playback_height
        
        if dpg.does_item_exist(viewport3d.group_tag):
            dpg.configure_item(viewport3d.group_tag, width=right_width, height=tab_height)
        if dpg.does_item_exist(dashboard.group_tag):
            dpg.configure_item(dashboard.group_tag, width=right_width, height=tab_height)
        
        # Resize plots and plot groups
        plot_width = int((right_width - 20) / 2)
        if dpg.does_item_exist("strain_plot_group"):
            dpg.configure_item("strain_plot_group", width=plot_width)
        if dpg.does_item_exist("energy_plot_group"):
            dpg.configure_item("energy_plot_group", width=plot_width)
        
        # Defer expensive viewport resize and plot height configuration S7.10
        _resize_pending = True
        _resize_time = time.time()
        _pending_right_width = right_width
        _pending_tab_height = tab_height

    dpg.set_viewport_resize_callback(_on_viewport_resize)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    # Force initial resize immediately AFTER show_viewport to sync dimensions correctly S7.10
    _on_viewport_resize(None, [1280, 830])
    _resize_pending = False
    if dpg.does_item_exist(strain_plot.plot_tag):
        dpg.configure_item(strain_plot.plot_tag, height=_pending_tab_height - 60)
    if dpg.does_item_exist(energy_plot.plot_tag):
        dpg.configure_item(energy_plot.plot_tag, height=_pending_tab_height - 60)
    viewport3d.resize(_pending_right_width - 20, _pending_tab_height - 60)

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
    viewport3d.reset(
        init_grid,
        blade_width=initial_cfg["projectile"]["blade_width"],
        edge_thickness=initial_cfg["projectile"]["edge_thickness"],
    )
    viewport3d.draw_projectile(
        np.array(initial_cfg["projectile"]["position"], dtype=np.float64),
        initial_cfg["projectile"]["blade_width"],
        initial_cfg["projectile"]["edge_thickness"],
    )

    last_grid_key = None

    while dpg.is_dearpygui_running():
        # Process pending deferred viewport and plot resize once settled S7.10
        if _resize_pending and (time.time() - _resize_time > 0.15):
            _resize_pending = False
            if dpg.does_item_exist(strain_plot.plot_tag):
                dpg.configure_item(strain_plot.plot_tag, height=_pending_tab_height - 60)
            if dpg.does_item_exist(energy_plot.plot_tag):
                dpg.configure_item(energy_plot.plot_tag, height=_pending_tab_height - 60)
            viewport3d.resize(_pending_right_width - 20, _pending_tab_height - 60)

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
                cfg["projectile"]["blade_width"],
                cfg["projectile"]["edge_thickness"],
                tuple(cfg["projectile"]["position"]),
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
                    viewport3d.reset(
                        preview_grid,
                        blade_width=cfg["projectile"]["blade_width"],
                        edge_thickness=cfg["projectile"]["edge_thickness"],
                    )
                    viewport3d.draw_projectile(
                        np.array(cfg["projectile"]["position"], dtype=np.float64),
                        cfg["projectile"]["blade_width"],
                        cfg["projectile"]["edge_thickness"],
                    )
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
                    "failure_dissipated": tel["failure_dissipated"],
                    "contact": 0.0,
                    "total": (
                        tel["ke"]
                        + tel["se"]
                        + tel["damp_dissipated"]
                        + tel["failure_dissipated"]
                        + tel["clamp_dissipated"]
                        + tel["proj_ke"]
                    ),
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

        # Deferred 3D viewport redraw processing to avoid event backlog S7.10
        if getattr(viewport3d, "_needs_redraw", False):
            viewport3d.redraw(force=True)
            viewport3d._needs_redraw = False

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


if __name__ == "__main__":
    launch()
