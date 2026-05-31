"""Main DearPyGui application launch and background thread simulation coordinator.

Provides multi-threaded execution management for the explicit solver
and handles UI state rendering, JSON save/load dialogues, auto-saves, and crash recovery.
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
    """Thread-safe background simulation runner."""

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
        """Restore initial telemetry variables."""
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

            # Rayleigh stiffness equivalent for viscous damping
            damping_coeff = sim_cfg["damping_coefficient"]
            duration = sim_cfg["duration"]

            # Loop tracking
            step = 0
            t_sim = 0.0
            last_time = time.time()
            last_step = 0

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
                if n_plies > 1 and t_ply is not None:
                    interply_forces, _ = compute_interply_contact_forces(
                        positions, n_nodes_per_layer, n_plies, t_ply, k_penalty
                    )

                # 3. Internal Spring Forces
                spring_forces = compute_spring_forces(
                    positions, grid.springs, grid.stiffnesses, grid.rest_lengths, grid.failed
                )

                # 4. Viscous Damping Forces
                damp_forces = -damping_coeff * velocities

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

                # Check spring failures
                strains = compute_spring_strains(positions, grid.springs, grid.rest_lengths)
                check_failures(strains, grid.failed, mat["failure_strain"])

                t_sim += dt
                step += 1

                # Update live telemetry every 50 steps
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

                    # Calculate energies
                    ke = compute_kinetic_energy(velocities, grid.masses)
                    se = compute_strain_energy(
                        strains, grid.stiffnesses, grid.rest_lengths, grid.failed
                    )
                    failed_count = int(np.sum(grid.failed))

                    with self.lock:
                        self.elapsed_time = t_sim
                        self.step = step
                        self.speed = steps_per_sec
                        self.eta = eta
                        self.ke = ke
                        self.se = se
                        self.failed_count = failed_count

                # Projectile arrest termination check
                if proj.velocity[2] <= 0.0:
                    break

            with self.lock:
                self.state = "completed"

        except Exception as e:
            traceback.print_exc()
            with self.lock:
                self.state = "error"
                self.error_message = str(e)


# Global instances
runner = SimRunner()
config_panel = ConfigPanel()
controls = SimulationControls()
last_autosave_time = 0.0
AUTOSAVE_DIR = ".autosave"
AUTOSAVE_PATH = f"{AUTOSAVE_DIR}/session.json"


def launch() -> None:
    """Initialise and launch the KevlarGrid GUI application."""
    if dpg is None:
        print("Error: DearPyGui is not installed. GUI cannot be launched.")
        return

    dpg.create_context()
    dpg.create_viewport(title="KevlarGrid Explicit Dynamic Solver v2.0", width=1280, height=800)

    # File Menu callbacks
    def _menu_save_config():
        config = config_panel.get_config()
        # Mock file dialog for simplified integration
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
            height=740,
            no_title_bar=True,
            no_move=True,
            no_resize=True,
        ),
        dpg.group(horizontal=True),
    ):
        # Left Panel: Sidebar Config
        config_panel.build()

        dpg.add_spacer(width=10)

        # Right Panel: Simulation telemetry & controls (and Sprint 5 plotting space)
        with dpg.group():
            controls.build()

            # Visual placeholders / info card
            with dpg.child_window(border=True, height=430):
                dpg.add_text("Woven Kevlar Dynamic Response Viewport", color=[0, 191, 255])
                dpg.add_separator()
                dpg.add_spacer(height=20)
                dpg.add_text(
                    "Explicit Finite Element Central-Difference Integration Engine",
                    color=[100, 149, 237],
                )
                dpg.add_text(
                    "Dynamic 3D interactive viewport rendering will be built in Sprint 5.",
                    color=[128, 128, 128],
                )
                dpg.add_spacer(height=40)
                dpg.add_text("System Diagnostics Ready.")

    # Core Thread-Safe Control Callbacks
    def _on_start_btn():
        cfg = config_panel.get_config()
        try:
            validate_config(cfg)
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

    def _on_reset_btn():
        runner.reset()
        controls.set_button_states("idle")
        # Sync progress bars
        controls.update_telemetry(0.0, 0.001, 0, 0.0, 0.0, 0, 0.0, 0.0)

    # Bind callbacks to control panel
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
            # Spawn a restore session modal popup request
            _spawn_recovery_modal(cached_config)
        except Exception:
            pass

    # Custom frames rendering loop to manage background synchronization
    global last_autosave_time
    last_autosave_time = time.time()

    while dpg.is_dearpygui_running():
        # Update metrics from solver thread
        tel = runner.get_telemetry()
        state = tel["state"]

        if state == "running":
            cfg = config_panel.get_config()
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
        elif state == "completed":
            controls.set_button_states("completed")
            cfg = config_panel.get_config()
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
                # Create .autosave directory if missing
                os.makedirs(AUTOSAVE_DIR, exist_ok=True)
                save_config(active_cfg, AUTOSAVE_PATH)
            except Exception:
                pass

        dpg.render_dearpygui_frame()

    # Cleanup active solver thread on exit
    runner.stop()
    dpg.destroy_context()


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
