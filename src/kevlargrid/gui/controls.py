"""Simulation controls and performance metrics telemetry widget.

Provides play / pause / stop / reset buttons and progress indicators
for monitoring dynamic explicitly integrated simulations.
"""

from __future__ import annotations

from collections.abc import Callable

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class SimulationControls:
    """DearPyGui widget group for simulation run controls and performance telemetry."""

    def __init__(self) -> None:
        self.group_tag = "controls_group"

        # Button tags
        self.start_btn = "controls_start"
        self.pause_btn = "controls_pause"
        self.stop_btn = "controls_stop"
        self.reset_btn = "controls_reset"

        # Progress & Telemetry tags
        self.progress_bar = "controls_progress_bar"
        self.text_time = "telemetry_text_time"
        self.text_steps = "telemetry_text_steps"
        self.text_speed = "telemetry_text_speed"
        self.text_eta = "telemetry_text_eta"
        self.text_failed = "telemetry_text_failed"
        self.text_energy = "telemetry_text_energy"

        # Button callbacks
        self.start_callback: Callable[[], None] | None = None
        self.pause_callback: Callable[[], None] | None = None
        self.stop_callback: Callable[[], None] | None = None
        self.reset_callback: Callable[[], None] | None = None

    def build(self) -> None:
        """Construct the control buttons, progress bar, and metrics labels."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, border=True, height=270):
            dpg.add_text("Simulation Controller", color=[0, 191, 255])
            dpg.add_separator()

            # Color-coded action buttons row
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Start",
                    tag=self.start_btn,
                    width=75,
                    callback=self._on_start,
                )
                dpg.add_button(
                    label="Pause",
                    tag=self.pause_btn,
                    width=75,
                    callback=self._on_pause,
                    enabled=False,
                )
                dpg.add_button(
                    label="Stop",
                    tag=self.stop_btn,
                    width=75,
                    callback=self._on_stop,
                    enabled=False,
                )
                dpg.add_button(
                    label="Reset",
                    tag=self.reset_btn,
                    width=75,
                    callback=self._on_reset,
                )

            # Apply elegant color styling to buttons
            # Start = Green
            dpg.bind_item_theme(
                self.start_btn, self._create_btn_theme([46, 139, 87], [60, 179, 113])
            )
            # Pause = Yellow
            dpg.bind_item_theme(
                self.pause_btn, self._create_btn_theme([218, 165, 32], [238, 220, 130])
            )
            # Stop = Red
            dpg.bind_item_theme(self.stop_btn, self._create_btn_theme([178, 34, 34], [220, 20, 60]))
            # Reset = Blue
            dpg.bind_item_theme(
                self.reset_btn, self._create_btn_theme([70, 130, 180], [100, 149, 237])
            )

            dpg.add_spacer(height=5)
            dpg.add_text("Simulation Progress:")
            dpg.add_progress_bar(tag=self.progress_bar, default_value=0.0, width=-1, overlay="0.0%")

            dpg.add_separator()
            dpg.add_text("Performance Telemetry", color=[0, 191, 255])

            with dpg.group(horizontal=True):
                with dpg.group(width=160):
                    dpg.add_text("Time (ms): 0.000 / 0.000", tag=self.text_time)
                    dpg.add_text("Step: 0", tag=self.text_steps)
                    dpg.add_text("Ruptured Springs: 0", tag=self.text_failed, color=[255, 69, 0])
                with dpg.group():
                    dpg.add_text("Speed (steps/s): 0.0", tag=self.text_speed)
                    dpg.add_text("ETA: 0.00s", tag=self.text_eta)
                    dpg.add_text("Energy: KE=0.0, SE=0.0 J", tag=self.text_energy)

    def _create_btn_theme(self, normal: list[int], hover: list[int]) -> str | int:
        """Create a theme wrapper for custom styled buttons."""
        with dpg.theme() as theme, dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, normal)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, hover)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, normal)
        return theme  # type: ignore[no-any-return]

    def _on_start(self, sender: str, app_data: str) -> None:
        if self.start_callback:
            self.start_callback()

    def _on_pause(self, sender: str, app_data: str) -> None:
        if self.pause_callback:
            self.pause_callback()

    def _on_stop(self, sender: str, app_data: str) -> None:
        if self.stop_callback:
            self.stop_callback()

    def _on_reset(self, sender: str, app_data: str) -> None:
        if self.reset_callback:
            self.reset_callback()

    def set_button_states(self, state: str) -> None:
        """Dynamically lock or unlock buttons based on simulation state.

        Parameters
        ----------
        state : str
            One of 'idle', 'running', 'paused', or 'completed'.
        """
        if dpg is None:  # pragma: no cover
            return

        if state == "running":
            dpg.configure_item(self.start_btn, enabled=False)
            dpg.configure_item(self.pause_btn, enabled=True)
            dpg.configure_item(self.stop_btn, enabled=True)
            dpg.configure_item(self.reset_btn, enabled=False)
        elif state == "paused":
            dpg.configure_item(self.start_btn, label="Resume", enabled=True)
            dpg.configure_item(self.pause_btn, enabled=False)
            dpg.configure_item(self.stop_btn, enabled=True)
            dpg.configure_item(self.reset_btn, enabled=False)
        elif state == "completed":
            dpg.configure_item(self.start_btn, label="Start", enabled=False)
            dpg.configure_item(self.pause_btn, enabled=False)
            dpg.configure_item(self.stop_btn, enabled=False)
            dpg.configure_item(self.reset_btn, enabled=True)
        else:  # idle
            dpg.configure_item(self.start_btn, label="Start", enabled=True)
            dpg.configure_item(self.pause_btn, enabled=False)
            dpg.configure_item(self.stop_btn, enabled=False)
            dpg.configure_item(self.reset_btn, enabled=True)

    def update_telemetry(
        self,
        elapsed_time: float,
        duration: float,
        step: int,
        speed: float,
        eta: float,
        failed_count: int,
        ke: float,
        se: float,
    ) -> None:
        """Update progress bar and text metrics values dynamically.

        Parameters
        ----------
        elapsed_time : float
            Current simulated time (seconds).
        duration : float
            Total simulation duration limit (seconds).
        step : int
            Completed simulation integration steps.
        speed : float
            Integration speed (steps per second).
        eta : float
            Estimated clock time remaining to complete the run (seconds).
        failed_count : int
            Total ruptured springs.
        ke : float
            Current total fabric kinetic energy (Joules).
        se : float
            Current total elastic strain energy (Joules).
        """
        if dpg is None:  # pragma: no cover
            return

        # 1. Update Progress Bar
        pct = (elapsed_time / duration) if duration > 0.0 else 0.0
        pct = min(1.0, max(0.0, pct))
        dpg.set_value(self.progress_bar, pct)
        dpg.configure_item(self.progress_bar, overlay=f"{pct * 100:.1f}%")

        # 2. Update Text Fields
        dpg.set_value(
            self.text_time, f"Time (ms): {elapsed_time * 1000:.3f} / {duration * 1000:.3f}"
        )
        dpg.set_value(self.text_steps, f"Step: {step}")
        dpg.set_value(self.text_failed, f"Ruptured Springs: {failed_count}")
        dpg.set_value(self.text_speed, f"Speed (steps/s): {speed:.1f}")
        dpg.set_value(self.text_eta, f"ETA: {eta:.2f}s" if eta > 0.0 else "ETA: --")
        dpg.set_value(self.text_energy, f"Energy: KE={ke:.2f}, SE={se:.2f} J")
