"""Simulation results summary telemetry dashboard.

Provides detailed impact reports, deceleration forces, ruptured springs
averages, energy dissipation rates, and color-coded PASS/FAIL status indicators.
"""

from __future__ import annotations

from typing import Any

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class ResultsDashboard:
    """Post-simulation dynamic results summary panel."""

    def __init__(self) -> None:
        self.group_tag = "dashboard_group"
        self.status_badge = "dashboard_status_badge"

        # Telemetry variables tags
        self.val_status = "dash_val_status"
        self.val_max_decel = "dash_val_max_decel"
        self.val_rupture_pct = "dash_val_rupture_pct"
        self.val_residual_vel = "dash_val_residual_vel"
        self.val_energy_loss = "dash_val_energy_loss"
        self.val_penetration_ply = "dash_val_penetration_ply"

        # References placeholders
        self.runner: Any = None
        self.config_panel: Any = None

    def set_references(self, runner: Any, config_panel: Any, viewport3d: Any = None) -> None:
        """Connect global thread runner and config panel to enable live data exporting."""
        self.runner = runner
        self.config_panel = config_panel
        self.viewport3d = viewport3d

    def build(self) -> None:
        """Construct the results dashboard DearPyGui widgets."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, border=True, height=410, width=-1):
            dpg.add_text("Impact Summary Dashboard", color=[0, 191, 255])
            dpg.add_separator()

            # Dynamic PASS / FAIL status banner
            with dpg.group(horizontal=True):
                dpg.add_text("System Status: ")
                dpg.add_text("Simulation Idle", tag=self.status_badge, color=[200, 200, 200])

            dpg.add_spacer(height=10)

            # High-fidelity tabular dashboard layout
            with dpg.table(
                header_row=True,
                borders_innerH=True,
                borders_outerH=True,
                borders_innerV=True,
                borders_outerV=True,
            ):
                dpg.add_table_column(label="Key Impact Metric")
                dpg.add_table_column(label="Recorded Value")

                # Row 1: Arrest Status
                with dpg.table_row():
                    dpg.add_text("Arrest Outcome")
                    dpg.add_text("Pending simulation run...", tag=self.val_status)

                # Row 2: Peak deceleration force
                with dpg.table_row():
                    dpg.add_text("Peak Deceleration Force (G's)")
                    dpg.add_text("0.0 G", tag=self.val_max_decel)

                # Row 3: Yarn Spring Rupture Percentage
                with dpg.table_row():
                    dpg.add_text("Broken Yarn Spring Percentage")
                    dpg.add_text("0.0%", tag=self.val_rupture_pct)

                # Row 4: Residual Strike Velocity
                with dpg.table_row():
                    dpg.add_text("Final/Residual Velocity (m/s)")
                    dpg.add_text("0.0 m/s", tag=self.val_residual_vel)

                # Row 5: Energy Absorbed Rate
                with dpg.table_row():
                    dpg.add_text("Energy Dissipated Efficiency")
                    dpg.add_text("0.0%", tag=self.val_energy_loss)

                # Row 6: Per-Layer Perforation Index (Checkout Mode)
                with dpg.table_row():
                    dpg.add_text("Max Ply Perforated Layer")
                    dpg.add_text("Layer --", tag=self.val_penetration_ply)

            dpg.add_spacer(height=10)

            with dpg.collapsing_header(label="6-DOF Projectile State Telemetry", default_open=True):
                with dpg.table(
                    header_row=True,
                    borders_innerH=True,
                    borders_outerH=True,
                    borders_innerV=True,
                    borders_outerV=True,
                ):
                    dpg.add_table_column(label="State Property")
                    dpg.add_table_column(label="Value")

                    with dpg.table_row():
                        dpg.add_text("Shape Profile")
                        dpg.add_text("Pending...", tag="dash_val_proj_shape")
                    with dpg.table_row():
                        dpg.add_text("Calculated Volume")
                        dpg.add_text("0.0", tag="dash_val_proj_vol")
                    with dpg.table_row():
                        dpg.add_text("Principal Inertia (Ixx, Iyy, Izz)")
                        dpg.add_text("0.0, 0.0, 0.0", tag="dash_val_proj_inertia")
                    with dpg.table_row():
                        dpg.add_text("Final Velocity (Vx, Vy, Vz)")
                        dpg.add_text("0.0, 0.0, 0.0", tag="dash_val_proj_vel")
                    with dpg.table_row():
                        dpg.add_text("Final Angular Velocity (rad/s)")
                        dpg.add_text("0.0, 0.0, 0.0", tag="dash_val_proj_omega")
                    with dpg.table_row():
                        dpg.add_text("Final Quaternion (qw, qx, qy, qz)")
                        dpg.add_text("1.0, 0.0, 0.0, 0.0", tag="dash_val_proj_quat")

            dpg.add_spacer(height=15)
            dpg.add_text("Actionable Exports & Reporting:", color=[0, 191, 255])
            dpg.add_separator()
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Export HDF5 Archive",
                    callback=self._export_hdf5,
                    width=180,
                    tag="btn_export_hdf5",
                    enabled=False,
                )
                dpg.add_button(
                    label="Export CSV Telemetry",
                    callback=self._export_csv,
                    width=180,
                    tag="btn_export_csv",
                    enabled=False,
                )
                dpg.add_button(
                    label="Compile Video Animation",
                    callback=self._compile_video,
                    width=180,
                    tag="btn_export_video",
                    enabled=False,
                )
                dpg.add_button(
                    label="Generate PDF Report",
                    callback=self._generate_pdf,
                    width=180,
                    tag="btn_export_pdf",
                    enabled=False,
                )

    def populate(self, results: dict) -> None:
        """Fill dashboard fields with structural parameters from completed dynamic runs.

        Parameters
        ----------
        results : dict
            Generated dynamic impact report summary.
        """
        if dpg is None:  # pragma: no cover
            return

        is_arrested = results.get("arrested", False)

        # 1. Update Bold Badge Banner
        if is_arrested:
            dpg.configure_item(
                self.status_badge, label="ARRESTED - PASS", color=[50, 205, 50]
            )  # Bold Green
            dpg.set_value(self.val_status, "Arrested Successfully")
            dpg.configure_item(self.val_status, color=[50, 205, 50])
        else:
            dpg.configure_item(
                self.status_badge, label="PERFORATED - FAIL", color=[220, 20, 60]
            )  # Bold Red
            dpg.set_value(self.val_status, "Fabric Perforated / Failure")
            dpg.configure_item(self.val_status, color=[220, 20, 60])

        # 2. Update Table values
        max_decel = results.get("peak_deceleration_g", 0.0)
        dpg.set_value(self.val_max_decel, f"{max_decel:.1f} G")

        rupture_pct = results.get("yarn_rupture_percentage", 0.0)
        dpg.set_value(self.val_rupture_pct, f"{rupture_pct:.2f}%")

        res_vel = results.get("residual_velocity_ms", 0.0)
        dpg.set_value(self.val_residual_vel, f"{res_vel:.1f} m/s")

        energy_loss = results.get("energy_dissipation_efficiency", 0.0)
        dpg.set_value(self.val_energy_loss, f"{energy_loss * 100:.2f}%")

        max_ply = results.get("max_layer_perforated", -1)
        if max_ply == -1:
            dpg.set_value(self.val_penetration_ply, "None (Arrested Layer 0)")
        else:
            dpg.set_value(self.val_penetration_ply, f"Layer {max_ply}")

        # Update 6-DOF Projectile Table values
        proj_shape = results.get("projectile_shape", "Box").capitalize()
        dpg.set_value("dash_val_proj_shape", proj_shape)

        proj_vol = results.get("projectile_volume", 0.0)
        dpg.set_value("dash_val_proj_vol", f"{proj_vol:.3e} m^3")

        proj_inertia = results.get("projectile_inertia", [0.0, 0.0, 0.0])
        dpg.set_value(
            "dash_val_proj_inertia",
            f"[{proj_inertia[0]:.3e}, {proj_inertia[1]:.3e}, {proj_inertia[2]:.3e}] kg*m^2",
        )

        proj_vel_final = results.get("projectile_velocity_final", [0.0, 0.0, 0.0])
        dpg.set_value(
            "dash_val_proj_vel",
            f"[{proj_vel_final[0]:.2f}, {proj_vel_final[1]:.2f}, {proj_vel_final[2]:.2f}] m/s",
        )

        proj_omega_final = results.get("projectile_omega_final", [0.0, 0.0, 0.0])
        dpg.set_value(
            "dash_val_proj_omega",
            f"[{proj_omega_final[0]:.2f}, {proj_omega_final[1]:.2f}, {proj_omega_final[2]:.2f}] rad/s",
        )

        proj_quat_final = results.get("projectile_quat_final", [1.0, 0.0, 0.0, 0.0])
        dpg.set_value(
            "dash_val_proj_quat",
            f"[{proj_quat_final[0]:.4f}, {proj_quat_final[1]:.4f}, {proj_quat_final[2]:.4f}, {proj_quat_final[3]:.4f}]",
        )

        # Enable actionable export buttons
        dpg.configure_item("btn_export_hdf5", enabled=True)
        dpg.configure_item("btn_export_csv", enabled=True)
        dpg.configure_item("btn_export_video", enabled=True)
        dpg.configure_item("btn_export_pdf", enabled=True)

    def _export_hdf5(self, sender: str, app_data: Any) -> None:
        import os

        from kevlargrid.io.export.h5_writer import export_to_h5

        os.makedirs("exports", exist_ok=True)
        filepath = "exports/simulation_archive.h5"

        try:
            cfg = self.config_panel.get_config()
            results = self.runner.get_telemetry()["results_report"]
            history = self.runner.history

            export_to_h5(cfg, results, history, filepath)
            self._show_modal(
                "Export Success", f"HDF5 Trajectory successfully archived to:\n{filepath}"
            )
        except Exception as e:
            self._show_modal("Export Error", f"Failed to export HDF5:\n{e}")

    def _export_csv(self, sender: str, app_data: Any) -> None:
        import os

        from kevlargrid.io.export.csv_writer import export_to_csv

        os.makedirs("exports", exist_ok=True)
        filepath = "exports/telemetry.csv"

        try:
            history = self.runner.history
            export_to_csv(history, filepath)
            self._show_modal("Export Success", f"CSV Summary successfully exported to:\n{filepath}")
        except Exception as e:
            self._show_modal("Export Error", f"Failed to export CSV:\n{e}")

    def _compile_video(self, sender: str, app_data: Any) -> None:
        import math
        import os
        import threading

        from kevlargrid.io.export.video_exporter import VideoExporter

        os.makedirs("exports", exist_ok=True)
        filepath = "exports/animation.mp4"

        self._show_modal(
            "Compiling Animation",
            "Rendering 3D frames in background thread...\nPlease wait.",
            has_button=False,
        )

        current_yaw = 45.0
        current_pitch = 30.0
        current_distance = None
        current_pan_x = None
        current_pan_y = None

        if hasattr(self, "viewport3d") and self.viewport3d is not None:
            current_yaw = float(math.degrees(self.viewport3d.yaw))
            current_pitch = float(math.degrees(self.viewport3d.pitch))
            current_distance = float(self.viewport3d.distance)
            current_pan_x = float(self.viewport3d.pan_x)
            current_pan_y = float(self.viewport3d.pan_y)

        def _thread_task():
            try:
                cfg = self.config_panel.get_config()
                history = self.runner.history

                nx = cfg["grid"]["nx"]
                ny = cfg["grid"]["ny"]
                n_plies = cfg["grid"]["n_plies"]

                exporter = VideoExporter(
                    config=cfg,
                    history=history,
                    nx=nx,
                    ny=ny,
                    n_plies=n_plies,
                    n_nodes_per_layer=nx * ny,
                )
                exporter.compile(
                    filepath,
                    yaw=current_yaw,
                    pitch=current_pitch,
                    fps=30,
                    distance=current_distance,
                    pan_x=current_pan_x,
                    pan_y=current_pan_y,
                )

                if dpg.does_item_exist("popup_modal_dashboard_message"):
                    dpg.delete_item("popup_modal_dashboard_message")
                self._show_modal(
                    "Compile Success", f"3D slowed-down animation compiled to:\n{filepath}"
                )
            except Exception as e:
                if dpg.does_item_exist("popup_modal_dashboard_message"):
                    dpg.delete_item("popup_modal_dashboard_message")
                self._show_modal("Compile Error", f"Animation compiler failed:\n{e}")

        threading.Thread(target=_thread_task, daemon=True).start()

    def _generate_pdf(self, sender: str, app_data: Any) -> None:
        import os

        from kevlargrid.io.export.report_builder import generate_pdf_report

        os.makedirs("exports", exist_ok=True)
        filepath = "exports/executive_report.pdf"

        try:
            cfg = self.config_panel.get_config()
            results = self.runner.get_telemetry()["results_report"]
            history = self.runner.history

            try:
                generate_pdf_report(cfg, results, history, filepath)
                self._show_modal(
                    "Report Success", f"Print-ready PDF Summary compiled to:\n{filepath}"
                )
            except ImportError as ie:
                self._show_modal("Report Success (HTML Fallback)", str(ie))
        except Exception as e:
            self._show_modal("Report Error", f"Report generation failed:\n{e}")

    def _show_modal(self, title: str, message: str, has_button: bool = True) -> None:
        """Utility dialog overlay box wrapper."""
        if dpg is None:
            return

        modal_tag = "popup_modal_dashboard_message"
        if dpg.does_item_exist(modal_tag):
            dpg.delete_item(modal_tag)

        with dpg.window(
            label=title,
            tag=modal_tag,
            modal=True,
            show=True,
            width=380,
            height=160,
            no_resize=True,
            no_move=True,
        ):
            dpg.add_text(message)
            dpg.add_spacer(height=10)
            if has_button:
                dpg.add_button(label="OK", width=75, callback=lambda: dpg.delete_item(modal_tag))
