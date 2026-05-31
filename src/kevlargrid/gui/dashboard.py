"""Simulation results summary telemetry dashboard.

Provides detailed impact reports, deceleration forces, ruptured springs
averages, energy dissipation rates, and color-coded PASS/FAIL status indicators.
"""

from __future__ import annotations

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

    def build(self) -> None:
        """Construct the results dashboard DearPyGui widgets."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, border=True, height=270, width=-1):
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
