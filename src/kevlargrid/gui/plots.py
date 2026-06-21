"""Real-time plot widgets.

Provides Peak Strain vs Time and Multi-Series Energy Balance plots
that render dynamically during explicit time integration runs.
"""

from __future__ import annotations

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]


class StrainPlot:
    """Live peak-strain time-history plot widget."""

    def __init__(self) -> None:
        self.plot_tag = "strain_plot_widget"
        self.x_axis = "strain_plot_x"
        self.y_axis = "strain_plot_y"
        self.series_tag = "strain_series"
        self.threshold_tag = "strain_threshold_series"

        # Buffer lists
        self.x_data: list[float] = []
        self.y_data: list[float] = []
        self.threshold_val = 0.036
        self.marker_tag = "strain_playback_marker"
        self._update_count = 0

    def build(self) -> None:
        """Construct the DearPyGui peak strain plot."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.plot(
            label="Peak Fabric Yarn Strain History", tag=self.plot_tag, height=220, width=-1
        ):
            dpg.add_plot_legend()

            # X-Axis: Time in seconds
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=self.x_axis)

            # Y-Axis: Peak Strain
            with dpg.plot_axis(dpg.mvYAxis, label="Strain", tag=self.y_axis):
                # Peak strain history trace (emerald green)
                dpg.add_line_series(
                    self.x_data,
                    self.y_data,
                    label="Peak Strain",
                    tag=self.series_tag,
                )
                # Rupture limit line (red)
                dpg.add_line_series(
                    [0.0, 0.001],
                    [self.threshold_val, self.threshold_val],
                    label="Rupture Threshold",
                    tag=self.threshold_tag,
                )
                # Timeline playback marker dot S6.5.4
                dpg.add_scatter_series(
                    [],
                    [],
                    label="Current Time",
                    tag=self.marker_tag,
                )

        # Style the threshold line as a dashed red line
        with dpg.theme() as threshold_theme, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, [220, 20, 60])  # crimson red
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0)
        dpg.bind_item_theme(self.threshold_tag, threshold_theme)

        # Style the peak strain line
        with dpg.theme() as strain_theme, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, [0, 191, 255])  # deep sky blue
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0)
        dpg.bind_item_theme(self.series_tag, strain_theme)

        # Style the marker dot S6.5.4
        with dpg.theme() as marker_theme, dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, [255, 215, 0])  # Gold
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6.0)
        dpg.bind_item_theme(self.marker_tag, marker_theme)

    def reset(self, threshold: float = 0.036) -> None:
        """Clear all active data buffers and set the horizontal threshold line.

        Parameters
        ----------
        threshold : float, optional
            Selected material's failure strain threshold.
        """
        self.x_data.clear()
        self.y_data.clear()
        self.threshold_val = threshold
        self._update_count = 0

        if dpg is None:  # pragma: no cover
            return

        if dpg.does_item_exist(self.series_tag):
            dpg.set_value(self.series_tag, [[], []])
        if dpg.does_item_exist(self.threshold_tag):
            # Render a line extending from 0 to 0.005 seconds originally
            dpg.set_value(self.threshold_tag, [[0.0, 0.005], [threshold, threshold]])
        if dpg.does_item_exist(self.marker_tag):
            dpg.set_value(self.marker_tag, [[], []])

    def update(self, time: float, peak_strain: float) -> None:
        """Append a new data point and refresh the line series.

        Parameters
        ----------
        time : float
            Current simulated elapsed time (seconds).
        peak_strain : float
            Maximum engineering strain recorded across all springs.
        """
        self.x_data.append(time)
        self.y_data.append(peak_strain)

        if dpg is None:  # pragma: no cover
            return

        if dpg.does_item_exist(self.series_tag):
            self._update_count += 1
            if self._update_count == 1 or self._update_count % 4 == 0:
                x_plot = self.x_data
                y_plot = self.y_data
                n_points = len(x_plot)
                if n_points > 2000:
                    stride = n_points // 1000
                    x_plot = self.x_data[::stride]
                    y_plot = self.y_data[::stride]
                    if x_plot[-1] != self.x_data[-1]:
                        x_plot.append(self.x_data[-1])
                        y_plot.append(self.y_data[-1])

                dpg.set_value(self.series_tag, [x_plot, y_plot])

                if self._update_count == 1 or self._update_count % 12 == 0:
                    dpg.fit_axis_data(self.x_axis)
                    dpg.fit_axis_data(self.y_axis)

        # Sync threshold line length to exceed current time
        if dpg.does_item_exist(self.threshold_tag):
            max_t = max(0.002, time * 1.5)
            dpg.set_value(
                self.threshold_tag, [[0.0, max_t], [self.threshold_val, self.threshold_val]]
            )

    def set_playback_marker(self, time: float, strain: float) -> None:
        """Update the position of the playback marker dot S6.5.4."""
        if dpg is None or not dpg.does_item_exist(self.marker_tag):
            return
        dpg.set_value(self.marker_tag, [[time], [strain]])


class EnergyPlot:
    """Live energy-balance time-history plot widget."""

    def __init__(self) -> None:
        self.plot_tag = "energy_plot_widget"
        self.x_axis = "energy_plot_x"
        self.y_axis = "energy_plot_y"
        self.ke_series = "energy_ke_series"
        self.proj_ke_series = "energy_proj_ke_series"
        self.se_series = "energy_se_series"
        self.damp_series = "energy_damp_series"
        self.failure_series = "energy_failure_series"
        self.contact_series = "energy_contact_series"
        self.total_series = "energy_total_series"
        self.marker_tag = "energy_playback_cursor"

        # Buffer lists
        self.x_data: list[float] = []
        self.ke_data: list[float] = []
        self.proj_ke_data: list[float] = []
        self.se_data: list[float] = []
        self.damp_data: list[float] = []
        self.failure_data: list[float] = []
        self.contact_data: list[float] = []
        self.total_data: list[float] = []
        self._update_count = 0

    def build(self) -> None:
        """Construct the DearPyGui energy balance plot."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.plot(
            label="System Energy Telemetry History", tag=self.plot_tag, height=220, width=-1
        ):
            dpg.add_plot_legend()

            # X-Axis: Time in seconds
            dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)", tag=self.x_axis)

            # Y-Axis: Energy (Joules)
            with dpg.plot_axis(dpg.mvYAxis, label="Energy (J)", tag=self.y_axis):
                # 1. Kevlar Kinetic Energy (gold/orange)
                dpg.add_line_series(
                    self.x_data, self.ke_data, label="Kevlar KE", tag=self.ke_series
                )
                # 1.5 Projectile Kinetic Energy S6.5.5 (darker orange red)
                dpg.add_line_series(
                    self.x_data, self.proj_ke_data, label="Projectile KE", tag=self.proj_ke_series
                )
                # 2. Elastic Strain Energy (sky blue)
                dpg.add_line_series(
                    self.x_data, self.se_data, label="Strain Energy", tag=self.se_series
                )
                # 3. Damping Dissipated Energy (gray/purple)
                dpg.add_line_series(
                    self.x_data, self.damp_data, label="Damping Energy", tag=self.damp_series
                )
                # 3.5 Fiber breakage/rupture dissipated energy S7.14 (crimson red)
                dpg.add_line_series(
                    self.x_data, self.failure_data, label="Rupture Energy", tag=self.failure_series
                )
                # 4. Projectile + Inter-ply Contact Energy (pink)
                dpg.add_line_series(
                    self.x_data,
                    self.contact_data,
                    label="Contact Potential",
                    tag=self.contact_series,
                )
                # 5. Total Energy (neon green)
                dpg.add_line_series(
                    self.x_data, self.total_data, label="Total Energy", tag=self.total_series
                )
                # Playback vertical time cursor S6.5.4
                dpg.add_line_series(
                    [],
                    [],
                    label="Current Time",
                    tag=self.marker_tag,
                )

        # Apply custom themes to trace colors
        self._style_trace(self.ke_series, [218, 165, 32])  # Goldenrod
        self._style_trace(self.proj_ke_series, [255, 69, 0])  # Orange Red S6.5.5
        self._style_trace(self.se_series, [30, 144, 255])  # Dodger Blue
        self._style_trace(self.damp_series, [138, 43, 226])  # Blue Violet
        self._style_trace(self.failure_series, [220, 20, 60])  # Crimson Red S7.14
        self._style_trace(self.contact_series, [255, 105, 180])  # Hot Pink
        self._style_trace(self.total_series, [50, 205, 50])  # Lime Green

        # Playback cursor theme S6.5.4
        with dpg.theme() as cursor_theme, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(
                dpg.mvThemeCol_PlotLines, [255, 215, 0, 150]
            )  # Semi-transparent gold
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1.5)
        dpg.bind_item_theme(self.marker_tag, cursor_theme)

    def _style_trace(self, tag: str, rgb: list[int]) -> None:
        """Apply color styling and line weights to specific plot traces."""
        if dpg is None:  # pragma: no cover
            return
        with dpg.theme() as theme, dpg.theme_component(dpg.mvLineSeries):
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines, rgb)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2.0)
        dpg.bind_item_theme(tag, theme)

    def reset(self) -> None:
        """Clear all active data buffers."""
        self.x_data.clear()
        self.ke_data.clear()
        self.proj_ke_data.clear()
        self.se_data.clear()
        self.damp_data.clear()
        self.failure_data.clear()
        self.contact_data.clear()
        self.total_data.clear()
        self._update_count = 0

        if dpg is None:  # pragma: no cover
            return

        for tag in [
            self.ke_series,
            self.proj_ke_series,
            self.se_series,
            self.damp_series,
            self.failure_series,
            self.contact_series,
            self.total_series,
            self.marker_tag,
        ]:
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, [[], []])

    def update(self, time: float, energies: dict[str, float]) -> None:
        """Append a new telemetry data slice and refresh all energy lines.

        Parameters
        ----------
        time : float
            Current simulated elapsed time (seconds).
        energies : dict of str to float
            Dictionary with keys 'kinetic', 'strain', 'damped', 'failure_dissipated', 'contact', and 'total'.
        """
        self.x_data.append(time)
        self.ke_data.append(energies.get("kinetic", 0.0))
        self.proj_ke_data.append(energies.get("proj_ke", 0.0))
        self.se_data.append(energies.get("strain", 0.0))
        self.damp_data.append(energies.get("damped", 0.0))
        self.failure_data.append(energies.get("failure_dissipated", 0.0))
        self.contact_data.append(energies.get("contact", 0.0))
        self.total_data.append(energies.get("total", 0.0))

        if dpg is None:  # pragma: no cover
            return

        if dpg.does_item_exist(self.ke_series):
            self._update_count += 1
            if self._update_count == 1 or self._update_count % 4 == 0:
                x_plot = self.x_data
                ke_plot = self.ke_data
                proj_ke_plot = self.proj_ke_data
                se_plot = self.se_data
                damp_plot = self.damp_data
                failure_plot = self.failure_data
                contact_plot = self.contact_data
                total_plot = self.total_data

                n_points = len(x_plot)
                if n_points > 2000:
                    stride = n_points // 1000
                    x_plot = self.x_data[::stride]
                    ke_plot = self.ke_data[::stride]
                    proj_ke_plot = self.proj_ke_data[::stride]
                    se_plot = self.se_data[::stride]
                    damp_plot = self.damp_data[::stride]
                    failure_plot = self.failure_data[::stride]
                    contact_plot = self.contact_data[::stride]
                    total_plot = self.total_data[::stride]

                    if x_plot[-1] != self.x_data[-1]:
                        x_plot.append(self.x_data[-1])
                        ke_plot.append(self.ke_data[-1])
                        proj_ke_plot.append(self.proj_ke_data[-1])
                        se_plot.append(self.se_data[-1])
                        damp_plot.append(self.damp_data[-1])
                        failure_plot.append(self.failure_data[-1])
                        contact_plot.append(self.contact_data[-1])
                        total_plot.append(self.total_data[-1])

                dpg.set_value(self.ke_series, [x_plot, ke_plot])
                dpg.set_value(self.proj_ke_series, [x_plot, proj_ke_plot])
                dpg.set_value(self.se_series, [x_plot, se_plot])
                dpg.set_value(self.damp_series, [x_plot, damp_plot])
                dpg.set_value(self.failure_series, [x_plot, failure_plot])
                dpg.set_value(self.contact_series, [x_plot, contact_plot])
                dpg.set_value(self.total_series, [x_plot, total_plot])

                if self._update_count == 1 or self._update_count % 12 == 0:
                    dpg.fit_axis_data(self.x_axis)
                    dpg.fit_axis_data(self.y_axis)

    def set_playback_marker(self, time: float, max_energy: float) -> None:
        """Update the position of the vertical time cursor S6.5.4."""
        if dpg is None or not dpg.does_item_exist(self.marker_tag):
            return
        dpg.set_value(self.marker_tag, [[time, time], [0.0, max_energy]])
