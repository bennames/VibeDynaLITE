"""Configuration input panel widget.

Provides parameters input for material, grid, projectile, and explicit
simulation settings with dynamic widgets visibility and live calculations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

try:
    import dearpygui.dearpygui as dpg
except ImportError:  # pragma: no cover
    dpg = None  # type: ignore[assignment]

from kevlargrid.materials.library import MATERIALS, get_material
from kevlargrid.solver import backend
from kevlargrid.solver.boundary import compute_min_radius
from kevlargrid.utils import get_logger

logger = get_logger("gui.config_panel")


class ConfigPanel:
    """DearPyGui panel for simulation configuration input."""

    def __init__(self) -> None:
        # Widget tags
        self.group_tag = "config_panel_group"

        # Material section tags
        self.mat_combo = "mat_combo"
        self.mat_modulus = "mat_modulus"
        self.mat_strain = "mat_strain"
        self.mat_strength = "mat_strength"
        self.mat_fiber_density = "mat_fiber_density"
        self.mat_areal_density = "mat_areal_density"
        self.mat_shear_ratio = "mat_shear_ratio"
        self.mat_crimp = "mat_crimp"
        self.mat_fracture_multiplier = "mat_fracture_multiplier"

        # Configuration file callbacks S6.5.9
        self.load_callback: Callable[[], None] | None = None
        self.save_callback: Callable[[], None] | None = None
        self.mat_yarn_count_x = "mat_yarn_count_x"
        self.mat_yarn_count_y = "mat_yarn_count_y"

        # Grid section tags
        self.grid_nx = "grid_nx"
        self.grid_ny = "grid_ny"
        self.grid_dx = "grid_dx"
        self.grid_mode = "grid_mode"
        self.grid_n_plies = "grid_n_plies"
        self.grid_t_ply = "grid_t_ply"
        self.grid_t_ply_group = "grid_t_ply_group"
        self.grid_boundary = "grid_boundary"
        self.grid_rmin_display = "grid_rmin_display"
        self.grid_rmin_group = "grid_rmin_group"
        self.grid_ideal_length_display = "grid_ideal_length_display"
        self.grid_actual_length_display = "grid_actual_length_display"
        self.grid_boundary_status = "grid_boundary_status"

        # Projectile section tags
        self.proj_mass = "proj_mass"
        self.proj_vx = "proj_vx"
        self.proj_vy = "proj_vy"
        self.proj_vz = "proj_vz"
        self.proj_px = "proj_px"
        self.proj_py = "proj_py"
        self.proj_pz = "proj_pz"
        self.proj_width = "proj_width"
        self.proj_thickness = "proj_thickness"
        self.proj_shape = "proj_shape"
        self.proj_radius = "proj_radius"
        self.proj_radius_cyl = "proj_radius_cyl"
        self.proj_length_cyl = "proj_length_cyl"
        self.proj_edge_radius = "proj_edge_radius"
        self.proj_radius_bullet = "proj_radius_bullet"
        self.proj_length_bullet = "proj_length_bullet"
        self.proj_ogive_multiplier = "proj_ogive_multiplier"
        self.proj_span = "proj_span"
        self.proj_root_chord = "proj_root_chord"
        self.proj_tip_chord = "proj_tip_chord"
        self.proj_twist = "proj_twist"
        self.proj_thickness_ratio = "proj_thickness_ratio"
        self.proj_tip_radius = "proj_tip_radius"
        self.proj_ke_display = "proj_ke_display"

        # Simulation section tags
        self.sim_duration = "sim_duration"
        self.sim_cfl = "sim_cfl"
        self.sim_damping_model = "sim_damping_model"
        self.sim_damping_coeff = "sim_damping_coeff"
        self.sim_rayleigh_alpha = "sim_rayleigh_alpha"
        self.sim_rayleigh_beta = "sim_rayleigh_beta"
        self.sim_snapshot_interval = "sim_snapshot_interval"
        self.sim_file_size_display = "sim_file_size_display"
        self.sim_auto_cfl = "sim_auto_cfl"
        self.sim_dt = "sim_dt"
        self.compute_backend = "compute_backend"
        self.hardware_device = "hardware_device"

        self.row_damping_coeff = "row_damping_coeff"
        self.row_rayleigh_alpha = "row_rayleigh_alpha"
        self.row_rayleigh_beta = "row_rayleigh_beta"

    def build(self) -> None:
        """Construct the panel's DearPyGui widgets."""
        if dpg is None:  # pragma: no cover
            return

        with dpg.child_window(tag=self.group_tag, width=400, border=True):
            dpg.add_text("Simulation Configuration", color=[0, 191, 255])
            dpg.add_separator()

            # --- MATERIAL LIBRARY SECTION ---
            with dpg.collapsing_header(label="Material Properties", default_open=True):
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Material Preset")
                        dpg.add_combo(
                            items=[*MATERIALS.keys(), "Custom"],
                            default_value="Kevlar 29",
                            tag=self.mat_combo,
                            callback=self._on_material_change,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Modulus (GPa)")
                        dpg.add_input_float(
                            tag=self.mat_modulus, default_value=71.0, enabled=False, width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Failure Strain")
                        dpg.add_input_float(
                            tag=self.mat_strain, default_value=0.036, enabled=False, width=-1
                        )
                    with dpg.table_row(show=False):
                        dpg.add_text("Strength (GPa)")
                        dpg.add_input_float(
                            tag=self.mat_strength, default_value=2.92, enabled=False, width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Fiber Density (g/cc)")
                        dpg.add_input_float(
                            tag=self.mat_fiber_density, default_value=1.44, enabled=False, width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Areal Density (kg/m2)")
                        dpg.add_input_float(
                            tag=self.mat_areal_density, default_value=0.47, enabled=False, width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Shear Stiffness Ratio")
                        dpg.add_input_float(
                            tag=self.mat_shear_ratio, default_value=0.0004, enabled=False, width=-1
                        )
                    with dpg.table_row(show=False):
                        dpg.add_text("Crimp Factor")
                        dpg.add_input_float(
                            tag=self.mat_crimp, default_value=0.10, enabled=False, width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Fracture Multiplier")
                        dpg.add_input_float(
                            tag=self.mat_fracture_multiplier,
                            default_value=1.5,
                            enabled=False,
                            format="%.2f",
                            width=-1,
                        )
                    with dpg.table_row(show=False):
                        dpg.add_text("Yarn X (Warp Count/in)")
                        dpg.add_input_int(
                            tag=self.mat_yarn_count_x, default_value=17, enabled=False, width=-1
                        )
                    with dpg.table_row(show=False):
                        dpg.add_text("Yarn Y (Weft Count/in)")
                        dpg.add_input_int(
                            tag=self.mat_yarn_count_y, default_value=17, enabled=False, width=-1
                        )


            # --- GRID GEOMETRY SECTION ---
            with dpg.collapsing_header(label="Grid Geometry & Boundaries", default_open=True):
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()

                    with dpg.table_row():
                        dpg.add_text("Nodes Nx")
                        dpg.add_input_int(
                            tag=self.grid_nx,
                            default_value=11,
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Nodes Ny")
                        dpg.add_input_int(
                            tag=self.grid_ny,
                            default_value=11,
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Spacing dx (m)")
                        dpg.add_input_float(
                            tag=self.grid_dx,
                            default_value=0.01,
                            format="%.4f",
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Analysis Mode")
                        dpg.add_combo(
                            items=["Mode A (Sizing Multiplier)", "Mode B (Checkout Stacking)"],
                            default_value="Mode A (Sizing Multiplier)",
                            tag=self.grid_mode,
                            callback=self._on_mode_change,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Number of Plies")
                        dpg.add_input_int(
                            tag=self.grid_n_plies,
                            default_value=1,
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )

                with dpg.group(tag=self.grid_t_ply_group, show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Spacing t_ply (m)")
                            dpg.add_input_float(
                                tag=self.grid_t_ply,
                                default_value=0.001,
                                format="%.4f",
                                callback=self._update_file_size_estimate_cb,
                                width=-1,
                            )

                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("Boundary Condition")
                        dpg.add_combo(
                            items=[
                                "Fixed Clamped Edges",
                                "Infinite Grid (Auto)",
                                "Non-Reflecting (Impedance Matched)",
                            ],
                            default_value="Fixed Clamped Edges",
                            tag=self.grid_boundary,
                            callback=self._on_boundary_change,
                            width=-1,
                        )

                with dpg.group(tag=self.grid_rmin_group, show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Auto Radius R_min (m)")
                            dpg.add_input_float(
                                tag=self.grid_rmin_display,
                                default_value=0.0,
                                enabled=False,
                                width=-1,
                            )
                        with dpg.table_row():
                            dpg.add_text("Idealized Length (m)")
                            dpg.add_input_float(
                                tag=self.grid_ideal_length_display,
                                default_value=0.0,
                                enabled=False,
                                width=-1,
                            )
                        with dpg.table_row():
                            dpg.add_text("Actual Length (m)")
                            dpg.add_input_float(
                                tag=self.grid_actual_length_display,
                                default_value=0.0,
                                enabled=False,
                                width=-1,
                            )

                    with dpg.group(horizontal=True):
                        dpg.add_text("Status: ")
                        dpg.add_text(
                            "✗ Insufficient — reflections likely",
                            tag=self.grid_boundary_status,
                            color=[255, 80, 80],
                        )
                    dpg.add_button(
                        label="Apply Infinite Boundary Dimensions",
                        callback=self._apply_infinite_boundary_dims,
                        tag="btn_apply_infinite_dims",
                        width=250,
                    )

            # --- PROJECTILE GEOMETRY SECTION ---
            with dpg.collapsing_header(label="Projectile Parameters", default_open=True):
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("Mass (kg)")
                        dpg.add_input_float(
                            tag=self.proj_mass,
                            default_value=0.05,
                            callback=self._on_projectile_change,
                            width=-1,
                        )

                dpg.add_text("Initial Velocity (m/s):")
                with dpg.group(horizontal=True):
                    dpg.add_input_float(
                        label="Vx",
                        width=70,
                        tag=self.proj_vx,
                        default_value=0.0,
                        callback=self._on_projectile_change,
                    )
                    dpg.add_input_float(
                        label="Vy",
                        width=70,
                        tag=self.proj_vy,
                        default_value=0.0,
                        callback=self._on_projectile_change,
                    )
                    dpg.add_input_float(
                        label="Vz",
                        width=70,
                        tag=self.proj_vz,
                        default_value=400.0,
                        callback=self._on_projectile_change,
                    )

                dpg.add_text("Initial Position (m):")
                with dpg.group(horizontal=True):
                    dpg.add_input_float(label="X0", width=70, tag=self.proj_px, default_value=0.0)
                    dpg.add_input_float(label="Y0", width=70, tag=self.proj_py, default_value=0.0)
                    dpg.add_input_float(
                        label="Z0", width=70, tag=self.proj_pz, default_value=-0.005
                    )

                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("Shape Type")
                        dpg.add_combo(
                            items=["Box", "Sphere", "Cylinder", "Bullet", "Propeller"],
                            tag=self.proj_shape,
                            default_value="Box",
                            callback=self._on_projectile_shape_change,
                            width=-1,
                        )

                # Groups for shape-specific parameters
                with dpg.group(tag="proj_box_group", show=True):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Blade Width (m)")
                            dpg.add_input_float(tag=self.proj_width, default_value=0.02, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Edge Thickness (m)")
                            dpg.add_input_float(tag=self.proj_thickness, default_value=0.005, width=-1)

                with dpg.group(tag="proj_sphere_group", show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Radius (m)")
                            dpg.add_input_float(tag=self.proj_radius, default_value=0.005, width=-1)

                with dpg.group(tag="proj_cylinder_group", show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Radius (m)")
                            dpg.add_input_float(tag=self.proj_radius_cyl, default_value=0.005, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Length (m)")
                            dpg.add_input_float(tag=self.proj_length_cyl, default_value=0.01, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Edge Radius (m)")
                            dpg.add_input_float(tag=self.proj_edge_radius, default_value=0.0, width=-1)

                with dpg.group(tag="proj_bullet_group", show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Radius (m)")
                            dpg.add_input_float(tag=self.proj_radius_bullet, default_value=0.005, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Length (m)")
                            dpg.add_input_float(tag=self.proj_length_bullet, default_value=0.01, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Ogive Multiplier")
                            dpg.add_input_float(tag=self.proj_ogive_multiplier, default_value=2.0, width=-1)

                with dpg.group(tag="proj_propeller_group", show=False):
                    with dpg.table(header_row=False):
                        dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                        dpg.add_table_column()
                        with dpg.table_row():
                            dpg.add_text("Span (m)")
                            dpg.add_input_float(tag=self.proj_span, default_value=0.05, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Root Chord (m)")
                            dpg.add_input_float(tag=self.proj_root_chord, default_value=0.01, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Tip Chord (m)")
                            dpg.add_input_float(tag=self.proj_tip_chord, default_value=0.005, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Twist (deg)")
                            dpg.add_input_float(tag=self.proj_twist, default_value=15.0, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Thickness Ratio (%)")
                            dpg.add_input_float(tag=self.proj_thickness_ratio, default_value=12.0, width=-1)
                        with dpg.table_row():
                            dpg.add_text("Tip Radius (m)")
                            dpg.add_input_float(tag=self.proj_tip_radius, default_value=0.002, width=-1)

                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("KE (Joules)")
                        dpg.add_input_float(
                            tag=self.proj_ke_display, default_value=4000.0, enabled=False, width=-1
                        )

            # --- SIMULATION & SOLVER CONTROLS SECTION ---
            with dpg.collapsing_header(label="Solver Settings", default_open=True):
                with dpg.table(header_row=False):
                    dpg.add_table_column(width_fixed=True, init_width_or_weight=180)
                    dpg.add_table_column()
                    with dpg.table_row():
                        dpg.add_text("Sim Duration (s)")
                        dpg.add_input_float(
                            tag=self.sim_duration,
                            default_value=0.001,
                            format="%.5f",
                            callback=self._on_boundary_change,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("CFL safety factor")
                        dpg.add_input_float(
                            tag=self.sim_cfl,
                            default_value=0.8,
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Dynamic Auto-CFL")
                        dpg.add_checkbox(
                            tag=self.sim_auto_cfl,
                            default_value=True,
                            callback=self._on_auto_cfl_change,
                        )
                    with dpg.table_row():
                        dpg.add_text("Static Timestep (s)")
                        dpg.add_input_float(
                            tag=self.sim_dt,
                            default_value=1.5e-7,
                            format="%.9f",
                            enabled=False,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Damping Model")
                        dpg.add_combo(
                            items=["Rayleigh Damping", "Viscous Damping"],
                            default_value="Rayleigh Damping",
                            tag=self.sim_damping_model,
                            callback=self._on_damping_change,
                            width=-1,
                        )

                    with dpg.table_row(tag=self.row_damping_coeff):
                        dpg.add_text("Viscous Damping Coeff")
                        dpg.add_input_float(
                            tag=self.sim_damping_coeff, default_value=0.05, format="%.4f", width=-1
                        )
                    with dpg.table_row(tag=self.row_rayleigh_alpha):
                        dpg.add_text("Rayleigh α (mass-prop)")
                        dpg.add_input_float(
                            tag=self.sim_rayleigh_alpha, default_value=0.0, format="%.4f", width=-1
                        )
                    with dpg.table_row(tag=self.row_rayleigh_beta):
                        dpg.add_text("Rayleigh β (stiff-prop)")
                        dpg.add_input_float(
                            tag=self.sim_rayleigh_beta, default_value=1e-9, format="%.9f", width=-1
                        )
                    with dpg.table_row():
                        dpg.add_text("Snapshot Interval")
                        dpg.add_input_int(
                            tag=self.sim_snapshot_interval,
                            default_value=100,
                            min_value=1,
                            max_value=1000,
                            callback=self._update_file_size_estimate_cb,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Est HDF5 File Size")
                        dpg.add_input_text(
                            tag=self.sim_file_size_display,
                            default_value="0.0 KB",
                            enabled=False,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Compute Backend")
                        dpg.add_combo(
                            items=["Taichi", "Numba"],
                            default_value=backend.get_backend_name().capitalize(),
                            tag=self.compute_backend,
                            callback=self._on_backend_change,
                            width=-1,
                        )
                    with dpg.table_row():
                        dpg.add_text("Active Hardware")
                        dpg.add_input_text(
                            tag=self.hardware_device,
                            default_value=backend.get_active_device(),
                            enabled=False,
                            width=-1,
                        )

            # --- CONFIGURATION PROFILES SECTION S6.5.9 ---
            with (
                dpg.collapsing_header(label="Configuration Profiles", default_open=True),
                dpg.group(horizontal=True),
            ):
                dpg.add_button(
                    label="Load Config",
                    callback=self._menu_load_config_cb,
                    width=110,
                )
                dpg.add_button(
                    label="Save Config",
                    callback=self._menu_save_config_cb,
                    width=110,
                )

            # --- Add Tooltips ---
            self._add_tooltips()

            # Initialize dynamic values
            self._on_material_change(None, dpg.get_value(self.mat_combo))
            self._on_projectile_change(None, None)
            self._on_damping_change(None, dpg.get_value(self.sim_damping_model))

    def _on_material_change(self, sender: str | None, app_data: str) -> None:
        """Triggered when selected material preset is changed."""
        if dpg is None:  # pragma: no cover
            return

        is_custom = app_data == "Custom"
        # Toggle fields enabled status
        dpg.configure_item(self.mat_modulus, enabled=is_custom)
        dpg.configure_item(self.mat_strain, enabled=is_custom)
        dpg.configure_item(self.mat_strength, enabled=is_custom)
        dpg.configure_item(self.mat_fiber_density, enabled=is_custom)
        dpg.configure_item(self.mat_areal_density, enabled=is_custom)
        dpg.configure_item(self.mat_shear_ratio, enabled=is_custom)
        dpg.configure_item(self.mat_crimp, enabled=is_custom)
        dpg.configure_item(self.mat_fracture_multiplier, enabled=is_custom)
        dpg.configure_item(self.mat_yarn_count_x, enabled=is_custom)
        dpg.configure_item(self.mat_yarn_count_y, enabled=is_custom)

        if not is_custom:
            mat = get_material(app_data)
            dpg.set_value(self.mat_modulus, mat.get("tensile_modulus_gpa", 71.0))
            dpg.set_value(self.mat_strain, mat.get("failure_strain", 0.036))
            dpg.set_value(self.mat_strength, mat.get("tensile_strength_gpa", 2.92))
            dpg.set_value(self.mat_fiber_density, mat.get("fiber_density_gcc", 1.44))
            dpg.set_value(self.mat_areal_density, mat.get("areal_density_kgm2", 0.47))
            dpg.set_value(self.mat_shear_ratio, mat.get("shear_ratio", 0.0004))
            dpg.set_value(self.mat_crimp, mat.get("crimp_factor", 0.10))
            dpg.set_value(self.mat_fracture_multiplier, mat.get("fracture_energy_multiplier", 1.5))
            yc = mat.get("yarn_count", (17, 17))
            if isinstance(yc, (list, tuple)):
                dpg.set_value(self.mat_yarn_count_x, yc[0])
                dpg.set_value(self.mat_yarn_count_y, yc[1])

        self._on_boundary_change(None, None)

    def _on_mode_change(self, sender: str | None, app_data: str) -> None:
        """Triggered when Mode selection (A vs B) is toggled."""
        if dpg is None:  # pragma: no cover
            return

        is_mode_b = "Mode B" in app_data
        dpg.configure_item(self.grid_t_ply_group, show=is_mode_b)
        self._update_file_size_estimate()

    def _on_boundary_change(self, sender: str | None, app_data: str | None) -> None:
        """Triggered when boundary selection is toggled or sim duration changes."""
        if dpg is None:  # pragma: no cover
            return

        b_type = dpg.get_value(self.grid_boundary)
        is_infinite = "Infinite" in b_type
        dpg.configure_item(self.grid_rmin_group, show=is_infinite)

        if is_infinite:
            # Auto size: wave speed c = sqrt(E / rho)
            modulus_gpa = dpg.get_value(self.mat_modulus)
            areal_density = dpg.get_value(self.mat_areal_density)
            fiber_density_gcc = dpg.get_value(self.mat_fiber_density)

            # Thickness t = areal_density / density
            thickness = areal_density / (fiber_density_gcc * 1000.0)
            e_mod = modulus_gpa * 1e9
            k_ortho = e_mod * thickness

            # Cell mass
            dx = dpg.get_value(self.grid_dx)
            m_cell = areal_density * dx * dx
            if m_cell > 0.0:
                c_transverse = dx * (k_ortho / m_cell) ** 0.5
            else:
                c_transverse = 0.0

            sim_duration = dpg.get_value(self.sim_duration)
            r_min = compute_min_radius(c_transverse, sim_duration, 1.5)
            dpg.set_value(self.grid_rmin_display, r_min)

            ideal_length = 2.0 * r_min
            nx = dpg.get_value(self.grid_nx)
            actual_length = (nx - 1) * dx
            dpg.set_value(self.grid_ideal_length_display, ideal_length)
            dpg.set_value(self.grid_actual_length_display, actual_length)
            is_sufficient = actual_length >= ideal_length
            dpg.configure_item(
                self.grid_boundary_status,
                default_value="✓ Sufficient"
                if is_sufficient
                else "✗ Insufficient — reflections likely",
                color=[0, 200, 0] if is_sufficient else [255, 80, 80],
            )

        self._update_file_size_estimate()

    def _on_projectile_change(self, sender: str | None, app_data: str | None) -> None:
        """Triggered when projectile mass or velocity inputs are modified."""
        if dpg is None:  # pragma: no cover
            return

        mass = dpg.get_value(self.proj_mass)
        vx = dpg.get_value(self.proj_vx)
        vy = dpg.get_value(self.proj_vy)
        vz = dpg.get_value(self.proj_vz)

        v_sq = vx**2 + vy**2 + vz**2
        ke = 0.5 * mass * v_sq
        dpg.set_value(self.proj_ke_display, ke)

    def _on_projectile_shape_change(self, sender: str | None, app_data: str | None) -> None:
        """Triggered when the projectile shape selection combo is changed."""
        if dpg is None:
            return
        shape = str(app_data).lower()
        dpg.configure_item("proj_box_group", show=shape == "box")
        dpg.configure_item("proj_sphere_group", show=shape == "sphere")
        dpg.configure_item("proj_cylinder_group", show=shape == "cylinder")
        dpg.configure_item("proj_bullet_group", show=shape == "bullet")
        dpg.configure_item("proj_propeller_group", show=shape == "propeller")

    def _add_tooltips(self) -> None:
        """Add documentation tooltips to configuration panel inputs."""
        if dpg is None:  # pragma: no cover
            return

        # Material Section
        with dpg.tooltip(self.mat_combo):
            dpg.add_text("Select a predefined Kevlar style or define your own 'Custom' preset")
        with dpg.tooltip(self.mat_modulus):
            dpg.add_text("Young's modulus of fiber material (GPa)")
        with dpg.tooltip(self.mat_strain):
            dpg.add_text("Axial strain value at which spring rupture irreversibly occurs")
        with dpg.tooltip(self.mat_fracture_multiplier):
            dpg.add_text(
                "Fracture energy multiplier factor (dimensionless, typical Kevlar default = 1.5)"
            )
        with dpg.tooltip(self.mat_yarn_count_x):
            dpg.add_text(
                "Warp Yarn Count: Number of longitudinal yarns per inch of Kevlar fabric construction."
            )
        with dpg.tooltip(self.mat_yarn_count_y):
            dpg.add_text(
                "Weft Yarn Count: Number of transverse yarns per inch of Kevlar fabric construction."
            )

        # Grid Section
        with dpg.tooltip(self.grid_nx):
            dpg.add_text("Number of discrete nodes along X-axis")
        with dpg.tooltip(self.grid_mode):
            dpg.add_text(
                "Mode A: single scaled layer for fast sizing calculations.\nMode B: stacked plies with contact."
            )
        with dpg.tooltip(self.grid_n_plies):
            dpg.add_text("Discrete counts of stacked ply layers")
        with dpg.tooltip(self.grid_boundary):
            dpg.add_text(
                "Fixed Clamped: Edges have zero velocity.\nInfinite Grid: Auto-computes minimum sizing to prevent stress wave reflection."
            )

        # Projectile Section
        with dpg.tooltip(self.proj_mass):
            dpg.add_text("Total mass of striking dynamic projectile (kg)")
        with dpg.tooltip(self.proj_ke_display):
            dpg.add_text("Live Kinetic Energy in Joules (0.5 * m * v^2)")

        # Solver Section
        with dpg.tooltip(self.sim_duration):
            dpg.add_text("Total physical simulation window duration (seconds)")
        with dpg.tooltip(self.sim_cfl):
            dpg.add_text("CFL stability safety factor multiplier (must be <= 1.0)")
        with dpg.tooltip(self.sim_rayleigh_alpha):
            dpg.add_text("Rayleigh mass-proportional damping coefficient α (1/s)")
        with dpg.tooltip(self.sim_rayleigh_beta):
            dpg.add_text("Rayleigh stiffness-proportional damping coefficient β (seconds)")
        with dpg.tooltip(self.grid_dx):
            dpg.add_text(
                "Spacing dx: Grid physical nodal mesh resolution.\nWarning: For stable contact mechanics, dx should be <= projectile edge thickness."
            )
        with dpg.tooltip(self.sim_snapshot_interval):
            dpg.add_text("How often solver data frames are logged (e.g. log every 100 timesteps)")

    def _update_file_size_estimate_cb(self, sender: str | None, app_data: Any) -> None:
        """DearPyGui callback proxy to trigger file size recalculation."""
        self._on_boundary_change(None, None)

    def _update_file_size_estimate(self) -> None:
        """Calculate and display the estimated HDF5 file size in real-time."""
        if dpg is None or not dpg.does_item_exist(self.sim_file_size_display):
            return

        try:
            nx = int(dpg.get_value(self.grid_nx))
            ny = int(dpg.get_value(self.grid_ny))
            dx = float(dpg.get_value(self.grid_dx))
            n_plies = int(dpg.get_value(self.grid_n_plies))

            # Determine Mode B vs Mode A
            grid_mode = dpg.get_value(self.grid_mode)
            is_mode_b = "Mode B" in grid_mode

            n_nodes = nx * ny
            if is_mode_b:
                n_nodes_tot = n_nodes * n_plies
            else:
                n_nodes_tot = n_nodes

            # Springs per layer
            ortho = nx * (ny - 1) + ny * (nx - 1)
            diag = 2 * (nx - 1) * (ny - 1)
            springs_per_ply = ortho + diag
            if is_mode_b:
                n_springs_tot = springs_per_ply * n_plies
            else:
                n_springs_tot = springs_per_ply

            # Wave velocity to get dt
            modulus_gpa = dpg.get_value(self.mat_modulus)
            areal_density = dpg.get_value(self.mat_areal_density)
            fiber_density_gcc = dpg.get_value(self.mat_fiber_density)

            thickness = areal_density / (fiber_density_gcc * 1000.0)
            e_mod = modulus_gpa * 1e9
            k_ortho = e_mod * thickness

            is_mode_b_flag = is_mode_b and n_plies > 1
            m_scale = 1.0 if is_mode_b_flag else float(n_plies)
            m_cell = m_scale * areal_density * dx * dx

            # critical timestep S6.5.7
            # Match actual solver which uses m_min = 0.25 * m_cell (corner node)
            # and k_max = max(k_ortho, k_penalty) where k_penalty = 10 * k_ortho
            m_min = 0.25 * m_cell if m_cell > 0.0 else 0.0
            k_penalty = 10.0 * k_ortho
            k_max = max(k_ortho, k_penalty)
            if m_min > 0.0 and k_max > 0.0:
                dt_crit = (m_min / k_max) ** 0.5
            else:
                dt_crit = 1e-6

            cfl = dpg.get_value(self.sim_cfl)
            dt = cfl * dt_crit
            if dt <= 0.0:
                dt = 1e-6

            duration = dpg.get_value(self.sim_duration)
            total_steps = int(duration / dt)

            snapshot_interval = int(dpg.get_value(self.sim_snapshot_interval))
            if snapshot_interval < 1:
                snapshot_interval = 1

            saved_steps = max(1, total_steps // snapshot_interval)

            # Estimate HDF5 size in bytes
            pos_size = saved_steps * n_nodes_tot * 3 * 4
            spring_size = saved_steps * n_springs_tot * 1
            energy_size = saved_steps * 5 * 4
            proj_size = saved_steps * 3 * 4
            time_size = saved_steps * 4
            metadata_size = 2000

            tot_bytes = pos_size + spring_size + energy_size + proj_size + time_size + metadata_size

            if tot_bytes < 1024:
                size_str = f"{tot_bytes} Bytes"
            elif tot_bytes < 1024 * 1024:
                size_str = f"{tot_bytes / 1024:.2f} KB"
            else:
                size_str = f"{tot_bytes / (1024 * 1024):.2f} MB"

            dpg.set_value(self.sim_file_size_display, size_str)
        except Exception:
            dpg.set_value(self.sim_file_size_display, "Error")

    def get_config(self) -> dict:
        """Read the current widget values and return a structured config dict."""
        if dpg is None:  # pragma: no cover
            return {}

        is_mode_b = "Mode B" in dpg.get_value(self.grid_mode)
        boundary_val = dpg.get_value(self.grid_boundary)
        if "Infinite" in boundary_val:
            b_type = "infinite"
        elif "Non-Reflecting" in boundary_val:
            b_type = "non-reflecting"
        else:
            b_type = "fixed"

        return {
            "material": {
                "name": dpg.get_value(self.mat_combo),
                "tensile_modulus_gpa": dpg.get_value(self.mat_modulus),
                "failure_strain": dpg.get_value(self.mat_strain),
                "tensile_strength_gpa": dpg.get_value(self.mat_strength),
                "fiber_density_gcc": dpg.get_value(self.mat_fiber_density),
                "areal_density_kgm2": dpg.get_value(self.mat_areal_density),
                "shear_ratio": dpg.get_value(self.mat_shear_ratio),
                "crimp_factor": dpg.get_value(self.mat_crimp),
                "fracture_energy_multiplier": dpg.get_value(self.mat_fracture_multiplier),
                "yarn_count": [
                    dpg.get_value(self.mat_yarn_count_x),
                    dpg.get_value(self.mat_yarn_count_y),
                ],
            },
            "grid": {
                "nx": dpg.get_value(self.grid_nx),
                "ny": dpg.get_value(self.grid_ny),
                "dx": dpg.get_value(self.grid_dx),
                "n_plies": dpg.get_value(self.grid_n_plies),
                "t_ply": dpg.get_value(self.grid_t_ply) if is_mode_b else None,
                "boundary_type": b_type,
            },
            "projectile": {
                "mass": dpg.get_value(self.proj_mass),
                "velocity": [
                    dpg.get_value(self.proj_vx),
                    dpg.get_value(self.proj_vy),
                    dpg.get_value(self.proj_vz),
                ],
                "position": [
                    dpg.get_value(self.proj_px),
                    dpg.get_value(self.proj_py),
                    dpg.get_value(self.proj_pz),
                ],
                "shape": dpg.get_value(self.proj_shape).lower(),
                "shape_type": dpg.get_value(self.proj_shape).lower(),
                "blade_width": dpg.get_value(self.proj_width),
                "edge_thickness": dpg.get_value(self.proj_thickness),
                "radius": dpg.get_value(self.proj_radius_cyl) if dpg.get_value(self.proj_shape).lower() == "cylinder" else (dpg.get_value(self.proj_radius_bullet) if dpg.get_value(self.proj_shape).lower() == "bullet" else dpg.get_value(self.proj_radius)),
                "length": dpg.get_value(self.proj_length_cyl) if dpg.get_value(self.proj_shape).lower() == "cylinder" else (dpg.get_value(self.proj_length_bullet) if dpg.get_value(self.proj_shape).lower() == "bullet" else 0.01),
                "edge_radius": dpg.get_value(self.proj_edge_radius),
                "ogive_multiplier": dpg.get_value(self.proj_ogive_multiplier),
                "span": dpg.get_value(self.proj_span),
                "root_chord": dpg.get_value(self.proj_root_chord),
                "tip_chord": dpg.get_value(self.proj_tip_chord),
                "twist": dpg.get_value(self.proj_twist),
                "thickness_ratio": dpg.get_value(self.proj_thickness_ratio),
                "tip_radius": dpg.get_value(self.proj_tip_radius),
            },
            "simulation": {
                "duration": dpg.get_value(self.sim_duration),
                "cfl_factor": dpg.get_value(self.sim_cfl),
                "damping_model": "viscous"
                if dpg.get_value(self.sim_damping_model) == "Viscous Damping"
                else "rayleigh",
                "damping_coefficient": dpg.get_value(self.sim_damping_coeff),
                "rayleigh_alpha": dpg.get_value(self.sim_rayleigh_alpha),
                "rayleigh_beta": dpg.get_value(self.sim_rayleigh_beta),
                "snapshot_interval": dpg.get_value(self.sim_snapshot_interval)
                if dpg.does_item_exist(self.sim_snapshot_interval)
                else 100,
                "auto_cfl": dpg.get_value(self.sim_auto_cfl),
                "dt": dpg.get_value(self.sim_dt),
                "backend": dpg.get_value(self.compute_backend).lower(),
            },
        }

    def set_config(self, config: dict) -> None:
        """Update all GUI input widgets based on config values.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        """
        if dpg is None:  # pragma: no cover
            return

        # 1. Update material preset
        mat = config["material"]
        preset_name = mat.get("name", "Custom")
        dpg.set_value(self.mat_combo, preset_name)

        # Force material callback to toggle edit states
        self._on_material_change(None, preset_name)

        # Explicitly set loaded values (essential if "Custom" was loaded)
        dpg.set_value(self.mat_modulus, mat.get("tensile_modulus_gpa", 71.0))
        dpg.set_value(self.mat_strain, mat.get("failure_strain", 0.036))
        dpg.set_value(self.mat_strength, mat.get("tensile_strength_gpa", 2.92))
        dpg.set_value(self.mat_fiber_density, mat.get("fiber_density_gcc", 1.44))
        dpg.set_value(self.mat_areal_density, mat.get("areal_density_kgm2", 0.47))
        dpg.set_value(self.mat_shear_ratio, mat.get("shear_ratio", 0.0004))
        dpg.set_value(self.mat_crimp, mat.get("crimp_factor", 0.10))
        dpg.set_value(self.mat_fracture_multiplier, mat.get("fracture_energy_multiplier", 1.5))
        yc = mat.get("yarn_count", [17, 17])
        dpg.set_value(self.mat_yarn_count_x, yc[0])
        dpg.set_value(self.mat_yarn_count_y, yc[1])

        # 2. Update Grid settings
        grid = config["grid"]
        dpg.set_value(self.grid_nx, grid.get("nx", 11))
        dpg.set_value(self.grid_ny, grid.get("ny", 11))
        dpg.set_value(self.grid_dx, grid.get("dx", 0.01))
        dpg.set_value(self.grid_n_plies, grid.get("n_plies", 1))

        t_ply = grid.get("t_ply")
        is_mode_b = t_ply is not None
        mode_val = "Mode B (Checkout Stacking)" if is_mode_b else "Mode A (Sizing Multiplier)"
        dpg.set_value(self.grid_mode, mode_val)
        self._on_mode_change(None, mode_val)
        if is_mode_b:
            dpg.set_value(self.grid_t_ply, t_ply)

        b_type = grid.get("boundary_type", "fixed")
        if b_type == "infinite":
            boundary_val = "Infinite Grid (Auto)"
        elif b_type == "non-reflecting":
            boundary_val = "Non-Reflecting (Impedance Matched)"
        else:
            boundary_val = "Fixed Clamped Edges"
        dpg.set_value(self.grid_boundary, boundary_val)
        self._on_boundary_change(None, boundary_val)

        # 3. Update Projectile settings
        proj = config["projectile"]
        dpg.set_value(self.proj_mass, proj.get("mass", 0.05))
        vel = proj.get("velocity", [0.0, 0.0, 400.0])
        dpg.set_value(self.proj_vx, vel[0])
        dpg.set_value(self.proj_vy, vel[1])
        dpg.set_value(self.proj_vz, vel[2])

        pos = proj.get("position", [0.0, 0.0, -0.005])
        dpg.set_value(self.proj_px, pos[0])
        dpg.set_value(self.proj_py, pos[1])
        dpg.set_value(self.proj_pz, pos[2])

        shape = proj.get("shape", proj.get("shape_type", "box")).lower()
        shape_cased = "Box"
        if shape == "sphere":
            shape_cased = "Sphere"
        elif shape == "cylinder":
            shape_cased = "Cylinder"
        elif shape == "bullet":
            shape_cased = "Bullet"
        elif shape == "propeller":
            shape_cased = "Propeller"
            
        dpg.set_value(self.proj_shape, shape_cased)
        self._on_projectile_shape_change(None, shape_cased)

        dpg.set_value(self.proj_width, proj.get("blade_width", 0.02))
        dpg.set_value(self.proj_thickness, proj.get("edge_thickness", 0.005))
        
        radius = proj.get("radius", 0.005)
        length = proj.get("length", 0.01)
        dpg.set_value(self.proj_radius, radius)
        dpg.set_value(self.proj_radius_cyl, radius)
        dpg.set_value(self.proj_length_cyl, length)
        dpg.set_value(self.proj_edge_radius, proj.get("edge_radius", 0.0))
        dpg.set_value(self.proj_radius_bullet, radius)
        dpg.set_value(self.proj_length_bullet, length)
        dpg.set_value(self.proj_ogive_multiplier, proj.get("ogive_multiplier", 2.0))
        dpg.set_value(self.proj_span, proj.get("span", 0.05))
        dpg.set_value(self.proj_root_chord, proj.get("root_chord", 0.01))
        dpg.set_value(self.proj_tip_chord, proj.get("tip_chord", 0.005))
        dpg.set_value(self.proj_twist, proj.get("twist", 15.0))
        dpg.set_value(self.proj_thickness_ratio, proj.get("thickness_ratio", 12.0))
        dpg.set_value(self.proj_tip_radius, proj.get("tip_radius", 0.002))
        
        self._on_projectile_change(None, None)

        # 4. Update Simulation settings
        sim = config["simulation"]
        dpg.set_value(self.sim_duration, sim.get("duration", 0.001))
        dpg.set_value(self.sim_cfl, sim.get("cfl_factor", 0.8))
        auto_cfl_val = sim.get("auto_cfl", True)
        dpg.set_value(self.sim_auto_cfl, auto_cfl_val)
        dpg.configure_item(self.sim_dt, enabled=not auto_cfl_val)
        dpg.set_value(self.sim_dt, sim.get("dt", 1.5e-7))
        # Determine damping model and set widgets
        model = sim.get("damping_model")
        if model is None:
            if "rayleigh_beta" in sim and sim["rayleigh_beta"] > 0.0:
                model = "rayleigh"
            elif "damping_coefficient" in sim:
                model = "viscous"
            else:
                model = "rayleigh"

        dpg.set_value(
            self.sim_damping_model, "Viscous Damping" if model == "viscous" else "Rayleigh Damping"
        )
        self._on_damping_change(
            None, "Viscous Damping" if model == "viscous" else "Rayleigh Damping"
        )

        dpg.set_value(self.sim_damping_coeff, sim.get("damping_coefficient", 0.05))
        dpg.set_value(self.sim_rayleigh_alpha, sim.get("rayleigh_alpha", 0.0))
        dpg.set_value(self.sim_rayleigh_beta, sim.get("rayleigh_beta", 1e-9))
        if "snapshot_interval" in sim and dpg.does_item_exist(self.sim_snapshot_interval):
            dpg.set_value(self.sim_snapshot_interval, sim["snapshot_interval"])

        # Update backend if present in the configuration TOML S6.5.9
        loaded_backend = sim.get("backend", "taichi").capitalize()
        if dpg.does_item_exist(self.compute_backend):
            dpg.set_value(self.compute_backend, loaded_backend)
            self._on_backend_change(None, loaded_backend)

        # Re-sync boundary sizing calculation & file size estimate
        self._on_boundary_change(None, None)
        self._update_file_size_estimate()

    def _apply_infinite_boundary_dims(self, sender: str, app_data: Any) -> None:
        """Calculate and set the exact Nx and Ny required to satisfy R_min S6.5.8."""
        if dpg is None:
            return
        dx = float(dpg.get_value(self.grid_dx))
        r_min = float(dpg.get_value(self.grid_rmin_display))
        if dx > 0.0 and r_min > 0.0:
            nx = math.ceil(2 * r_min / dx + 1)
            if nx % 2 == 0:
                nx += 1
            # Clamp to a safe max to avoid system freezing before sprint 7 parallelization
            nx = max(11, min(501, nx))
            if nx > 201:
                logger.warning(
                    f"Large grid size selected ({nx}x{nx}). This may cause slow simulation runs."
                )
            dpg.set_value(self.grid_nx, nx)
            dpg.set_value(self.grid_ny, nx)
            self._on_boundary_change(None, None)

    def _menu_load_config_cb(self, sender: str, app_data: Any) -> None:
        """Forward sidebar Load trigger S6.5.9."""
        if hasattr(self, "load_callback") and self.load_callback is not None:
            self.load_callback()

    def _menu_save_config_cb(self, sender: str, app_data: Any) -> None:
        """Forward sidebar Save trigger S6.5.9."""
        if hasattr(self, "save_callback") and self.save_callback is not None:
            self.save_callback()

    def _on_backend_change(self, sender: Any, app_data: Any) -> None:
        """Callback triggered when the compute backend combo selection changes."""
        backend_name = str(app_data).lower()
        import os

        os.environ["KEVLARGRID_BACKEND"] = backend_name
        backend.BACKEND = backend_name
        if dpg.does_item_exist(self.hardware_device):
            dpg.set_value(self.hardware_device, backend.get_active_device())

    def _on_damping_change(self, sender: Any, app_data: str) -> None:
        """Toggle visibility of damping parameter inputs based on selected model."""
        if dpg is None:
            return
        is_viscous = app_data == "Viscous Damping"
        dpg.configure_item(self.row_damping_coeff, show=is_viscous)
        dpg.configure_item(self.row_rayleigh_alpha, show=not is_viscous)
        dpg.configure_item(self.row_rayleigh_beta, show=not is_viscous)

    def _on_auto_cfl_change(self, sender: Any, app_data: Any) -> None:
        """Toggle enabled state of static timestep input based on Auto-CFL checkbox."""
        if dpg is None:
            return
        auto_cfl = dpg.get_value(self.sim_auto_cfl)
        dpg.configure_item(self.sim_dt, enabled=not auto_cfl)
