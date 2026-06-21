from __future__ import annotations

import os

import dearpygui.dearpygui as dpg
import numpy as np
import pytest

from kevlargrid.gui.config_panel import ConfigPanel
from kevlargrid.gui.controls import SimulationControls
from kevlargrid.gui.dashboard import ResultsDashboard
from kevlargrid.gui.plots import EnergyPlot, StrainPlot
from kevlargrid.gui.viewport3d import Viewport3D
from kevlargrid.io.config import ValidationError, load_config, save_config, validate_config
from kevlargrid.solver.grid import generate_rectangular_grid

VALID_CONFIG = {
    "material": {
        "name": "Kevlar 29",
        "tensile_modulus_gpa": 71.0,
        "failure_strain": 0.036,
        "tensile_strength_gpa": 2.92,
        "fiber_density_gcc": 1.44,
        "areal_density_kgm2": 0.47,
        "shear_ratio": 0.0004,
        "crimp_factor": 0.10,
        "yarn_count": [17, 17],
    },
    "grid": {
        "nx": 10,
        "ny": 10,
        "dx": 0.01,
        "n_plies": 2,
        "t_ply": 0.002,
        "boundary_type": "fixed",
    },
    "projectile": {
        "mass": 0.05,
        "velocity": [0.0, 0.0, 400.0],
        "position": [0.0, 0.0, -0.005],
        "blade_width": 0.02,
        "edge_thickness": 0.005,
    },
    "simulation": {
        "duration": 0.001,
        "cfl_factor": 0.8,
        "damping_model": "rayleigh",
        "damping_coefficient": 0.05,
        "rayleigh_alpha": 0.0,
        "rayleigh_beta": 0.0001,
        "auto_cfl": True,
        "dt": 1.5e-7,
    },
}


class TestConfigRoundtrip:
    """GUI-driven configuration roundtrip and validation tests."""

    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that configuration can be saved and reloaded exactly.

        Saving a config dict to TOML and then reading it back should yield
        identical parameter structures and values.
        """
        temp_dir = tmp_path / "configs"
        temp_dir.mkdir()
        config_path = str(temp_dir / "test_config.toml")

        # 1. Save config
        save_config(VALID_CONFIG, config_path)
        assert os.path.exists(config_path)

        # 2. Load config
        loaded = load_config(config_path)

        # 3. Assert equality
        assert loaded == VALID_CONFIG

    def test_unit_parsing_and_conversion(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that string inputs with units are parsed and converted correctly."""
        temp_dir = tmp_path / "configs"
        temp_dir.mkdir()
        config_path = str(temp_dir / "unit_config.toml")

        unit_config = {
            "material": {
                "name": "Kevlar 29",
                "tensile_modulus_gpa": "71.0 GPa",
                "failure_strain": 0.036,
                "tensile_strength_gpa": "2.92 GPa",
                "fiber_density_gcc": "1.44 g/cc",
                "areal_density_kgm2": "0.47 kg/m^2",
                "shear_ratio": 0.0004,
                "crimp_factor": 0.10,
                "yarn_count": [17, 17],
            },
            "grid": {
                "nx": 10,
                "ny": 10,
                "dx": "10 mm",
                "n_plies": 2,
                "t_ply": "2.0 mm",
                "boundary_type": "fixed",
            },
            "projectile": {
                "mass": "0.05 kg",
                "velocity": ["0 m/s", "0 m/s", "400 m/s"],
                "position": ["0.0 mm", "0.0 mm", "-5.0 mm"],
                "blade_width": "20.0 mm",
                "edge_thickness": "5.0 mm",
            },
            "simulation": {
                "duration": "1.0 ms",
                "cfl_factor": 0.8,
                "damping_model": "rayleigh",
                "damping_coefficient": 0.05,
                "rayleigh_alpha": 0.0,
                "rayleigh_beta": "1.0 ns",
                "auto_cfl": True,
                "dt": "150.0 ns",
            },
        }

        save_config(unit_config, config_path)
        assert os.path.exists(config_path)

        # Loaded config must have all values normalized to numeric SI base floats
        loaded = load_config(config_path)

        assert loaded["material"]["tensile_modulus_gpa"] == 71.0
        assert loaded["material"]["tensile_strength_gpa"] == 2.92
        assert loaded["material"]["fiber_density_gcc"] == 1.44
        assert loaded["material"]["areal_density_kgm2"] == 0.47

        assert loaded["grid"]["dx"] == pytest.approx(0.01)
        assert loaded["grid"]["t_ply"] == pytest.approx(0.002)

        assert loaded["projectile"]["mass"] == 0.05
        assert loaded["projectile"]["velocity"] == [0.0, 0.0, 400.0]
        assert loaded["projectile"]["position"] == [0.0, 0.0, -0.005]
        assert loaded["projectile"]["blade_width"] == pytest.approx(0.02)
        assert loaded["projectile"]["edge_thickness"] == pytest.approx(0.005)

        assert loaded["simulation"]["duration"] == pytest.approx(0.001)
        assert loaded["simulation"]["rayleigh_beta"] == pytest.approx(1e-9)
        assert loaded["simulation"]["dt"] == pytest.approx(1.5e-7)

    def test_invalid_unit_rejected(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that incompatible units are rejected with ValidationError."""
        temp_dir = tmp_path / "configs"
        temp_dir.mkdir()
        config_path = str(temp_dir / "invalid_unit.toml")

        # tensile_modulus_gpa is specified in mm (incompatible!)
        invalid_unit_config = {
            **VALID_CONFIG,
            "material": {
                **VALID_CONFIG["material"],
                "tensile_modulus_gpa": "71.0 mm"
            }
        }

        save_config(invalid_unit_config, config_path)
        with pytest.raises(ValidationError, match="Unknown or incompatible unit"):
            load_config(config_path)

    def test_invalid_config_rejected(self) -> None:
        """Verify that invalid config formats or values are rejected.

        Validation rules should catch missing parameters or out-of-bounds metrics.
        """
        # Valid config passes
        assert validate_config(VALID_CONFIG) is True

        # Missing structural section
        invalid_structure = {k: v for k, v in VALID_CONFIG.items() if k != "material"}
        with pytest.raises(ValidationError, match="Missing or invalid section: 'material'"):
            validate_config(invalid_structure)

        # Out-of-bounds parameter: negative Modulus
        invalid_modulus = {
            **VALID_CONFIG,
            "material": {**VALID_CONFIG["material"], "tensile_modulus_gpa": -10.0},
        }
        with pytest.raises(
            ValidationError,
            match="Material property 'tensile_modulus_gpa' must be a positive number",
        ):
            validate_config(invalid_modulus)

        # Invalid CFL range
        invalid_cfl = {
            **VALID_CONFIG,
            "simulation": {**VALID_CONFIG["simulation"], "cfl_factor": 1.2},
        }
        with pytest.raises(
            ValidationError, match="Simulation parameter 'cfl_factor' must be in the range"
        ):
            validate_config(invalid_cfl)

        # Invalid position vector shape
        invalid_vector = {
            **VALID_CONFIG,
            "projectile": {**VALID_CONFIG["projectile"], "position": [0.0, 0.0]},
        }
        with pytest.raises(
            ValidationError, match="Projectile parameter 'position' must be a list of three numbers"
        ):
            validate_config(invalid_vector)


class TestGUIWidgets:
    """Headless unit tests verifying DearPyGui bindings and callbacks."""

    @pytest.fixture(autouse=True)
    def setup_dpg(self) -> None:
        """Initialize DPG context for headless tests."""
        dpg.create_context()
        dpg.add_window(tag="test_parent_window")
        dpg.push_container_stack("test_parent_window")
        yield
        dpg.pop_container_stack()
        dpg.destroy_context()

    def test_material_presets_sync(self) -> None:
        """Verify changing preset disables entries and syncs material values."""
        panel = ConfigPanel()
        panel.build()

        # Check default: Kevlar 29
        assert dpg.get_value(panel.mat_combo) == "Kevlar 29"
        assert dpg.get_value(panel.mat_modulus) == pytest.approx(71.0)
        assert not dpg.get_item_configuration(panel.mat_modulus)["enabled"]

        # Select Kevlar 49
        dpg.set_value(panel.mat_combo, "Kevlar 49")
        panel._on_material_change(None, "Kevlar 49")
        assert dpg.get_value(panel.mat_modulus) == pytest.approx(112.4)
        assert not dpg.get_item_configuration(panel.mat_modulus)["enabled"]

        # Select Custom -> must enable inputs
        dpg.set_value(panel.mat_combo, "Custom")
        panel._on_material_change(None, "Custom")
        assert dpg.get_item_configuration(panel.mat_modulus)["enabled"]

    def test_mode_a_b_toggles(self) -> None:
        """Verify Mode A/B toggles hide/show ply spacing group."""
        panel = ConfigPanel()
        panel.build()

        # Mode A (default): t_ply group is hidden
        assert not dpg.get_item_configuration(panel.grid_t_ply_group)["show"]

        # Switch to Mode B (Checkout Stacking)
        dpg.set_value(panel.grid_mode, "Mode B (Checkout Stacking)")
        panel._on_mode_change(None, "Mode B (Checkout Stacking)")
        assert dpg.get_item_configuration(panel.grid_t_ply_group)["show"]

    def test_damping_model_visibility_and_defaults(self) -> None:
        """Verify damping widgets visibility toggles correctly and default values are stable."""
        panel = ConfigPanel()
        panel.build()

        # Check default: Rayleigh Damping
        assert dpg.get_value(panel.sim_damping_model) == "Rayleigh Damping"
        assert dpg.get_value(panel.sim_damping_coeff) == pytest.approx(0.05)
        assert dpg.get_value(panel.sim_rayleigh_alpha) == pytest.approx(0.0)
        assert dpg.get_value(panel.sim_rayleigh_beta) == pytest.approx(1e-9)

        # Check initial visibility (Rayleigh shown, Viscous hidden)
        assert not dpg.get_item_configuration(panel.row_damping_coeff)["show"]
        assert dpg.get_item_configuration(panel.row_rayleigh_alpha)["show"]
        assert dpg.get_item_configuration(panel.row_rayleigh_beta)["show"]

        # Switch to Viscous Damping
        dpg.set_value(panel.sim_damping_model, "Viscous Damping")
        panel._on_damping_change(None, "Viscous Damping")
        assert dpg.get_item_configuration(panel.row_damping_coeff)["show"]
        assert not dpg.get_item_configuration(panel.row_rayleigh_alpha)["show"]
        assert not dpg.get_item_configuration(panel.row_rayleigh_beta)["show"]

        # Switch back to Rayleigh Damping
        dpg.set_value(panel.sim_damping_model, "Rayleigh Damping")
        panel._on_damping_change(None, "Rayleigh Damping")
        assert not dpg.get_item_configuration(panel.row_damping_coeff)["show"]
        assert dpg.get_item_configuration(panel.row_rayleigh_alpha)["show"]
        assert dpg.get_item_configuration(panel.row_rayleigh_beta)["show"]

    def test_boundary_rmin_autocalc(self) -> None:
        """Verify Infinite boundary computes R_min dynamically."""
        panel = ConfigPanel()
        panel.build()

        # Fixed Clamped (default)
        assert not dpg.get_item_configuration(panel.grid_rmin_group)["show"]

        # Switch to Infinite Grid
        dpg.set_value(panel.grid_boundary, "Infinite Grid (Auto)")
        panel._on_boundary_change(None, "Infinite Grid (Auto)")
        assert dpg.get_item_configuration(panel.grid_rmin_group)["show"]
        r_min = dpg.get_value(panel.grid_rmin_display)
        # Should be calculated and non-zero
        assert r_min > 0.0

    def test_projectile_live_ke(self) -> None:
        """Verify projectile KE is calculated dynamically."""
        panel = ConfigPanel()
        panel.build()

        # Initial values: mass=0.05, Vz=400 -> KE = 0.5 * 0.05 * 160000 = 4000 J
        assert dpg.get_value(panel.proj_ke_display) == pytest.approx(4000.0)

        # Change mass to 0.1
        dpg.set_value(panel.proj_mass, 0.1)
        panel._on_projectile_change(None, None)
        assert dpg.get_value(panel.proj_ke_display) == pytest.approx(8000.0)

    def test_config_panel_get_set_sync(self) -> None:
        """Verify that get_config and set_config round-trip successfully in GUI."""
        panel = ConfigPanel()
        panel.build()

        # Apply config dict
        panel.set_config(VALID_CONFIG)

        # Retrieve config dict and verify round-trip parity
        retrieved = panel.get_config()

        # Assert sections are identical (DearPyGui floats are single-precision)
        assert retrieved["material"]["name"] == VALID_CONFIG["material"]["name"]
        assert retrieved["material"]["tensile_modulus_gpa"] == pytest.approx(
            VALID_CONFIG["material"]["tensile_modulus_gpa"]
        )
        assert retrieved["grid"]["nx"] == VALID_CONFIG["grid"]["nx"]
        assert retrieved["grid"]["n_plies"] == VALID_CONFIG["grid"]["n_plies"]
        assert retrieved["grid"]["t_ply"] == pytest.approx(VALID_CONFIG["grid"]["t_ply"])
        assert retrieved["projectile"]["mass"] == pytest.approx(VALID_CONFIG["projectile"]["mass"])
        assert retrieved["projectile"]["velocity"][2] == pytest.approx(
            VALID_CONFIG["projectile"]["velocity"][2]
        )
        assert retrieved["simulation"]["damping_model"] == VALID_CONFIG["simulation"]["damping_model"]
        assert retrieved["simulation"]["damping_coefficient"] == pytest.approx(
            VALID_CONFIG["simulation"]["damping_coefficient"]
        )
        assert retrieved["simulation"]["rayleigh_alpha"] == pytest.approx(
            VALID_CONFIG["simulation"]["rayleigh_alpha"]
        )
        assert retrieved["simulation"]["rayleigh_beta"] == pytest.approx(
            VALID_CONFIG["simulation"]["rayleigh_beta"]
        )
        assert retrieved["simulation"]["snapshot_interval"] == 100

        # Verify setting custom snapshot_interval
        cfg_custom = VALID_CONFIG.copy()
        cfg_custom["simulation"] = {
            **VALID_CONFIG["simulation"],
            "snapshot_interval": 250,
        }
        panel.set_config(cfg_custom)
        retrieved_custom = panel.get_config()
        assert retrieved_custom["simulation"]["snapshot_interval"] == 250

    def test_file_size_estimate(self) -> None:
        """Verify dynamic HDF5 file size calculation and snapshot interval widgets."""
        panel = ConfigPanel()
        panel.build()

        # Check default snapshot interval widget exists
        assert dpg.get_value(panel.sim_snapshot_interval) == 100

        # Change grid size and verify estimate changes
        dpg.set_value(panel.grid_nx, 25)
        dpg.set_value(panel.grid_ny, 25)
        panel._update_file_size_estimate()

        size_str_25 = dpg.get_value(panel.sim_file_size_display)
        assert "KB" in size_str_25 or "MB" in size_str_25 or "Bytes" in size_str_25

        # Decrease snapshot interval (saving more steps) -> size should increase
        dpg.set_value(panel.sim_snapshot_interval, 10)
        panel._update_file_size_estimate()
        size_str_10 = dpg.get_value(panel.sim_file_size_display)

        # Helper to parse size
        def parse_kb(s: str) -> float:
            val = float(s.split()[0])
            if "MB" in s:
                return val * 1024
            elif "Bytes" in s:
                return val / 1024
            return val

        assert parse_kb(size_str_10) > parse_kb(size_str_25)

    def test_viewport_mouse_down_button_extraction(self, monkeypatch) -> None:
        """Verify that _on_mouse_down correctly extracts button ID from list/tuple payloads."""
        view = Viewport3D()
        view.build()

        monkeypatch.setattr(dpg, "is_item_hovered", lambda tag: True)
        monkeypatch.setattr(dpg, "get_mouse_pos", lambda local: [100.0, 100.0])

        # 1. Test single integer app_data
        view._on_mouse_down("test_sender", 0)
        assert view.drag_button == 0
        assert view.is_dragging is True

        # Reset dragging state to allow subsequent calls to process S7.10
        view.is_dragging = False

        # 2. Test list/tuple app_data [button_id, hold_time]
        view._on_mouse_down("test_sender", [1, 0.5])
        assert view.drag_button == 1
        assert view.is_dragging is True


class TestControlsWidget:
    """Unit tests for simulation controls and telemetry updates."""

    @pytest.fixture(autouse=True)
    def setup_dpg(self) -> None:
        """Initialize DPG context for controls test."""
        dpg.create_context()
        dpg.add_window(tag="test_parent_window")
        dpg.push_container_stack("test_parent_window")
        yield
        dpg.pop_container_stack()
        dpg.destroy_context()

    def test_button_states_toggling(self) -> None:
        """Verify button enabled/disabled states based on simulation runner phase."""
        ctrls = SimulationControls()
        ctrls.build()

        # 1. Idle state
        ctrls.set_button_states("idle")
        assert dpg.get_item_configuration(ctrls.start_btn)["enabled"]
        assert not dpg.get_item_configuration(ctrls.pause_btn)["enabled"]
        assert not dpg.get_item_configuration(ctrls.stop_btn)["enabled"]
        assert dpg.get_item_configuration(ctrls.reset_btn)["enabled"]

        # 2. Running state
        ctrls.set_button_states("running")
        assert not dpg.get_item_configuration(ctrls.start_btn)["enabled"]
        assert dpg.get_item_configuration(ctrls.pause_btn)["enabled"]
        assert dpg.get_item_configuration(ctrls.stop_btn)["enabled"]
        assert not dpg.get_item_configuration(ctrls.reset_btn)["enabled"]

        # 3. Paused state
        ctrls.set_button_states("paused")
        assert dpg.get_item_configuration(ctrls.start_btn)["enabled"]
        assert dpg.get_item_configuration(ctrls.start_btn)["label"] == "Resume"

    def test_progress_and_telemetry_rendering(self) -> None:
        """Verify dynamic text rendering and progress updates."""
        ctrls = SimulationControls()
        ctrls.build()

        # Update telemetry values
        ctrls.update_telemetry(
            elapsed_time=0.0005,
            duration=0.001,
            step=1200,
            speed=50000.0,
            eta=0.015,
            failed_count=4,
            ke=120.0,
            se=80.0,
        )

        # Assert values
        assert dpg.get_value(ctrls.progress_bar) == pytest.approx(0.5)
        assert dpg.get_item_configuration(ctrls.progress_bar)["overlay"] == "50.0%"
        assert "Step: 1200" in dpg.get_value(ctrls.text_steps)
        assert "Ruptured Springs: 4" in dpg.get_value(ctrls.text_failed)
        assert "KE=120.00" in dpg.get_value(ctrls.text_energy)


class TestVisualizationWidgets:
    """Unit tests for plots, 3D viewport, results dashboard, and playback timeline."""

    @pytest.fixture(autouse=True)
    def setup_dpg(self) -> None:
        """Initialize DPG context for visualization widgets."""
        dpg.create_context()
        dpg.add_window(tag="test_parent_window")
        dpg.push_container_stack("test_parent_window")
        yield
        dpg.pop_container_stack()
        dpg.destroy_context()

    def test_strain_and_energy_plots(self) -> None:
        """Verify real-time plots update dynamic buffers correctly."""
        p_strain = StrainPlot()
        p_strain.build()
        p_strain.reset(threshold=0.04)

        # Confirm reset state
        assert p_strain.threshold_val == 0.04
        assert len(p_strain.x_data) == 0

        # Append data slices
        p_strain.update(0.0001, 0.015)
        p_strain.update(0.0002, 0.025)
        assert len(p_strain.x_data) == 2
        assert p_strain.y_data[1] == pytest.approx(0.025)

        p_energy = EnergyPlot()
        p_energy.build()
        p_energy.reset()
        assert len(p_energy.x_data) == 0

        # Append energy values
        p_energy.update(
            0.0001,
            {"kinetic": 100.0, "strain": 50.0, "damped": 5.0, "contact": 1.0, "total": 156.0},
        )
        assert len(p_energy.x_data) == 1
        assert p_energy.ke_data[0] == pytest.approx(100.0)
        assert p_energy.total_data[0] == pytest.approx(156.0)

    def test_results_dashboard(self) -> None:
        """Verify color-coded PASS/FAIL outcome indicators and Deceleration populate."""
        dash = ResultsDashboard()
        dash.build()

        # Outcome 1: Arrested (PASS)
        arrest_report = {
            "arrested": True,
            "peak_deceleration_g": 320.0,
            "yarn_rupture_percentage": 14.5,
            "residual_velocity_ms": 0.0,
            "energy_dissipation_efficiency": 1.0,
            "max_layer_perforated": -1,
        }
        dash.populate(arrest_report)
        assert "ARRESTED - PASS" in dpg.get_item_configuration(dash.status_badge)["label"]
        assert "320.0 G" in dpg.get_value(dash.val_max_decel)
        assert "14.50%" in dpg.get_value(dash.val_rupture_pct)

        # Outcome 2: Perforated (FAIL)
        perf_report = {
            "arrested": False,
            "peak_deceleration_g": 120.0,
            "yarn_rupture_percentage": 95.0,
            "residual_velocity_ms": 150.0,
            "energy_dissipation_efficiency": 0.85,
            "max_layer_perforated": 1,
        }
        dash.populate(perf_report)
        assert "PERFORATED - FAIL" in dpg.get_item_configuration(dash.status_badge)["label"]
        assert "120.0 G" in dpg.get_value(dash.val_max_decel)
        assert "95.00%" in dpg.get_value(dash.val_rupture_pct)
        assert "Layer 1" in dpg.get_value(dash.val_penetration_ply)

    def test_viewport_perspective_projection(self) -> None:
        """Verify viewport initializes camera values and updates layer states."""
        view = Viewport3D()
        view.build()

        # Defaults
        assert view.yaw == pytest.approx(0.785)
        assert view.pitch == pytest.approx(0.523)
        assert view.distance == pytest.approx(0.3)

        # Load grid coordinates
        grid = generate_rectangular_grid(
            5, 5, 0.01, VALID_CONFIG["material"], n_plies=2, t_ply=0.002
        )
        view.reset(grid, n_plies=2, n_nodes_per_layer=25)

        assert view.n_plies == 2
        assert len(view.layer_visibility) == 2
        assert view.layer_visibility[0] is True

        # Test slide updates
        dpg.set_value(view.slider_yaw, 90.0)
        view._on_slider_change(view.slider_yaw, 90.0)
        assert view.yaw == pytest.approx(np.radians(90.0))

        dpg.set_value(view.slider_zoom, 0.5)
        view._on_slider_change(view.slider_zoom, 0.5)
        assert view.distance == pytest.approx(0.5)
