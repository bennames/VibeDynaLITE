from __future__ import annotations

import os

import dearpygui.dearpygui as dpg
import pytest

from kevlargrid.gui.config_panel import ConfigPanel
from kevlargrid.gui.controls import SimulationControls
from kevlargrid.io.config import ValidationError, load_config, save_config, validate_config

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
        "damping_coefficient": 1.5,
    },
}


class TestConfigRoundtrip:
    """GUI-driven configuration roundtrip and validation tests."""

    def test_save_load_roundtrip(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify that configuration can be saved and reloaded exactly.

        Saving a config dict to JSON and then reading it back should yield
        identical parameter structures and values.
        """
        temp_dir = tmp_path / "configs"
        temp_dir.mkdir()
        config_path = str(temp_dir / "test_config.json")

        # 1. Save config
        save_config(VALID_CONFIG, config_path)
        assert os.path.exists(config_path)

        # 2. Load config
        loaded = load_config(config_path)

        # 3. Assert equality
        assert loaded == VALID_CONFIG

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
        assert retrieved["simulation"]["damping_coefficient"] == pytest.approx(
            VALID_CONFIG["simulation"]["damping_coefficient"]
        )


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
