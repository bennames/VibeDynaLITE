import numpy as np

from kevlargrid.io.config import normalize_config_units, validate_config
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.projectile import Projectile


def test_grid_corrugation_and_elements():
    """Verify corrugations and elements connectivity are generated correctly."""
    material = {
        "name": "steel",
        "tensile_modulus_gpa": 200.0,
        "failure_strain": 0.20,
        "tensile_strength_gpa": 0.45,
        "fiber_density_gcc": 7.8,
        "areal_density_kgm2": 7.8,  # 1mm thick steel sheet
        "shear_ratio": 0.38,
    }

    # Flat grid
    grid_flat = generate_rectangular_grid(nx=5, ny=5, dx=0.1, material=material)
    assert len(grid_flat.elements) == 16
    assert np.all(grid_flat.nodes[:, 2] == 0.0)

    # Corrugated grid
    grid_corr = generate_rectangular_grid(
        nx=5,
        ny=5,
        dx=0.1,
        material=material,
        corrugation_amplitude=0.02,
        corrugation_period=0.2,
        corrugation_axis="x",
    )
    # Check that Z positions are perturbed
    assert not np.all(grid_corr.nodes[:, 2] == 0.0)
    assert np.max(np.abs(grid_corr.nodes[:, 2])) <= 0.02


def test_config_validation_metallic_sheet():
    """Verify validation rules for metallic_sheet and j2_plasticity."""
    cfg = {
        "material": {
            "name": "Steel",
            "tensile_modulus_gpa": 200.0,
            "failure_strain": 0.20,
            "tensile_strength_gpa": 0.45,
            "fiber_density_gcc": 7.8,
            "areal_density_kgm2": 7.8,
            "shear_ratio": 0.38,
            "material_model": "j2_plasticity",
            "yield_strength_gpa": 0.25,
            "hardening_modulus_gpa": 1.0,
            "ultimate_strain": 0.15,
        },
        "grid": {
            "nx": 10,
            "ny": 10,
            "dx": "10 mm",
            "n_plies": 1,
            "boundary_type": "fixed",
            "corrugation_amplitude": "5 mm",
            "corrugation_period": "50 mm",
        },
        "projectile": {
            "shape": "box",
            "mass": "0.1 kg",
            "velocity": [0.0, 0.0, -10.0],
            "position": [0.0, 0.0, 0.05],
            "blade_width": "20 mm",
            "edge_thickness": "2 mm",
        },
        "simulation": {
            "structure_type": "metallic_sheet",
            "duration": 1e-4,
            "dt": 1e-7,
            "auto_cfl": False,
            "cfl_factor": 0.5,
            "damping_model": "rayleigh",
            "damping_coefficient": 0.0,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 1e-9,
        },
    }

    # Normalize units
    normalize_config_units(cfg)
    assert cfg["grid"]["dx"] == 0.01
    assert cfg["grid"]["corrugation_amplitude"] == 0.005

    # Validate format
    assert validate_config(cfg) is True
    assert cfg["material"]["poisson_ratio"] == 0.3
    assert cfg["grid"]["corrugation_axis"] == "x"


def test_metallic_sheet_simulation():
    """Verify that simulation in metallic_sheet mode runs successfully."""
    material = {
        "name": "Steel",
        "tensile_modulus_gpa": 200.0,
        "failure_strain": 0.20,
        "tensile_strength_gpa": 0.45,
        "fiber_density_gcc": 7.8,
        "areal_density_kgm2": 7.8,
        "shear_ratio": 0.38,
        "material_model": "j2_plasticity",
        "yield_strength_gpa": 0.25,
        "hardening_modulus_gpa": 1.0,
        "ultimate_strain": 0.15,
        "poisson_ratio": 0.3,
    }

    grid = generate_rectangular_grid(
        nx=6,
        ny=6,
        dx=0.01,
        material=material,
        corrugation_amplitude=0.0,
        corrugation_period=0.04,
        corrugation_axis="x",
    )

    proj = Projectile(
        mass=0.1,
        velocity=[0.0, 0.0, -100.0],
        position=[0.0, 0.0, 0.0005],  # Start overlapping so contact is immediate
        shape_type="box",
        blade_width=0.02,
        edge_thickness=0.002,
    )

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    # boundary clamp edges
    for i in range(6):
        for j in range(6):
            if i == 0 or i == 5 or j == 0 or j == 5:
                boundary_mask[i * 6 + j] = 1

    nodal_external_forces = np.zeros_like(positions)

    thickness = 7.8 / (7.8 * 1000.0)  # 0.001

    # Run the loop directly
    res = fused_leapfrog_loop(
        positions=positions,
        velocities=velocities,
        grid_springs=grid.springs,
        grid_stiffnesses=grid.stiffnesses,
        grid_rest_lengths=grid.rest_lengths,
        grid_failed=grid.failed,
        grid_masses=grid.masses,
        grid_tension_only=grid.tension_only,
        boundary_mask=boundary_mask,
        nodal_external_forces=nodal_external_forces,
        proj_position=proj.position,
        proj_velocity=proj.velocity,
        proj_mass=proj.mass,
        proj_blade_width=proj.blade_width,
        proj_edge_thickness=proj.edge_thickness,
        n_plies=1,
        n_nodes_per_layer=36,
        t_ply=0.002,
        dx=0.01,
        k_penalty=10000.0,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-9,
        failure_strain=0.20,
        damage_onset_strain=0.15,
        fracture_energy_multiplier=1.0,
        dt=1e-7,
        n_steps=100,
        save_interval=5,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=-1.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
        use_viscous=False,
        cfl_factor=0.5,
        proj_quat=proj.quat,
        proj_omega=proj.omega,
        proj_shape_type="box",
        contact_energy_init=0.0,
        mu_s=0.1,
        friction_dissipated_init=0.0,
        structure_type="metallic_sheet",
        material_model="j2_plasticity",
        yield_strength_gpa=0.25,
        hardening_modulus_gpa=1.0,
        ultimate_strain=0.15,
        poisson_ratio=0.3,
        elements=grid.elements,
        youngs_modulus_gpa=200.0,
        thickness=thickness,
    )

    assert res is not None
    # Check that positions have changed due to impact
    new_positions = res[0]
    assert not np.allclose(new_positions, grid.nodes)

    # Check that projectile velocity has changed
    new_proj_vel = res[4]
    assert new_proj_vel[2] != -100.0
