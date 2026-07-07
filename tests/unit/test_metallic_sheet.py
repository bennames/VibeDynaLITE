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
        density_kgm3=7800.0,
    )

    assert res is not None
    # Check that positions have changed due to impact
    new_positions = res[0]
    assert not np.allclose(new_positions, grid.nodes)

    # Check that projectile velocity has changed
    new_proj_vel = res[4]
    assert new_proj_vel[2] != -100.0


def test_metallic_sheet_post_processing_and_orientation_fixes():
    """Verify that worker post-processing mapping and JIT quaternion integration work correctly in metallic_sheet mode."""
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
        nx=5,
        ny=5,
        dx=0.01,
        material=material,
    )

    # 1. Verify manual mapping of element failure to spring failure
    n_nodes = len(grid.nodes)
    node_elements = [[] for _ in range(n_nodes)]
    for e_idx, elem in enumerate(grid.elements):
        for node in elem:
            node_elements[node].append(e_idx)
    spring_elements = []
    for n0, n1 in grid.springs:
        shared = list(set(node_elements[n0]).intersection(node_elements[n1]))
        spring_elements.append(shared)

    # If all elements fail, spring fails
    failed_elements = np.ones(len(grid.elements), dtype=bool)
    failed_springs = np.zeros(len(grid.springs), dtype=bool)
    for s_idx, el_indices in enumerate(spring_elements):
        if len(el_indices) > 0:
            failed_springs[s_idx] = True
            for e_idx in el_indices:
                if not failed_elements[e_idx]:
                    failed_springs[s_idx] = False
                    break
    assert np.all(failed_springs)

    # 2. Verify quaternion integration in Numba shell solver
    proj = Projectile(
        mass=0.1,
        velocity=[0.0, 0.0, -10.0],
        position=[0.0, 0.0, 0.05],
        shape_type="propeller",
        blade_width=0.02,
        edge_thickness=0.002,
        span=0.05,
        omega=[0.0, 1000.0, 0.0],
        quat=[0.7071, 0.0, 0.7071, 0.0],
    )

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    nodal_external_forces = np.zeros_like(positions)
    thickness = 0.001
    hist_proj_quat = np.zeros((10, 4), dtype=np.float64)

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
        n_nodes_per_layer=25,
        t_ply=0.002,
        dx=0.01,
        k_penalty=10000.0,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-9,
        failure_strain=0.20,
        damage_onset_strain=0.15,
        fracture_energy_multiplier=1.0,
        dt=1e-7,
        n_steps=50,
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
        proj_shape_type="propeller",
        proj_span=0.05,
        hist_proj_quat=hist_proj_quat,
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
        density_kgm3=7800.0,
    )

    assert res is not None
    assert not np.allclose(hist_proj_quat[0], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(hist_proj_quat[0], [0.7071, 0.0, 0.7071, 0.0], atol=1e-3)
    assert not np.allclose(hist_proj_quat[-1], [0.7071, 0.0, 0.7071, 0.0], atol=1e-3)


def test_metallic_sheet_stabilization():
    """Verify that the explicit shell solver is numerically stable and damped."""
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
        nx=5,
        ny=5,
        dx=0.01,
        material=material,
        corrugation_amplitude=0.0,
    )

    proj = Projectile(
        mass=0.1,
        velocity=[0.0, 0.0, -10.0],
        position=[0.0, 0.0, 0.001],
        shape_type="box",
        blade_width=0.02,
        edge_thickness=0.002,
    )

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    # fixed boundary edges
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                boundary_mask[i * 5 + j] = 1

    nodal_external_forces = np.zeros_like(positions)
    thickness = 0.001

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
        n_nodes_per_layer=25,
        t_ply=0.002,
        dx=0.01,
        k_penalty=100000.0,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-5,  # active damping
        failure_strain=0.20,
        damage_onset_strain=0.15,
        fracture_energy_multiplier=1.0,
        dt=1e-7,
        n_steps=50,
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
        density_kgm3=7800.0,
    )

    assert res is not None
    final_pos, final_vel, final_failed, final_proj_pos, final_proj_vel, damp_diss, *rest = res

    # 1. System must be stable (node velocities remain reasonable, no explosion to 1e6 J)
    # Projectile initial kinetic energy is 0.5 * 0.1 * 10^2 = 5.0 Joules.
    # Grid kinetic energy should remain bounded and not diverge.
    grid_ke = 0.5 * np.sum(grid.masses[:, None] * final_vel**2)
    assert grid_ke < 5.0

    # 2. Damping energy should be correctly accumulated and positive due to active motion
    assert damp_diss > 0.0
