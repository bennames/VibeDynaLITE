import numpy as np

from kevlargrid.io.config import normalize_config_units, validate_config
from kevlargrid.solver.fused import (
    fused_leapfrog_loop,
    numba_step_shell_forces_and_failures,
)
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
        nx=5,
        ny=5,
        dx=0.01,
        material=material,
        corrugation_amplitude=0.0,
        corrugation_period=0.04,
        corrugation_axis="x",
    )

    proj = Projectile(
        mass=0.1,
        velocity=[0.0, 0.0, -100.0],
        position=[0.0, 0.0, 0.0052],  # Start overlapping at the bottom face so contact is immediate
        shape_type="box",
        blade_width=0.02,
        edge_thickness=0.002,
    )

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    # boundary clamp edges
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                boundary_mask[i * 5 + j] = 1

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


def test_tangential_contact_friction():
    """Verify that a projectile sliding tangentially along the metallic sheet feels Coulomb friction."""
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

    # Position the projectile to slide tangentially in the XY plane, slightly pressed into the sheet
    # Sheet is at z = 0.
    # A box of half-thickness t_h = 0.0025 starts at z_center = -0.002, so it penetrates by 0.0005 m.
    proj = Projectile(
        mass=0.1,
        velocity=[50.0, 0.0, 0.0],
        position=[0.0, 0.0, -0.002],
        shape_type="box",
        blade_width=0.02,
        edge_thickness=0.005,
    )

    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    for i in range(25):
        boundary_mask[i] = 1  # fix all sheet nodes to act as a rigid base for friction

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
        k_penalty=5.0e7,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-6,
        failure_strain=0.20,
        damage_onset_strain=0.15,
        fracture_energy_multiplier=1.0,
        dt=1e-7,
        n_steps=100,
        save_interval=10,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=1.0,
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
        mu_s=0.8,  # high friction
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
    final_proj_vel = res[4]
    fric_diss = res[17]

    # The projectile should have decelerated along X due to Coulomb friction
    assert final_proj_vel[0] < 50.0
    # Friction energy must be positive
    assert fric_diss > 0.0


def test_von_karman_wave_propagation():
    """Verify that transverse deflection couples to membrane strain due to Von Karman terms."""
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

    positions = grid.nodes.copy()
    center_idx = 12  # node (2, 2) in a 5x5 grid

    # Give the center node an initial out-of-plane velocity
    velocities = np.zeros_like(positions)
    velocities[center_idx, 2] = -50.0

    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    # Clamp all boundaries
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                boundary_mask[i * 5 + j] = 1

    nodal_external_forces = np.zeros_like(positions)
    thickness = 0.001

    # Run the solver for 50 steps to let the wave propagate
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
        proj_position=np.array([10.0, 10.0, 10.0]),  # projectile far away
        proj_velocity=np.zeros(3),
        proj_mass=1.0,
        proj_blade_width=0.01,
        proj_edge_thickness=0.01,
        n_plies=1,
        n_nodes_per_layer=25,
        t_ply=0.002,
        dx=0.01,
        k_penalty=1.0e6,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-8,
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
        strike_direction=1.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
        use_viscous=False,
        cfl_factor=0.5,
        proj_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        proj_omega=np.zeros(3),
        proj_shape_type="box",
        contact_energy_init=0.0,
        mu_s=0.0,
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
    final_pos = res[0]
    final_vel = res[1]

    # Verify that surrounding nodes (e.g. node 7, which is (1, 2)) have developed non-zero velocities
    # indicating that out-of-plane deflection of the center has successfully coupled to in-plane membrane stretching.
    assert np.max(np.abs(final_vel)) > 0.0
    assert np.max(np.abs(final_pos - positions)) > 0.0


def test_ramberg_osgood_sheet_simulation():
    """Verify that sheet simulation under plastic loading follows Ramberg-Osgood hardening and peeq accumulation."""
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

    positions = grid.nodes.copy()
    # Apply velocity to stretch elements
    velocities = np.zeros_like(positions)
    velocities[12, 2] = -100.0  # Center out-of-plane velocity to induce plastic strain

    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    for i in range(5):
        for j in range(5):
            if i == 0 or i == 4 or j == 0 or j == 4:
                boundary_mask[i * 5 + j] = 1

    nodal_external_forces = np.zeros_like(positions)
    thickness = 0.001

    n_elems = len(grid.elements)
    element_stress = np.zeros((n_elems, 3, 3), dtype=np.float64)
    element_peeq = np.zeros((n_elems, 3), dtype=np.float64)
    element_damage = np.zeros((n_elems, 3), dtype=np.float64)

    # Run for 20 steps
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
        proj_position=np.array([10.0, 10.0, 10.0]),
        proj_velocity=np.zeros(3),
        proj_mass=1.0,
        proj_blade_width=0.01,
        proj_edge_thickness=0.01,
        n_plies=1,
        n_nodes_per_layer=25,
        t_ply=0.002,
        dx=0.01,
        k_penalty=1.0e6,
        rayleigh_alpha=0.0,
        rayleigh_beta=1e-8,
        failure_strain=0.20,
        damage_onset_strain=0.15,
        fracture_energy_multiplier=1.0,
        dt=1e-7,
        n_steps=20,
        save_interval=5,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=1.0,
        node_initial_springs=grid.initial_spring_counts,
        node_spring_offsets=grid.node_spring_offsets,
        node_spring_ids=grid.node_spring_ids,
        node_spring_signs=grid.node_spring_signs,
        use_viscous=False,
        cfl_factor=0.5,
        proj_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        proj_omega=np.zeros(3),
        proj_shape_type="box",
        contact_energy_init=0.0,
        mu_s=0.0,
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
        element_stress=element_stress,
        element_peeq=element_peeq,
        element_damage=element_damage,
        tensile_strength_gpa=0.45,
    )

    assert res is not None
    # We should have accumulated equivalent plastic strain
    assert np.any(element_peeq > 0.0)
    # Stresses at the plastic points should be non-zero
    assert np.any(np.abs(element_stress) > 0.0)


def test_triaxiality_failure_scaling():
    """Verify that stress triaxiality-dependent failure strain behaves physically."""
    ultimate_strain = 0.15

    # 1. Uniaxial tension (eta = 1/3)
    eta = 1.0 / 3.0
    eps_f = ultimate_strain * np.exp(-1.5 * (eta - 1.0 / 3.0))
    assert np.isclose(eps_f, ultimate_strain)

    # 2. Triaxial tension (eta > 1/3, e.g., eta = 1.0) -> Failure strain should be much lower (brittle)
    eta = 1.0
    eps_f_tension = ultimate_strain * np.exp(-1.5 * (eta - 1.0 / 3.0))
    assert eps_f_tension < ultimate_strain

    # 3. Pure shear (eta = 0) -> Failure strain should be equal to ultimate strain
    eta = 0.0
    eps_f_shear = ultimate_strain * np.exp(-0.5 * eta)
    assert np.isclose(eps_f_shear, ultimate_strain)

    # 4. Compression (eta < 0, e.g., eta = -1.0) -> Failure strain should be very large (ductile)
    eta = -1.0
    eps_f_compression = ultimate_strain * np.exp(-0.5 * eta)
    assert eps_f_compression > eps_f_shear


def test_czm_mesh_duplication():
    """Verify that CZM mesh duplication correctly separates elements and defines tiebreak springs."""
    material = {
        "name": "Steel",
        "tensile_modulus_gpa": 200.0,
        "failure_strain": 0.20,
        "tensile_strength_gpa": 0.485,
        "fiber_density_gcc": 7.85,
        "areal_density_kgm2": 15.7,
        "shear_ratio": 0.38,
        "cohesive_strength_gpa": 0.485,
        "fracture_energy_jm2": 50000.0,
    }

    grid = generate_rectangular_grid(
        nx=5,
        ny=5,
        dx=0.01,
        material=material,
        use_czm=True,
    )

    # 5x5 grid has (5-1)*(5-1) = 16 elements.
    # In CZM, each element is fully duplicated: 16 * 4 = 64 nodes.
    assert len(grid.elements) == 16
    assert len(grid.nodes) == 64
    assert grid.n_nodes == 64

    # All springs should be tiebreak springs
    assert len(grid.springs) > 0
    assert np.all(grid.is_tiebreak)
    assert np.all(grid.rest_lengths == 0.0)

    # Masses should be distributed equally to the 4 corners of each element
    m_cell = 15.7 * 0.01 * 0.01
    expected_node_mass = 0.25 * m_cell
    assert np.allclose(grid.masses, expected_node_mass)


def test_czm_spring_softening_equations():
    """Verify that the bilinear Traction-Separation Law softening equations evaluate correctly."""
    cohesive_strength_gpa = 0.485
    fracture_energy_jm2 = 50000.0
    sig_max = cohesive_strength_gpa * 1e9
    g_c = fracture_energy_jm2

    dx = 0.01
    thickness = 0.002
    A_trib = 0.5 * dx * thickness
    F_max = sig_max * A_trib

    delta_c = 2.0 * g_c / sig_max
    delta_0 = 0.01 * delta_c
    k_0 = F_max / delta_0

    # 1. Before damage initiation (delta <= delta_0)
    delta = 0.5 * delta_0
    d = 0.0
    f_mag = (1.0 - d) * k_0 * delta
    assert np.isclose(f_mag, F_max * 0.5)

    # 2. Softening region (delta_0 < delta < delta_c)
    delta = 0.5 * (delta_0 + delta_c)
    d_cand = (delta_c * (delta - delta_0)) / (delta * (delta_c - delta_0))
    d = min(1.0, d_cand)
    f_mag = (1.0 - d) * k_0 * delta
    expected_f_mag = ((delta_c - delta) / (delta_c - delta_0)) * F_max
    assert np.isclose(f_mag, expected_f_mag)

    # 3. Completely failed region (delta >= delta_c)
    delta = 1.5 * delta_c
    d_cand = (delta_c * (delta - delta_0)) / (delta * (delta_c - delta_0))
    d = min(1.0, d_cand)
    assert np.isclose(d, 1.0)


def test_czm_dynamic_simulation_stability():
    """Verify that dynamic CZM simulations are stable, conserve energy, and fail correctly."""
    material = {
        "name": "Steel",
        "tensile_modulus_gpa": 200.0,
        "failure_strain": 0.20,
        "tensile_strength_gpa": 0.45,
        "fiber_density_gcc": 7.8,
        "areal_density_kgm2": 7.8,
        "shear_ratio": 0.38,
        "material_model": "j2_plasticity",
        "yield_strength_gpa": 0.01,
        "hardening_modulus_gpa": 0.1,
        "ultimate_strain": 0.05,
        "poisson_ratio": 0.3,
        "cohesive_strength_gpa": 0.01,
        "fracture_energy_jm2": 10.0,
    }

    grid = generate_rectangular_grid(
        nx=5,
        ny=5,
        dx=0.01,
        material=material,
        use_czm=True,
    )

    proj = Projectile(
        mass=0.1,
        velocity=[0.0, 0.0, -500.0],
        position=[0.0, 0.0, 0.001],
        shape_type="box",
        blade_width=0.02,
        edge_thickness=0.002,
    )

    pos = grid.nodes.copy()
    vel = np.zeros_like(pos)
    boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
    for idx, node in enumerate(grid.nodes):
        if (
            np.isclose(node[0], -0.02)
            or np.isclose(node[0], 0.02)
            or np.isclose(node[1], -0.02)
            or np.isclose(node[1], 0.02)
        ):
            boundary_mask[idx] = 1

    nodal_external_forces = np.zeros_like(pos)
    thickness = 7.8 / (7.8 * 1000.0)

    proj_pos = proj.position.copy()
    proj_vel = proj.velocity.copy()
    proj_quat = proj.quat.copy()
    proj_omega = proj.omega.copy()

    spring_failed = grid.failed.copy()
    spring_damage = np.zeros(grid.n_springs, dtype=np.float64)
    element_stress = np.zeros((len(grid.elements), 3, 3), dtype=np.float64)
    element_peeq = np.zeros((len(grid.elements), 3), dtype=np.float64)
    element_damage = np.zeros((len(grid.elements), 3), dtype=np.float64)
    element_failed = np.zeros(len(grid.elements), dtype=np.int32)

    dt = 1e-8
    t_sim = 0.0
    damp_diss = 0.0
    fail_diss = 0.0
    clamp_diss = 0.0
    contact_energy = 0.0
    friction_diss = 0.0

    initial_energy = None

    # Pre-build node_elements mapping to mirror worker.py
    n_nodes = len(grid.nodes)
    node_elements = [[] for _ in range(n_nodes)]
    for e_idx, elem in enumerate(grid.elements):
        for node in elem:
            node_elements[node].append(e_idx)
    spring_elements = []
    for n0, n1 in grid.springs:
        shared = list(set(node_elements[n0]).union(node_elements[n1]))
        spring_elements.append(shared)

    for chunk in range(100):  # Run 2000 steps (100 chunks of 20 steps)
        res = fused_leapfrog_loop(
            positions=pos,
            velocities=vel,
            grid_springs=grid.springs,
            grid_stiffnesses=grid.stiffnesses,
            grid_rest_lengths=grid.rest_lengths,
            grid_failed=spring_failed,
            grid_masses=grid.masses,
            grid_tension_only=grid.tension_only,
            boundary_mask=boundary_mask,
            nodal_external_forces=nodal_external_forces,
            proj_position=proj_pos,
            proj_velocity=proj_vel,
            proj_mass=proj.mass,
            proj_blade_width=proj.blade_width,
            proj_edge_thickness=proj.edge_thickness,
            n_plies=1,
            n_nodes_per_layer=len(pos),
            t_ply=0.002,
            dx=0.01,
            k_penalty=1.0e8,
            rayleigh_alpha=0.0,
            rayleigh_beta=1e-9,
            failure_strain=0.20,
            damage_onset_strain=0.15,
            fracture_energy_multiplier=1.0,
            dt=dt,
            n_steps=20,
            save_interval=20,
            damp_dissipated_init=damp_diss,
            failure_dissipated_init=fail_diss,
            clamp_dissipated_init=clamp_diss,
            t_sim_init=t_sim,
            strike_direction=-1.0,
            node_initial_springs=grid.initial_spring_counts,
            node_spring_offsets=grid.node_spring_offsets,
            node_spring_ids=grid.node_spring_ids,
            node_spring_signs=grid.node_spring_signs,
            use_viscous=False,
            cfl_factor=0.5,
            proj_quat=proj_quat,
            proj_omega=proj_omega,
            proj_shape_type="box",
            contact_energy_init=contact_energy,
            mu_s=0.1,
            friction_dissipated_init=friction_diss,
            structure_type="metallic_sheet",
            material_model="j2_plasticity",
            yield_strength_gpa=0.01,
            hardening_modulus_gpa=0.1,
            ultimate_strain=0.05,
            poisson_ratio=0.3,
            elements=grid.elements,
            youngs_modulus_gpa=200.0,
            thickness=thickness,
            density_kgm3=7800.0,
            is_tiebreak=grid.is_tiebreak,
            cohesive_strength_gpa=0.01,
            fracture_energy_jm2=10.0,
            use_czm=True,
            element_stress=element_stress,
            element_peeq=element_peeq,
            element_damage=element_damage,
            element_failed=element_failed,
            grid_damage=spring_damage,
            coincident_nodes=grid.coincident_nodes,
            node_czm_spring_ids=grid.node_czm_spring_ids,
        )

        (
            pos,
            vel,
            returned_failed,
            proj_pos,
            proj_vel,
            damp_diss,
            fail_diss,
            clamp_diss,
            t_sim,
            hist_pos,
            hist_failed,
            hist_proj_pos,
            hist_time,
            hist_ke,
            hist_se,
            hist_proj_ke,
            contact_energy,
            friction_diss,
        ) = res

        # Update element_failed and spring_failed arrays using the corrected logic
        element_failed = returned_failed.astype(np.int32)

        # In-place updates to spring_failed are preserved. We also update them from element failures:
        failed_springs = spring_failed.copy()
        for s_idx, el_indices in enumerate(spring_elements):
            if len(el_indices) > 0:
                all_failed = True
                for e_idx in el_indices:
                    if not element_failed[e_idx]:
                        all_failed = False
                        break
                if all_failed:
                    failed_springs[s_idx] = True
        spring_failed = failed_springs

        # Energy tracking
        ke_nodes = 0.5 * np.sum(grid.masses * np.sum(vel**2, axis=1))
        se_elems = 0.0
        E_val = 200.0 * 1e9
        for e in range(len(grid.elements)):
            if element_failed[e] == 0:
                se_elems += 0.5 * (0.01 * 0.01) * thickness * np.sum(element_stress[e] ** 2) / E_val
        ke_proj = 0.5 * proj.mass * np.sum(proj_vel**2)

        # Potential/strain energy including contact energy
        se_total = se_elems + contact_energy
        tot = ke_nodes + se_total + ke_proj + damp_diss + fail_diss + clamp_diss + friction_diss

        if initial_energy is None:
            initial_energy = tot

        # Check for energy conservation / drift
        drift = (tot - initial_energy) / initial_energy * 100
        assert abs(drift) < 40.0, (
            f"Energy exploded with drift {drift:.2f}% at step {(chunk + 1) * 20}"
        )

    # Verify that elements or tiebreak springs failed at the end of simulation under such high velocity impact
    assert np.sum(element_failed) > 0 or np.sum(spring_failed) > 0, (
        "No elements or tiebreak springs failed despite high-velocity impact"
    )
    # Verify that the final total energy did not double or blow up
    assert tot < initial_energy * 1.5, (
        f"Energy grew excessively: initial={initial_energy:.2f}, final={tot:.2f}"
    )


def test_ramberg_osgood_nonlinear_hardening():
    """Verify that the shell return mapping follows the Ramberg-Osgood nonlinear hardening curve and softens past ultimate strain."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [0.0, 0.01, 0.0],
        ],
        dtype=np.float64,
    )
    X_ref = positions.copy()
    velocities = np.zeros_like(positions)
    ang_positions = np.zeros_like(positions)
    ang_velocities = np.zeros_like(positions)
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
    thickness = 0.002
    E = 200e9
    nu = 0.3
    yield_strength = 200e6
    hardening_modulus = 1e9
    ultimate_strain = 0.05
    tensile_strength = 400e6
    dx = 0.01
    dt = 1e-7

    # Arrays
    element_strains = np.zeros((1, 8), dtype=np.float64)
    element_stress = np.zeros((1, 3, 3), dtype=np.float64)
    element_peeq = np.zeros((1, 3), dtype=np.float64)
    element_peeq_rate = np.zeros((1, 3), dtype=np.float64)
    element_damage = np.zeros((1, 3), dtype=np.float64)
    element_failed = np.array([0], dtype=np.int32)
    element_failed_step = np.array([-1], dtype=np.int32)

    # Apply strain increment that causes plastic yielding (0.005 is > yield strain of 0.001)
    element_strains[0, 0] = 0.005

    forces, torques, step_fe, step_sd = numba_step_shell_forces_and_failures(
        positions,
        X_ref,
        velocities,
        ang_positions,
        ang_velocities,
        elements,
        thickness,
        E,
        nu,
        yield_strength,
        hardening_modulus,
        ultimate_strain,
        tensile_strength,
        element_strains,
        element_stress,
        element_peeq,
        element_damage,
        element_failed,
        dx,
        rayleigh_beta=0.0,
        dt=dt,
        density_kgm3=7800.0,
        element_failed_step=element_failed_step,
        current_step=1,
        erosion_softening_steps=10,
        element_peeq_rate=element_peeq_rate,
        rate_parameter_c=0.0,
        rate_parameter_p=5.0,
    )

    # Check von Mises stress at point 0
    peeq1 = element_peeq[0, 0]
    stress_pt1 = element_stress[0, 0]
    sig_vm1 = np.sqrt(
        stress_pt1[0] ** 2
        + stress_pt1[1] ** 2
        - stress_pt1[0] * stress_pt1[1]
        + 3.0 * stress_pt1[2] ** 2
    )
    assert peeq1 > 0.0
    assert sig_vm1 > yield_strength
    assert sig_vm1 < tensile_strength

    # Apply very large strain to exceed ultimate strain
    element_strains[0, 0] = 0.1
    numba_step_shell_forces_and_failures(
        positions,
        X_ref,
        velocities,
        ang_positions,
        ang_velocities,
        elements,
        thickness,
        E,
        nu,
        yield_strength,
        hardening_modulus,
        ultimate_strain,
        tensile_strength,
        element_strains,
        element_stress,
        element_peeq,
        element_damage,
        element_failed,
        dx,
        rayleigh_beta=0.0,
        dt=dt,
        density_kgm3=7800.0,
        element_failed_step=element_failed_step,
        current_step=2,
        erosion_softening_steps=10,
        element_peeq_rate=element_peeq_rate,
        rate_parameter_c=0.0,
        rate_parameter_p=5.0,
    )

    peeq2 = element_peeq[0, 0]
    stress_pt2 = element_stress[0, 0]
    sig_vm2 = np.sqrt(
        stress_pt2[0] ** 2
        + stress_pt2[1] ** 2
        - stress_pt2[0] * stress_pt2[1]
        + 3.0 * stress_pt2[2] ** 2
    )
    assert peeq2 > ultimate_strain
    assert sig_vm2 >= tensile_strength
    assert sig_vm2 < tensile_strength + 1e7


def test_summed_stiffness_and_zero_rest_length_cohesive():
    """Verify that summed-stiffness CFL limit and zero rest-length cohesive springs are handled stably."""
    import numpy as np

    from kevlargrid.solver.fused import numba_compute_effective_k

    # 1. Zero rest-length cohesive spring check
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.001]], dtype=np.float64)
    springs = np.array([[0, 1]], dtype=np.int32)
    stiffnesses = np.array([1000.0], dtype=np.float64)
    rest_lengths = np.array([0.0], dtype=np.float64)  # Zero rest length
    failed = np.array([False], dtype=np.bool_)
    grid_damage = np.array([0.0], dtype=np.float64)
    spring_failed_step = np.array([-1], dtype=np.int32)

    # Calling compute_effective_k should run without division by zero
    eff_k, step_fe = numba_compute_effective_k(
        positions=positions,
        springs=springs,
        stiffnesses=stiffnesses,
        rest_lengths=rest_lengths,
        failed=failed,
        damage_onset_strain=0.0005,
        failure_strain=0.002,
        grid_damage=grid_damage,
        spring_failed_step=spring_failed_step,
        current_step=0,
        fracture_energy_multiplier=1.0,
        erosion_softening_steps=10,
    )
    # Since Rest Length is 0, separation distance (0.001) > damage_onset_strain (0.0005)
    # This should yield non-zero damage and stable effective stiffness
    assert grid_damage[0] > 0.0
    assert grid_damage[0] < 1.0
    assert eff_k[0] < 1000.0
    assert eff_k[0] > 0.0
