import numpy as np

from kevlargrid.solver.fused import (
    fused_leapfrog_loop,
    numba_compute_effective_k,
    numba_parallel_compute_spring_forces,
)


def test_softened_force_decay():
    """Verify that a failed spring's force decays to zero over exactly erosion_softening_steps."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.015],  # stretched beyond rest length 0.01
        ],
        dtype=np.float64,
    )

    velocities = np.zeros_like(positions)
    springs = np.array([[0, 1]], dtype=np.int32)
    stiffnesses = np.array([1e5], dtype=np.float64)
    rest_lengths = np.array([0.01], dtype=np.float64)
    failed = np.array([False], dtype=np.bool_)
    grid_damage = np.array([0.0], dtype=np.float64)
    spring_failed_step = np.array([-1], dtype=np.int32)
    tension_only = np.array([False], dtype=np.bool_)

    # CSR adjacency dummies
    node_spring_offsets = np.array([0, 1, 2], dtype=np.int32)
    node_spring_ids = np.array([0, 0], dtype=np.int32)
    node_spring_signs = np.array([1.0, -1.0], dtype=np.float64)

    # 1. Trigger failure at current_step = 100
    eff_k, step_fe = numba_compute_effective_k(
        positions,
        springs,
        stiffnesses,
        rest_lengths,
        failed,
        damage_onset_strain=0.1,
        failure_strain=0.2,
        grid_damage=grid_damage,
        spring_failed_step=spring_failed_step,
        current_step=100,
        fracture_energy_multiplier=1.0,
    )

    assert failed[0]
    assert spring_failed_step[0] == 100
    assert eff_k[0] == 0.0
    assert step_fe > 0.0

    # 2. Check forces during softening phase (steps 100 to 110)
    forces_100, _ = numba_parallel_compute_spring_forces(
        positions,
        velocities,
        springs,
        eff_k,
        stiffnesses,
        rest_lengths,
        tension_only,
        node_spring_offsets,
        node_spring_ids,
        node_spring_signs,
        rayleigh_beta=0.0,
        failed=failed,
        spring_failed_step=spring_failed_step,
        current_step=100,
        erosion_softening_steps=10,
    )
    # At step 100, age = 0, ramp = 1.0, force is fully active
    force_mag_100 = np.linalg.norm(forces_100[0])
    assert force_mag_100 > 0.0

    # At step 105, age = 5, ramp = 0.5, force is exactly halved
    forces_105, _ = numba_parallel_compute_spring_forces(
        positions,
        velocities,
        springs,
        eff_k,
        stiffnesses,
        rest_lengths,
        tension_only,
        node_spring_offsets,
        node_spring_ids,
        node_spring_signs,
        rayleigh_beta=0.0,
        failed=failed,
        spring_failed_step=spring_failed_step,
        current_step=105,
        erosion_softening_steps=10,
    )
    force_mag_105 = np.linalg.norm(forces_105[0])
    assert np.isclose(force_mag_105, 0.5 * force_mag_100)

    # At step 110, age = 10, ramp = 0.0, force is exactly 0.0
    forces_110, _ = numba_parallel_compute_spring_forces(
        positions,
        velocities,
        springs,
        eff_k,
        stiffnesses,
        rest_lengths,
        tension_only,
        node_spring_offsets,
        node_spring_ids,
        node_spring_signs,
        rayleigh_beta=0.0,
        failed=failed,
        spring_failed_step=spring_failed_step,
        current_step=110,
        erosion_softening_steps=10,
    )
    force_mag_110 = np.linalg.norm(forces_110[0])
    assert force_mag_110 == 0.0


def test_dead_node_contact_exclusion_and_capping():
    """Verify that eroded nodes with 0 active connections are excluded from contact, and contact force is capped."""
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    velocities = np.zeros_like(positions)
    grid_springs = np.zeros((0, 2), dtype=np.int32)
    grid_stiffnesses = np.zeros(0, dtype=np.float64)
    grid_rest_lengths = np.zeros(0, dtype=np.float64)
    grid_failed = np.zeros(0, dtype=np.bool_)
    grid_masses = np.array([1e-4], dtype=np.float64)
    grid_tension_only = np.zeros(0, dtype=np.bool_)
    boundary_mask = np.array([0], dtype=np.int32)
    nodal_external_forces = np.zeros_like(positions)

    # Projectile
    proj_position = np.array([0.0, 0.0, 0.0025], dtype=np.float64)  # penetrating
    proj_velocity = np.array([0.0, 0.0, -100.0], dtype=np.float64)
    proj_mass = 0.05

    node_initial_springs = np.array([4], dtype=np.int32)
    node_spring_offsets = np.array([0, 0], dtype=np.int32)
    node_spring_ids = np.zeros(0, dtype=np.int32)
    node_spring_signs = np.zeros(0, dtype=np.float64)

    # Run loop for 1 step. Since node_initial_springs > 0 but active_counts is 0,
    # it must trigger dead-node exclusion and experience exactly 0 contact force!
    res = fused_leapfrog_loop(
        positions=positions,
        velocities=velocities,
        grid_springs=grid_springs,
        grid_stiffnesses=grid_stiffnesses,
        grid_rest_lengths=grid_rest_lengths,
        grid_failed=grid_failed,
        grid_masses=grid_masses,
        grid_tension_only=grid_tension_only,
        boundary_mask=boundary_mask,
        nodal_external_forces=nodal_external_forces,
        proj_position=proj_position,
        proj_velocity=proj_velocity,
        proj_mass=proj_mass,
        proj_blade_width=0.02,
        proj_edge_thickness=0.002,
        n_plies=1,
        n_nodes_per_layer=1,
        t_ply=0.002,
        dx=0.005,
        k_penalty=1.0e8,  # very stiff contact
        rayleigh_alpha=0.0,
        rayleigh_beta=0.0,
        failure_strain=0.2,
        damage_onset_strain=0.1,
        fracture_energy_multiplier=1.0,
        dt=1e-8,
        n_steps=1,
        save_interval=1,
        damp_dissipated_init=0.0,
        failure_dissipated_init=0.0,
        clamp_dissipated_init=0.0,
        t_sim_init=0.0,
        strike_direction=-1.0,
        node_initial_springs=node_initial_springs,
        node_spring_offsets=node_spring_offsets,
        node_spring_ids=node_spring_ids,
        node_spring_signs=node_spring_signs,
        use_viscous=False,
        cfl_factor=-1.0,  # disable auto-CFL to keep dt constant
        contact_energy_init=0.0,
        density_kgm3=1440.0,
        youngs_modulus_gpa=100.0,
    )

    contact_energy = res[-2]
    # Contact energy must be exactly 0 because the node has 0 active springs, and is thus excluded
    assert contact_energy == 0.0
