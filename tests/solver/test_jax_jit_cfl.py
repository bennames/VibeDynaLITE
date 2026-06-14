from __future__ import annotations

import sys
import os
import importlib
import pytest
import numpy as np

from kevlargrid.solver import backend


def test_jax_jit_cfl_compilation() -> None:
    """Verify that fused_leapfrog_loop compiles successfully with JAX backend under both dynamic and static CFL factors."""
    if not backend.HAS_JAX:
        pytest.skip("JAX backend is not installed/available in this environment.")

    # Save original backend env and state S7.14
    old_env = os.environ.get("KEVLARGRID_BACKEND")
    os.environ["KEVLARGRID_BACKEND"] = "jax"

    solver_modules = [
        "kevlargrid.solver.backend",
        "kevlargrid.solver.damping",
        "kevlargrid.solver.energy",
        "kevlargrid.solver.failure",
        "kevlargrid.solver.forces",
        "kevlargrid.solver.integrator",
        "kevlargrid.solver.fused",
    ]

    try:
        # Reload solver backend and related modules in dependency order to trigger JAX binding of imported functions
        for mod in solver_modules:
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])

        import jax
        from kevlargrid.solver.fused import fused_leapfrog_loop

        py_func = getattr(fused_leapfrog_loop, "py_func", fused_leapfrog_loop)
        
        # Force JAX JIT compilation using jax.jit directly S7.14
        jax_jit_fn = jax.jit(
            py_func,
            static_argnames=(
                "n_plies",
                "n_nodes_per_layer",
                "n_steps",
                "save_interval",
                "use_viscous",
                "cfl_factor",
            ),
        )
        
        # Define double-precision dummy arguments to prevent type unification mismatches under Numba/fallback modes
        n_nodes = 4
        n_springs = 2
        positions = np.zeros((n_nodes, 3), dtype=np.float64)
        velocities = np.zeros((n_nodes, 3), dtype=np.float64)
        grid_springs = np.array([[0, 1], [2, 3]], dtype=np.int32)
        grid_stiffnesses = np.array([1000.0, 1000.0], dtype=np.float64)
        grid_rest_lengths = np.array([1.0, 1.0], dtype=np.float64)
        grid_failed = np.zeros(n_springs, dtype=bool)
        grid_masses = np.ones(n_nodes, dtype=np.float64)
        grid_tension_only = np.zeros(n_springs, dtype=bool)
        boundary_mask = np.zeros(n_nodes, dtype=bool)
        nodal_external_forces = np.zeros((n_nodes, 3), dtype=np.float64)
        proj_position = np.array([0.0, 0.0, 0.1], dtype=np.float64)
        proj_velocity = np.array([0.0, 0.0, -10.0], dtype=np.float64)
        proj_mass = 0.5
        proj_blade_width = 0.1
        proj_edge_thickness = 0.01
        
        node_initial_springs = np.array([1, 1, 1, 1], dtype=np.int32)
        node_spring_offsets = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        node_spring_ids = np.array([0, 0, 1, 1], dtype=np.int32)
        node_spring_signs = np.array([1, -1, 1, -1], dtype=np.float64)

        # Call JAX-wrapped function with dynamic CFL enabled (cfl_factor > 0.0)
        res_dynamic = jax_jit_fn(
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
            proj_blade_width=proj_blade_width,
            proj_edge_thickness=proj_edge_thickness,
            n_plies=1,
            n_nodes_per_layer=4,
            t_ply=0.002,
            dx=1.0,
            k_penalty=10000.0,
            rayleigh_alpha=0.0,
            rayleigh_beta=0.0,
            failure_strain=0.05,
            damage_onset_strain=0.03,
            fracture_energy_multiplier=1.5,
            dt=1e-6,
            n_steps=2,
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
            cfl_factor=0.8,
        )

        # Call JAX-wrapped function with static CFL (cfl_factor = -1.0)
        res_static = jax_jit_fn(
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
            proj_blade_width=proj_blade_width,
            proj_edge_thickness=proj_edge_thickness,
            n_plies=1,
            n_nodes_per_layer=4,
            t_ply=0.002,
            dx=1.0,
            k_penalty=10000.0,
            rayleigh_alpha=0.0,
            rayleigh_beta=0.0,
            failure_strain=0.05,
            damage_onset_strain=0.03,
            fracture_energy_multiplier=1.5,
            dt=1e-6,
            n_steps=2,
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
            cfl_factor=-1.0,
        )
        
        assert len(res_dynamic) == 16
        assert len(res_static) == 16
    finally:
        # Restore original backend env and state S7.14
        if old_env is not None:
            os.environ["KEVLARGRID_BACKEND"] = old_env
        else:
            os.environ.pop("KEVLARGRID_BACKEND", None)
        for mod in solver_modules:
            if mod in sys.modules:
                importlib.reload(sys.modules[mod])
