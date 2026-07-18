import numpy as np

from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.projectile import Projectile
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
from kevlargrid.solver.timestep import compute_cfl_timestep

MOCK_MATERIAL = {
    "tensile_modulus_gpa": 71.0,
    "areal_density_kgm2": 0.47,
    "fiber_density_gcc": 1.44,
    "shear_ratio": 0.0004,
}


def run_smith(nx=201, num_steps=150):
    ny, dx = 1, 0.01
    n_nodes = nx * ny
    grid = generate_rectangular_grid(nx, ny, dx, MOCK_MATERIAL)

    center = nx // 2
    k = grid.stiffnesses[0]
    m = grid.masses[center]
    c_fiber = dx * np.sqrt(k / m)

    v_proj = 100.0
    proj = Projectile(
        mass=1.0,
        velocity=[0.0, 0.0, v_proj],
        position=[0.0, 0.0, -0.001],
        blade_width=0.005,
        edge_thickness=0.002,
    )

    # Solve analytical Smith's equation
    eps_left = 1e-6
    eps_right = 0.5
    for _ in range(50):
        eps_mid = 0.5 * (eps_left + eps_right)
        v_analytical = c_fiber * eps_mid * np.sqrt((1.0 + eps_mid) * (2.0 + eps_mid))
        if v_analytical < v_proj:
            eps_left = eps_mid
        else:
            eps_right = eps_mid
    eps_analytical = eps_mid
    u_analytical = (
        c_fiber * np.sqrt(eps_analytical * (1.0 + eps_analytical)) - c_fiber * eps_analytical
    )

    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.4)
    positions = grid.nodes.copy()
    velocities = np.zeros_like(positions)

    boundary_mask = np.zeros(n_nodes, dtype=bool)
    boundary_mask[0] = True
    boundary_mask[-1] = True

    (
        positions_out,
        _velocities_out,
        _grid_failed,
        _proj_pos,
        _proj_vel,
        *_,
    ) = taichi_leapfrog_loop(
        positions,
        velocities,
        grid.springs,
        grid.stiffnesses,
        grid.rest_lengths,
        grid.failed,
        grid.masses,
        grid.tension_only,
        boundary_mask,
        np.zeros((n_nodes, 3)),
        proj.position,
        proj.velocity,
        proj.mass,
        proj.blade_width,
        proj.edge_thickness,
        1,
        n_nodes,
        0.002,
        dx,
        1e7,
        0.0,
        0.0,
        0.1,
        0.06,
        1.0,
        dt,
        num_steps,
        num_steps,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        grid.initial_spring_counts,
        grid.node_spring_offsets,
        grid.node_spring_ids,
        grid.node_spring_signs,
    )

    z_deflections = positions_out[:, 2]
    t_elapsed = num_steps * dt
    print(f"u_analytical: {u_analytical:.4f}")
    for thresh_frac in [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.15,
        0.20,
        0.25,
    ]:
        threshold = thresh_frac * z_deflections[center]
        kink_node = center
        for idx in range(center, nx - 1):
            if z_deflections[idx] >= threshold > z_deflections[idx + 1]:
                z0 = z_deflections[idx]
                z1 = z_deflections[idx + 1]
                frac = (z0 - threshold) / (z0 - z1)
                kink_node = idx + frac
                break
        dist_kink = (kink_node - center) * dx
        u_num = dist_kink / t_elapsed
        err = np.abs(u_num - u_analytical) / u_analytical
        print(f"  thresh_frac={thresh_frac:.2f}: u_numerical={u_num:.4f}, error={err:.4%}")


if __name__ == "__main__":
    run_smith()
