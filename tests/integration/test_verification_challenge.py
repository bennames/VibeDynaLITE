import sys
import numpy as np
from pathlib import Path

# Import solver components
from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.timestep import compute_cfl_timestep
from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy

def run_simulation(v_strike: float, nx: int, ny: int, n_plies: int, k_penalty: float, mu_s: float, max_steps: int = 1500, grid=None, debug=False) -> dict:
    dx = 0.008  # 250mm panel / 31 nodes
    n_nodes_per_layer = nx * ny
    
    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.002,
    }
    
    if grid is None:
        grid = generate_rectangular_grid(nx, ny, dx, material_kev29, n_plies=n_plies, t_ply=0.0001)
    else:
        grid.failed[:] = False
    
    # Boundary conditions: Clamped on all outer edges
    boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
    for ply in range(n_plies):
        offset = ply * n_nodes_per_layer
        for i in range(nx):
            for j in range(ny):
                if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                    boundary_mask[offset + i * ny + j] = True
                    
    proj_mass = 0.0011  # 1.10 grams
    R = 0.00273         # 5.46 mm diameter
    L = 0.006           # 6 mm length
    I_zz = 0.5 * proj_mass * R**2
    I_xx = (1.0 / 12.0) * proj_mass * (3.0 * R**2 + L**2)
    proj_inertia_inv = np.diag([1.0/I_xx, 1.0/I_xx, 1.0/I_zz])
    
    proj_pos = np.array([0.0, 0.0, -0.002], dtype=np.float64)
    proj_vel = np.array([0.0, 0.0, v_strike], dtype=np.float64)
    proj_omega = np.zeros(3, dtype=np.float64)
    proj_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    
    dt = compute_cfl_timestep(grid.stiffnesses, grid.masses, dx, 0.1)
    
    node_initial_springs = grid.initial_spring_counts
    node_spring_offsets = grid.node_spring_offsets
    node_spring_ids = grid.node_spring_ids
    node_spring_signs = grid.node_spring_signs
    
    initial_energy = 0.5 * proj_mass * (v_strike**2)
    save_interval = 20
    
    t_sim = 0.0
    damp_dissipated = 0.0
    failure_dissipated = 0.0
    clamp_dissipated = 0.0
    contact_energy = 0.0
    friction_dissipated = 0.0
    
    hist_total_energy = []
    grid_damage = np.zeros(grid.n_springs, dtype=np.float64)
    failed = grid.failed
    
    pos = grid.nodes.copy()
    vel = np.zeros_like(pos)
    step = 0
    
    ext_forces = np.zeros((grid.n_nodes, 3))
    
    if debug:
        print(f"Initial Energy: {initial_energy:.3f} J")
        
    while step < max_steps:
        (
            pos,
            vel,
            failed,
            proj_pos_new,
            proj_vel_new,
            damp_dissipated,
            failure_dissipated,
            clamp_dissipated,
            t_sim,
            _, _, _, _, _, _, _, _,
            contact_energy,
            friction_dissipated,
        ) = taichi_leapfrog_loop(
            pos,
            vel,
            grid.springs,
            grid.stiffnesses,
            grid.rest_lengths,
            failed,
            grid.masses,
            grid.tension_only,
            boundary_mask,
            ext_forces,
            proj_pos,
            proj_vel,
            proj_mass,
            0.0,
            0.0,
            n_plies,
            n_nodes_per_layer,
            0.0001,
            dx,
            k_penalty,
            0.0,   # rayleigh_alpha
            5e-8,  # rayleigh_beta
            0.038, # failure_strain
            0.0228, # damage_onset_strain
            1.5,   # fracture_energy_multiplier
            dt,
            save_interval,
            save_interval,
            damp_dissipated,
            failure_dissipated,
            clamp_dissipated,
            t_sim,
            1.0,
            node_initial_springs,
            node_spring_offsets,
            node_spring_ids,
            node_spring_signs,
            use_viscous=False,
            cfl_factor=0.1,
            mu_s=mu_s,
            proj_quat=proj_quat,
            proj_omega=proj_omega,
            proj_shape_type="cylinder",
            proj_radius=R,
            proj_length=L,
            proj_inertia_inv=proj_inertia_inv,
            grid_damage=grid_damage,
            contact_energy_init=contact_energy,
            friction_dissipated_init=friction_dissipated,
        )
        
        proj_pos = proj_pos_new
        proj_vel = proj_vel_new
        step += save_interval
        
        # Calculate energies
        ke_nodes = 0.5 * np.sum(grid.masses * np.sum(vel**2, axis=1))
        p1 = pos[grid.springs[:, 0]]
        p2 = pos[grid.springs[:, 1]]
        lens = np.sqrt(np.sum((p2 - p1)**2, axis=1))
        strains = (lens - grid.rest_lengths) / grid.rest_lengths
        strains_eff = np.where(grid.tension_only & (strains < 0.0), 0.0, strains)
        se_springs_array = 0.5 * grid.stiffnesses * (1.0 - grid_damage) * (strains_eff * grid.rest_lengths)**2
        se_springs = float(np.sum(np.where(failed, 0.0, se_springs_array)))
        ke_proj = 0.5 * proj_mass * np.sum(proj_vel**2)
        
        total_energy = ke_nodes + se_springs + ke_proj + damp_dissipated + failure_dissipated + clamp_dissipated + contact_energy + friction_dissipated
        hist_total_energy.append(total_energy)
        
        if debug:
            print(f"Step {step:4d}: Total={total_energy:6.2f} | KE_nodes={ke_nodes:6.2f} | SE_springs={se_springs:6.2f} | KE_proj={ke_proj:6.2f} | Damp={damp_dissipated:6.2f} | Fail={failure_dissipated:6.2f} | Contact={contact_energy:6.2f} | Friction={friction_dissipated:6.2f}")
            
        if proj_vel[2] <= 0.0:
            break
        if proj_pos[2] > (n_plies * 0.0001 + 0.005) and proj_vel[2] > 0.0:
            break
            
    residual_vel = max(0.0, float(proj_vel[2]))
    energy_drift = float(np.max(np.abs(np.array(hist_total_energy) - initial_energy)) / initial_energy)
    
    return {
        "residual_velocity": residual_vel,
        "energy_drift_pct": energy_drift * 100,
        "failed_springs": failed,
        "springs": grid.springs,
        "tension_only": grid.tension_only,
    }

def fit_jonas_laval(v_strike: np.ndarray, v_residual: np.ndarray) -> tuple[float, float]:
    best_v50 = 200.0
    best_alpha = 1.0
    min_sse = 1e20
    for v50 in np.linspace(50.0, 400.0, 351):
        for alpha in np.linspace(0.7, 1.3, 61):
            pred = np.zeros_like(v_strike)
            mask = v_strike > v50
            pred[mask] = alpha * np.sqrt(v_strike[mask] ** 2 - v50 ** 2)
            sse = np.sum((v_residual - pred) ** 2)
            if sse < min_sse:
                min_sse = sse
                best_v50 = v50
                best_alpha = alpha
    return float(best_v50), float(best_alpha)

def test_print_energy_debug():
    print("\n[DEBUG] Running a single simulation case to print detailed energy terms...")
    res = run_simulation(500.0, nx=31, ny=31, n_plies=13, k_penalty=2e5, mu_s=0.20, max_steps=300, grid=None, debug=True)

def test_jonas_laval_curve_fit():
    print("\n[VERIFICATION] Running Jonas-Laval Curve Fit sweep...")
    dx = 0.008
    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.002,
    }
    grid = generate_rectangular_grid(31, 31, dx, material_kev29, n_plies=1, t_ply=0.0001)
    
    strike_vels = np.array([150, 200, 250, 300, 350, 400], dtype=np.float64)
    residual_vels = []
    for v_s in strike_vels:
        res = run_simulation(v_s, nx=31, ny=31, n_plies=1, k_penalty=2e5, mu_s=0.20, max_steps=600, grid=grid)
        residual_vels.append(res["residual_velocity"])
        print(f"  Strike: {v_s:3.0f} m/s | Residual: {res['residual_velocity']:5.1f} m/s")
    
    residual_vels = np.array(residual_vels, dtype=np.float64)
    v50_fit, alpha_fit = fit_jonas_laval(strike_vels, residual_vels)
    print(f"  --> Fitted Jonas-Laval V50: {v50_fit:.1f} m/s, alpha: {alpha_fit:.3f}")

def test_energy_drift_verification():
    print("\n[VERIFICATION] Running Energy Drift tests with different parameters...")
    dx = 0.008
    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.002,
    }
    grid = generate_rectangular_grid(31, 31, dx, material_kev29, n_plies=13, t_ply=0.0001)
    
    frictions = [0.0, 0.20, 0.40]
    stiffnesses = [2e5, 1e6, 5e6]
    
    for mu in frictions:
        for k in stiffnesses:
            res = run_simulation(500.0, nx=31, ny=31, n_plies=13, k_penalty=k, mu_s=mu, max_steps=400, grid=grid)
            print(f"  Friction: {mu:.2f} | Stiffness: {k:.1e} | Energy Drift: {res['energy_drift_pct']:.3f}%")

def test_physical_plausibility_failure_modes():
    print("\n[VERIFICATION] Running ply failure mode analysis...")
    dx = 0.008
    material_kev29 = {
        "tensile_modulus_gpa": 70.5,
        "areal_density_kgm2": 0.475,
        "fiber_density_gcc": 1.44,
        "failure_strain": 0.038,
        "shear_ratio": 0.002,
    }
    grid = generate_rectangular_grid(31, 31, dx, material_kev29, n_plies=13, t_ply=0.0001)
    
    # Run a perforating case to check failure patterns in plies
    res = run_simulation(450.0, nx=31, ny=31, n_plies=13, k_penalty=1e6, mu_s=0.20, max_steps=800, grid=grid)
    
    failed = res["failed_springs"]
    springs = res["springs"]
    tension_only = res["tension_only"]
    
    n_nodes_per_layer = 31 * 31
    n_springs_per_ply = len(tension_only) // 13
    
    print("  Failed springs per layer:")
    for ply in range(13):
        start = ply * n_springs_per_ply
        end = (ply + 1) * n_springs_per_ply
        ply_failed = failed[start:end]
        ply_tension_only = tension_only[start:end]
        
        failed_ortho = np.sum(ply_failed & ply_tension_only)  # Tensile/Strain failure
        failed_diag = np.sum(ply_failed & ~ply_tension_only)  # Shear failure
        total_failed = np.sum(ply_failed)
        
        if total_failed > 0:
            shear_ratio = failed_diag / total_failed
            strain_ratio = failed_ortho / total_failed
        else:
            shear_ratio, strain_ratio = 0.0, 0.0
            
        print(f"    Ply {ply:2d}: Total Failed = {total_failed:4d} | Shear/Diag = {failed_diag:4d} ({shear_ratio*100:5.1f}%) | Tensile/Ortho = {failed_ortho:4d} ({strain_ratio*100:5.1f}%)")
