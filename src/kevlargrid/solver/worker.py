"""Background process worker executing KevlarGrid explicit dynamics simulation loops.

Runs the physics integration in a separate OS process to bypass the Python GIL and
prevent Objective-C class name collisions between DearPyGui and Taichi on macOS.
"""

import logging
import time
import traceback

import numpy as np


def run_solver_process(config: dict, queue, pipe) -> None:
    """Target function for the solver subprocess.

    Parameters
    ----------
    config : dict
        Full session settings dictionary.
    queue : multiprocessing.Queue
        Queue to send telemetry/state updates back to GUI.
    pipe : multiprocessing.connection.Connection
        Pipe to receive controls (pause, resume, stop) from GUI.
    """
    try:
        logger = logging.getLogger("kevlargrid")
        # 1. Setup simulation components
        mat = config["material"]
        grid_cfg = config["grid"]
        proj_cfg = config["projectile"]
        sim_cfg = config["simulation"]

        solver_backend = sim_cfg.get("backend", "taichi")
        import os

        os.environ["KEVLARGRID_BACKEND"] = solver_backend

        # Import solver modules after backend environment variable is set
        from kevlargrid.solver import backend
        from kevlargrid.solver.grid import generate_rectangular_grid
        from kevlargrid.solver.projectile import Projectile, check_termination, update_contact_zone
        from kevlargrid.solver.timestep import compute_cfl_timestep

        backend.BACKEND = solver_backend

        # Configure dynamic Numba threads
        num_threads = sim_cfg.get("num_threads")
        if solver_backend == "numba":
            if num_threads:
                backend.set_numba_threads(num_threads)
            else:
                try:
                    import psutil

                    num_threads = psutil.cpu_count(logical=False) or os.cpu_count() or 4
                except ImportError:
                    num_threads = os.cpu_count() or 4
                backend.set_numba_threads(num_threads)

        # Setup File Logger
        log_to_file = sim_cfg.get("log_to_file", True)
        if log_to_file:
            import os

            os.makedirs("logs", exist_ok=True)
            log_filename = f"logs/simulation_run_{int(time.time())}.log"
            file_handler = logging.FileHandler(log_filename, encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)

        nx, ny, dx = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["dx"]
        n_plies = grid_cfg["n_plies"]
        t_ply = grid_cfg["t_ply"]

        # Build grid
        grid = generate_rectangular_grid(
            nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
        )

        # Build boundary mask
        boundary_mask = np.zeros(grid.n_nodes, dtype=np.int32)
        n_nodes_per_layer = nx * ny
        n_layers = n_plies if (t_ply is not None and n_plies > 1) else 1
        if grid_cfg["boundary_type"] == "fixed":
            for ply in range(n_layers):
                offset = ply * n_nodes_per_layer
                for i in range(nx):
                    for j in range(ny):
                        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                            boundary_mask[offset + i * ny + j] = 1
        elif grid_cfg["boundary_type"] == "non-reflecting":
            for ply in range(n_layers):
                offset = ply * n_nodes_per_layer
                for i in range(nx):
                    for j in range(ny):
                        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                            boundary_mask[offset + i * ny + j] = 2

        # Nodal external forces setup
        if "nodal_external_forces" in config:
            nodal_external_forces = np.array(config["nodal_external_forces"], dtype=np.float64)
        else:
            nodal_external_forces = np.zeros((grid.n_nodes, 3), dtype=np.float64)

        # Calculate timestep
        k_penalty = sim_cfg.get("k_penalty", 10.0 * np.mean(grid.stiffnesses))
        auto_cfl = sim_cfg.get("auto_cfl", True)
        if auto_cfl:
            k_max_effective = max(np.max(grid.stiffnesses), k_penalty)
            dt = compute_cfl_timestep(
                np.array([k_max_effective]), grid.masses, dx, sim_cfg["cfl_factor"]
            )
        else:
            dt = sim_cfg.get("dt", 1.5e-7)

        # Determine strike direction based on initial velocity Z-component
        strike_direction = -1.0 if proj_cfg["velocity"][2] < 0.0 else 1.0

        # Projectile setup
        shape_type = proj_cfg.get("shape_type", "box")
        radius = proj_cfg.get("radius", 0.005)
        length = proj_cfg.get("length", 0.01)
        edge_thickness = proj_cfg.get("edge_thickness", 0.005)

        # Calculate half-height along Z axis based on shape
        s_lower = shape_type.lower()
        if s_lower == "box":
            h_half = edge_thickness / 2.0
        elif s_lower == "sphere":
            h_half = radius
        elif s_lower == "cylinder" or s_lower == "bullet":
            h_half = length / 2.0
        else:
            h_half = radius

        # Check for initial penetration Z-overlap
        z_pos = proj_cfg["position"][2]
        n_layers = n_plies if (t_ply is not None and n_plies > 1) else 1
        t_ply_val = t_ply if t_ply is not None else 0.0
        z_grid_bottom = 0.0
        z_grid_top = (n_layers - 1) * t_ply_val

        if strike_direction > 0.0:
            # Striking from below: top of projectile must start below or at grid bottom (0.0)
            if z_pos + h_half > z_grid_bottom:
                z_pos = z_grid_bottom - h_half
                logger.warning(
                    "Projectile initially overlaps Kevlar grid. Adjusted Z starting position to %f m to ensure tangent contact.",
                    z_pos,
                )
        else:
            # Striking from above: bottom of projectile must start above or at grid top
            if z_pos - h_half < z_grid_top:
                z_pos = z_grid_top + h_half
                logger.warning(
                    "Projectile initially overlaps Kevlar grid. Adjusted Z starting position to %f m to ensure tangent contact.",
                    z_pos,
                )

        proj_position = np.array(
            [proj_cfg["position"][0], proj_cfg["position"][1], z_pos], dtype=np.float64
        )

        proj = Projectile(
            mass=proj_cfg["mass"],
            velocity=proj_cfg["velocity"],
            position=proj_position,
            shape_type=shape_type,
            blade_width=proj_cfg.get("blade_width", 0.02),
            edge_thickness=edge_thickness,
            radius=radius,
            length=length,
            edge_radius=proj_cfg.get("edge_radius", 0.0),
            ogive_multiplier=proj_cfg.get("ogive_multiplier", 2.0),
            span=proj_cfg.get("span", 0.05),
            root_chord=proj_cfg.get("root_chord", 0.01),
            tip_chord=proj_cfg.get("tip_chord", 0.005),
            twist=proj_cfg.get("twist", 15.0),
            thickness_ratio=proj_cfg.get("thickness_ratio", 12.0),
            tip_radius=proj_cfg.get("tip_radius", 0.002),
            omega=proj_cfg.get("omega", None),
            quat=proj_cfg.get("quat", None),
        )

        positions = grid.nodes.copy()
        velocities = np.zeros_like(positions)
        damping_model = sim_cfg.get("damping_model", "rayleigh")
        if damping_model == "viscous":
            rayleigh_alpha = sim_cfg.get("damping_coefficient", 0.05)
            rayleigh_beta = 0.0
        else:
            rayleigh_alpha = sim_cfg.get("rayleigh_alpha", 0.0)
            rayleigh_beta = sim_cfg.get("rayleigh_beta", 1e-9)
        duration = sim_cfg["duration"]

        failure_strain = mat["failure_strain"]
        damage_onset_strain = mat.get("damage_onset_strain", 0.6 * failure_strain)
        fracture_energy_multiplier = mat.get("fracture_energy_multiplier", 1.5)

        t_sim = 0.0
        damp_dissipated = 0.0
        failure_dissipated = 0.0
        clamp_dissipated = 0.0
        contact_energy = 0.0
        friction_dissipated = 0.0

        proj_peak_deceleration = np.zeros(1, dtype=np.float64)

        # Calculate initial energy
        initial_proj_ke = 0.5 * proj.mass * np.sum(proj.velocity**2)
        initial_proj_rot_ke = 0.0
        if getattr(proj, "omega", None) is not None:
            initial_proj_rot_ke = 0.5 * np.sum(proj.omega**2 / np.diagonal(proj.inertia_inv))
        initial_energy = initial_proj_ke + initial_proj_rot_ke

        if log_to_file:
            import platform

            logger.info("=" * 60)
            logger.info("VibeDynaLITE Solver Subprocess Started")
            logger.info(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
            logger.info(f"Python: {platform.python_version()}")
            logger.info(f"Active Backend: {solver_backend}")
            if solver_backend == "numba":
                logger.info(f"Numba Threads Configured: {num_threads}")
            elif solver_backend == "taichi":
                logger.info("Taichi CPU/GPU Integration active")
            logger.info(
                f"Grid Dimensions: {grid_cfg['nx']}x{grid_cfg['ny']} (plies: {grid_cfg['n_plies']})"
            )
            logger.info(f"Grid Nodes: {grid.n_nodes} | Springs: {grid.n_springs}")
            logger.info(
                f"Projectile: Shape={proj_cfg.get('shape_type', 'box')} | Mass={proj_cfg['mass']:.5f} kg | Initial Velocity={proj_cfg['velocity']} m/s"
            )
            logger.info(f"Initial System Energy: {initial_energy:.4f} J")
            logger.info("=" * 60)

        # Send configuration metadata back to GUI process
        queue.put(
            {
                "type": "init",
                "dt": dt,
                "n_nodes": grid.n_nodes,
                "n_springs": len(grid.springs),
                "n_nodes_per_layer": n_nodes_per_layer,
                "n_layers": n_layers,
                "device": backend.get_active_device(),
                "projectile_pos": proj_position.tolist(),
            }
        )

        # Fused chunk parameters
        n_chunk = 100
        save_interval = 10
        is_paused = False

        while t_sim < duration:
            # Check for control signals from GUI process
            if pipe.poll():
                msg = pipe.recv()
                if msg == "stop":
                    break
                elif msg == "pause":
                    is_paused = True
                elif msg == "resume":
                    is_paused = False

            if is_paused:
                time.sleep(0.02)
                continue

            steps_remaining = int(np.ceil((duration - t_sim) / dt))
            current_steps = min(n_chunk, steps_remaining)
            if current_steps <= 0:
                break

            # Allocate history array for orientation
            m_frames = max(1, current_steps // save_interval)
            hist_proj_quat = np.zeros((m_frames, 4), dtype=np.float64)

            # Extra solver arguments for 6-DOF and shape
            extra_kwargs = {
                "grid_damage": grid.damage,
                "proj_quat": proj.quat,
                "proj_omega": proj.omega,
                "proj_shape_type": proj.shape_type,
                "proj_radius": proj.radius,
                "proj_length": proj.length,
                "proj_edge_radius": proj.edge_radius,
                "proj_ogive_multiplier": proj.ogive_multiplier,
                "proj_span": proj.span,
                "proj_root_chord": proj.root_chord,
                "proj_tip_chord": proj.tip_chord,
                "proj_twist": proj.twist,
                "proj_thickness_ratio": proj.thickness_ratio,
                "proj_tip_radius": proj.tip_radius,
                "proj_z_com": getattr(proj, "z_com", 0.0),
                "proj_y_com": getattr(proj, "y_com", 0.0),
                "proj_c_damping": proj_cfg.get("c_damping", 0.0),
                "proj_inertia_inv": proj.inertia_inv,
                "hist_proj_quat": hist_proj_quat,
            }

            # Execute explicit integration step using Taichi or Numba backend
            solver_backend = sim_cfg.get("backend", "taichi")
            mu_s = sim_cfg.get("mu_s", sim_cfg.get("friction_coefficient", 0.0))
            prev_clamp = clamp_dissipated
            if solver_backend == "numba":
                from kevlargrid.solver.fused import fused_leapfrog_loop

                extra_kwargs["proj_peak_deceleration"] = proj_peak_deceleration

                (
                    positions,
                    velocities,
                    grid.failed,
                    proj.position,
                    proj.velocity,
                    damp_dissipated,
                    failure_dissipated,
                    clamp_dissipated,
                    t_sim,
                    hist_pos,
                    hist_failed,
                    hist_proj_pos,
                    hist_time,
                    hist_ke,
                    hist_se,
                    hist_proj_ke,
                    contact_energy,
                    friction_dissipated,
                ) = fused_leapfrog_loop(
                    positions,
                    velocities,
                    grid.springs,
                    grid.stiffnesses,
                    grid.rest_lengths,
                    grid.failed,
                    grid.masses,
                    grid.tension_only,
                    boundary_mask,
                    nodal_external_forces,
                    proj.position,
                    proj.velocity,
                    proj.mass,
                    proj.blade_width,
                    proj.edge_thickness,
                    n_layers,
                    n_nodes_per_layer,
                    t_ply if t_ply is not None else 0.002,
                    dx,
                    k_penalty,
                    rayleigh_alpha,
                    rayleigh_beta,
                    failure_strain,
                    damage_onset_strain,
                    fracture_energy_multiplier,
                    dt,
                    current_steps,
                    save_interval,
                    damp_dissipated,
                    failure_dissipated,
                    clamp_dissipated,
                    t_sim,
                    strike_direction,
                    grid.initial_spring_counts,
                    grid.node_spring_offsets,
                    grid.node_spring_ids,
                    grid.node_spring_signs,
                    use_viscous=(damping_model == "viscous"),
                    cfl_factor=sim_cfg["cfl_factor"] if auto_cfl else 0.0,
                    contact_energy_init=contact_energy,
                    mu_s=mu_s,
                    friction_dissipated_init=friction_dissipated,
                    **extra_kwargs,
                )
                hist_peak_strain = np.zeros(len(hist_time))
                n_springs = len(grid.springs)
                if n_springs > 0:
                    s0 = grid.springs[:, 0]
                    s1 = grid.springs[:, 1]
                    L0 = grid.rest_lengths
                    for f in range(len(hist_time)):
                        pos_f = hist_pos[f]
                        failed_f = hist_failed[f]
                        dx_f = pos_f[s1, 0] - pos_f[s0, 0]
                        dy_f = pos_f[s1, 1] - pos_f[s0, 1]
                        dz_f = pos_f[s1, 2] - pos_f[s0, 2]
                        lens_f = np.sqrt(dx_f**2 + dy_f**2 + dz_f**2)
                        strains_f = (lens_f - L0) / L0
                        active_strains = np.where(failed_f, 0.0, strains_f)
                        hist_peak_strain[f] = (
                            np.max(active_strains) if len(active_strains) > 0 else 0.0
                        )

                if clamp_dissipated > prev_clamp:
                    logger.warning(
                        "Velocity clamping occurred in Numba solver: dissipated energy increased by %e J",
                        clamp_dissipated - prev_clamp,
                    )
            else:
                from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop

                (
                    positions,
                    velocities,
                    grid.failed,
                    proj.position,
                    proj.velocity,
                    damp_dissipated,
                    failure_dissipated,
                    clamp_dissipated,
                    t_sim,
                    hist_pos,
                    hist_failed,
                    hist_proj_pos,
                    hist_time,
                    hist_ke,
                    hist_se,
                    hist_proj_ke,
                    hist_peak_strain,
                    contact_energy,
                    friction_dissipated,
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
                    nodal_external_forces,
                    proj.position,
                    proj.velocity,
                    proj.mass,
                    proj.blade_width,
                    proj.edge_thickness,
                    n_layers,
                    n_nodes_per_layer,
                    t_ply if t_ply is not None else 0.002,
                    dx,
                    k_penalty,
                    rayleigh_alpha,
                    rayleigh_beta,
                    failure_strain,
                    damage_onset_strain,
                    fracture_energy_multiplier,
                    dt,
                    current_steps,
                    save_interval,
                    damp_dissipated,
                    failure_dissipated,
                    clamp_dissipated,
                    t_sim,
                    strike_direction,
                    grid.initial_spring_counts,
                    grid.node_spring_offsets,
                    grid.node_spring_ids,
                    grid.node_spring_signs,
                    use_viscous=(damping_model == "viscous"),
                    cfl_factor=sim_cfg["cfl_factor"] if auto_cfl else 0.0,
                    contact_energy_init=contact_energy,
                    mu_s=mu_s,
                    friction_dissipated_init=friction_dissipated,
                    **extra_kwargs,
                )
                if clamp_dissipated > prev_clamp:
                    logger.warning(
                        "Velocity clamping occurred in Taichi solver: dissipated energy increased by %e J",
                        clamp_dissipated - prev_clamp,
                    )

            # Check for numerical instability (NaN/inf values)
            if np.isnan(positions[0, 0]) or np.any(np.isnan(positions)):
                raise ValueError(
                    "Numerical instability detected: coordinates have diverged to NaN.\n"
                    "Please reduce your CFL safety factor or increase spacing dx."
                )

            # Calculate metrics for live progress
            ke = float(hist_ke[-1]) if len(hist_ke) > 0 else 0.0
            se = float(hist_se[-1]) if len(hist_se) > 0 else 0.0
            proj_ke = float(hist_proj_ke[-1]) if len(hist_proj_ke) > 0 else 0.0
            peak_strain = float(hist_peak_strain[-1]) if len(hist_peak_strain) > 0 else 0.0
            failed_count = int(np.sum(grid.failed))

            hist_pos_np = np.asarray(hist_pos)
            hist_failed_np = np.asarray(hist_failed)

            # Send telemetry snapshot chunk back to parent GUI process
            # Convert Taichi fields to standard NumPy arrays for DPG/PyVista compatibility S7.6.1
            queue.put(
                {
                    "type": "telemetry",
                    "steps": current_steps,
                    "t_sim": float(t_sim),
                    "positions": np.asarray(positions).copy(),
                    "failed": np.asarray(grid.failed).copy(),
                    "projectile_pos": np.asarray(proj.position).copy(),
                    "projectile_vel": np.asarray(proj.velocity).copy(),
                    "projectile_quat": np.asarray(proj.quat).copy(),
                    "projectile_omega": np.asarray(proj.omega).copy(),
                    "ke": float(ke),
                    "se": float(se),
                    "damp_dissipated": float(damp_dissipated),
                    "failure_dissipated": float(failure_dissipated),
                    "clamp_dissipated": float(clamp_dissipated),
                    "friction_dissipated": float(friction_dissipated),
                    "peak_strain": float(peak_strain),
                    "proj_ke": float(proj_ke),
                    "failed_count": int(failed_count),
                    "hist_pos": hist_pos_np,
                    "hist_failed": hist_failed_np,
                    "hist_proj_pos": np.asarray(hist_proj_pos),
                    "hist_proj_quat": hist_proj_quat.copy(),
                    "hist_time": np.asarray(hist_time),
                    "hist_ke": np.asarray(hist_ke),
                    "hist_se": np.asarray(hist_se),
                    "hist_proj_ke": np.asarray(hist_proj_ke),
                    "hist_peak_strain": np.asarray(hist_peak_strain),
                }
            )

            if log_to_file:
                total_energy = (
                    ke
                    + se
                    + proj_ke
                    + damp_dissipated
                    + failure_dissipated
                    + clamp_dissipated
                    + contact_energy
                    + friction_dissipated
                )
                drift_pct = (
                    abs(total_energy - initial_energy) / initial_energy * 100.0
                    if initial_energy > 0
                    else 0.0
                )
                try:
                    import kevlargrid.solver.taichi_solver as taichi_solver

                    taichi_decel = (
                        float(taichi_solver._SOLVER_CACHE.peak_deceleration_g[None])
                        if taichi_solver._SOLVER_CACHE is not None
                        else 0.0
                    )
                except Exception:
                    taichi_decel = 0.0
                decel_g = (
                    float(proj_peak_deceleration[0]) if solver_backend == "numba" else taichi_decel
                )
                logger.info(
                    "Step %d: t=%.1f us | z_pos=%.3f mm | v_z=%.2f m/s | Decel=%.2f g | Failed=%d (%.2f%%) | Drift=%.4f%%",
                    int(t_sim / dt) if dt > 0 else 0,
                    t_sim * 1e6,
                    proj.position[2] * 1000,
                    proj.velocity[2],
                    decel_g,
                    failed_count,
                    (failed_count / grid.n_springs) * 100.0 if grid.n_springs > 0 else 0.0,
                    drift_pct,
                )
                logger.info(
                    "Energy (J): KE_grid=%.4f | SE_grid=%.4f | KE_proj=%.4f | Damp=%.4f | Fric=%.4f | Fail=%.4f | Clamp=%.4f | Contact=%.4f | Total=%.4f",
                    ke,
                    se,
                    proj_ke,
                    damp_dissipated,
                    friction_dissipated,
                    failure_dissipated,
                    clamp_dissipated,
                    contact_energy,
                    total_energy,
                )

            # Check for termination condition S7.6.1
            proximity_threshold = dx * 2.0
            positions_np = np.asarray(positions)
            update_contact_zone(proj, grid, proximity_threshold, positions=positions_np)
            reason = check_termination(
                proj,
                grid,
                positions_np,
                t_sim,
                duration,
                proj_cfg["velocity"][2],
            )
            if reason is not None:
                break

        # Calculate final reports
        float(proj.velocity[2])
        is_penetrated = reason == "penetration"
        is_arrested = reason == "arrest" or reason == "timeout" or reason is None

        initial_ke = 0.5 * proj.mass * np.sum(np.array(proj_cfg["velocity"]) ** 2)
        final_ke = 0.5 * proj.mass * np.sum(proj.velocity**2)
        energy_eff = float((initial_ke - final_ke) / initial_ke) if initial_ke > 0.0 else 0.0

        # Retrieve peak deceleration Gs
        if solver_backend == "numba":
            peak_decel = float(proj_peak_deceleration[0])
        else:
            import kevlargrid.solver.taichi_solver as taichi_solver

            peak_decel = (
                float(taichi_solver._SOLVER_CACHE.peak_deceleration_g[None])
                if taichi_solver._SOLVER_CACHE is not None
                else 0.0
            )

        # Find closest node in-plane in the base layer to identify center node per layer
        base_nodes = positions[:n_nodes_per_layer]
        dists_in_plane = (base_nodes[:, 0] - proj_cfg["position"][0]) ** 2 + (
            base_nodes[:, 1] - proj_cfg["position"][1]
        ) ** 2
        center_idx = int(np.argmin(dists_in_plane))

        max_layer_perf = -1
        for layer in range(n_layers):
            c = center_idx + layer * n_nodes_per_layer
            start_sp = grid.node_spring_offsets[c]
            end_sp = grid.node_spring_offsets[c + 1]
            sp_ids = grid.node_spring_ids[start_sp:end_sp]
            if len(sp_ids) > 0 and np.all(grid.failed[sp_ids]):
                max_layer_perf = layer

        diff_vec = positions[grid.springs[:, 1]] - positions[grid.springs[:, 0]]
        lengths = np.sqrt(np.sum(diff_vec**2, axis=1))
        strains = (lengths - grid.rest_lengths) / grid.rest_lengths
        active_strains = strains[~grid.failed]
        final_peak_strain = float(np.max(active_strains)) if len(active_strains) > 0 else 0.0

        report = {
            "arrested": bool(is_arrested),
            "peak_deceleration_g": float(peak_decel),
            "yarn_rupture_percentage": float(np.sum(grid.failed) / len(grid.failed) * 100.0)
            if len(grid.failed) > 0
            else 0.0,
            "residual_velocity_ms": float(np.linalg.norm(proj.velocity)) if is_penetrated else 0.0,
            "energy_dissipation_efficiency": float(energy_eff),
            "max_layer_perforated": int(max_layer_perf),
            # Detailed 6-DOF projectile metrics
            "projectile_shape": str(proj.shape_type),
            "projectile_volume": float(proj.volume),
            "projectile_inertia": [
                float(proj.inertia[0, 0]),
                float(proj.inertia[1, 1]),
                float(proj.inertia[2, 2]),
            ],
            "projectile_velocity_final": [
                float(proj.velocity[0]),
                float(proj.velocity[1]),
                float(proj.velocity[2]),
            ],
            "projectile_omega_final": [
                float(proj.omega[0]),
                float(proj.omega[1]),
                float(proj.omega[2]),
            ],
            "projectile_quat_final": [
                float(proj.quat[0]),
                float(proj.quat[1]),
                float(proj.quat[2]),
                float(proj.quat[3]),
            ],
            # Keep legacy keys for any other potential backward compatibility
            "penetrated": bool(is_penetrated),
            "final_velocity": float(np.linalg.norm(proj.velocity)),
            "energy_loss_pct": float(energy_eff * 100.0),
            "failed_springs": int(np.sum(grid.failed)),
            "peak_strain": final_peak_strain,
            "damp_dissipated": float(damp_dissipated),
            "failure_dissipated": float(failure_dissipated),
            "clamp_dissipated": float(clamp_dissipated),
        }

        queue.put(
            {
                "type": "completed",
                "report": report,
            }
        )
        queue.close()
        queue.join_thread()

        if log_to_file:
            logger.info("=" * 60)
            logger.info("Simulation Run Completed Successfully")
            logger.info(f"Outcome: {'ARRESTED' if is_arrested else 'PENETRATED'}")
            logger.info(f"Final Velocity: {report['final_velocity']:.2f} m/s")
            logger.info(f"Peak Deceleration: {report['peak_deceleration_g']:.2f} g")
            logger.info(
                f"Failed Springs: {report['failed_springs']} ({report['yarn_rupture_percentage']:.2f}%)"
            )
            logger.info(f"Max Layer Perforated: {report['max_layer_perforated']}")
            logger.info(
                f"Energy Dissipation Efficiency: {report['energy_dissipation_efficiency'] * 100:.2f}%"
            )
            logger.info("=" * 60)
            logger.removeHandler(file_handler)
            file_handler.close()

        import os

        os._exit(0)

    except Exception as e:
        tb = traceback.format_exc()
        queue.put(
            {
                "type": "error",
                "message": str(e),
                "traceback": tb,
            }
        )
        queue.close()
        queue.join_thread()

        if "log_to_file" in locals() and log_to_file:
            logger.error("Simulation run failed with exception:")
            logger.error(tb)
            logger.removeHandler(file_handler)
            file_handler.close()

        import os

        os._exit(1)
