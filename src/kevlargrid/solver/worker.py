"""Background process worker executing KevlarGrid explicit dynamics simulation loops.

Runs the physics integration in a separate OS process to bypass the Python GIL and
prevent Objective-C class name collisions between DearPyGui and Taichi on macOS.
"""

import time
import traceback

import numpy as np

from kevlargrid.solver import backend
from kevlargrid.solver.energy import compute_kinetic_energy, compute_strain_energy
from kevlargrid.solver.fused import fused_leapfrog_loop
from kevlargrid.solver.grid import generate_rectangular_grid
from kevlargrid.solver.projectile import Projectile, update_contact_zone, check_termination
from kevlargrid.solver.timestep import compute_cfl_timestep


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
        # 1. Setup simulation components
        mat = config["material"]
        grid_cfg = config["grid"]
        proj_cfg = config["projectile"]
        sim_cfg = config["simulation"]

        nx, ny, dx = grid_cfg["nx"], grid_cfg["ny"], grid_cfg["dx"]
        n_plies = grid_cfg["n_plies"]
        t_ply = grid_cfg["t_ply"]

        # Build grid
        grid = generate_rectangular_grid(
            nx=nx, ny=ny, dx=dx, material=mat, n_plies=n_plies, t_ply=t_ply
        )

        # Build boundary mask
        boundary_mask = np.zeros(grid.n_nodes, dtype=bool)
        n_nodes_per_layer = nx * ny
        n_layers = n_plies if (t_ply is not None and n_plies > 1) else 1
        if grid_cfg["boundary_type"] == "fixed":
            for ply in range(n_layers):
                offset = ply * n_nodes_per_layer
                for i in range(nx):
                    for j in range(ny):
                        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                            boundary_mask[offset + i * ny + j] = True

        # Nodal external forces setup
        if "nodal_external_forces" in config:
            nodal_external_forces = np.array(config["nodal_external_forces"], dtype=np.float64)
        else:
            nodal_external_forces = np.zeros((grid.n_nodes, 3), dtype=np.float64)

        # Calculate timestep
        k_penalty = 10.0 * np.mean(grid.stiffnesses)
        k_max_effective = max(np.max(grid.stiffnesses), k_penalty)
        dt = compute_cfl_timestep(
            np.array([k_max_effective]), grid.masses, dx, sim_cfg["cfl_factor"]
        )

        # Projectile setup
        proj = Projectile(
            mass=proj_cfg["mass"],
            velocity=proj_cfg["velocity"],
            position=proj_cfg["position"],
            blade_width=proj_cfg["blade_width"],
            edge_thickness=proj_cfg["edge_thickness"],
        )

        # Determine strike direction based on initial velocity Z-component
        strike_direction = -1.0 if proj_cfg["velocity"][2] < 0.0 else 1.0

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

            # Execute explicit integration step
            if backend.get_backend_name() == "taichi":
                # Defer taichi solver import entirely to the subprocess
                from kevlargrid.solver.taichi_solver import taichi_leapfrog_loop as solver_loop
            else:
                solver_loop = fused_leapfrog_loop

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
            ) = solver_loop(
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
                cfl_factor=sim_cfg["cfl_factor"],
            )

            # Check for numerical instability (NaN/inf values)
            if np.isnan(positions[0, 0]) or np.any(np.isnan(positions)):
                raise ValueError(
                    "Numerical instability detected: coordinates have diverged to NaN.\n"
                    "Please reduce your CFL safety factor or increase spacing dx."
                )

            # Calculate metrics for live progress
            ke = compute_kinetic_energy(velocities, grid.masses)
            diff_vec = positions[grid.springs[:, 1]] - positions[grid.springs[:, 0]]
            lengths = np.sqrt(np.sum(diff_vec**2, axis=1))
            strains = (lengths - grid.rest_lengths) / grid.rest_lengths
            
            # Compute progressive damage fraction for active-only/degraded strain energy calculation S7.14
            denom = failure_strain - damage_onset_strain
            denom_safe = denom if denom != 0.0 else 1.0
            damage = np.minimum(np.maximum((strains - damage_onset_strain) / denom_safe, 0.0), 1.0)
            se = compute_strain_energy(strains, grid.stiffnesses, grid.rest_lengths, grid.failed, damage)
            
            proj_ke = 0.5 * proj.mass * np.sum(proj.velocity**2)
            failed_count = int(np.sum(grid.failed))
            
            # Compute peak strain only on active (non-failed) springs S7.14
            active_strains = strains[~grid.failed]
            peak_strain = float(np.max(active_strains)) if len(active_strains) > 0 else 0.0

            # Calculate frame-by-frame history of active-only peak strain S7.14
            hist_pos_np = np.asarray(hist_pos)
            hist_failed_np = np.asarray(hist_failed)
            hist_peak_strain_list = []
            for idx in range(len(hist_time)):
                f_pos = hist_pos_np[idx]
                f_failed = hist_failed_np[idx]
                f_diff = f_pos[grid.springs[:, 1]] - f_pos[grid.springs[:, 0]]
                f_lens = np.sqrt(np.sum(f_diff**2, axis=1))
                f_strains = (f_lens - grid.rest_lengths) / grid.rest_lengths
                f_active = f_strains[~f_failed]
                f_peak = float(np.max(f_active)) if len(f_active) > 0 else 0.0
                hist_peak_strain_list.append(f_peak)
            hist_peak_strain = np.array(hist_peak_strain_list, dtype=np.float32)

            # Send telemetry snapshot chunk back to parent GUI process
            # Convert JAX arrays to standard NumPy arrays for DPG/PyVista compatibility S7.6.1
            queue.put(
                {
                    "type": "telemetry",
                    "steps": current_steps,
                    "t_sim": float(t_sim),
                    "positions": np.asarray(positions).copy(),
                    "failed": np.asarray(grid.failed).copy(),
                    "projectile_pos": np.asarray(proj.position).copy(),
                    "ke": float(ke),
                    "se": float(se),
                    "damp_dissipated": float(damp_dissipated),
                    "failure_dissipated": float(failure_dissipated),
                    "clamp_dissipated": float(clamp_dissipated),
                    "peak_strain": float(peak_strain),
                    "proj_ke": float(proj_ke),
                    "failed_count": int(failed_count),
                    "hist_pos": hist_pos_np,
                    "hist_failed": hist_failed_np,
                    "hist_proj_pos": np.asarray(hist_proj_pos),
                    "hist_time": np.asarray(hist_time),
                    "hist_ke": np.asarray(hist_ke),
                    "hist_se": np.asarray(hist_se),
                    "hist_proj_ke": np.asarray(hist_proj_ke),
                    "hist_peak_strain": hist_peak_strain,
                }
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
        # Compute final velocity and kinetic energy
        final_velocity_z = float(proj.velocity[2])
        e_loss_pct = 100.0 * (
            1.0 - (proj_ke / (0.5 * proj.mass * np.sum(np.array(proj_cfg["velocity"]) ** 2)))
        )

        active_strains = strains[~grid.failed]
        final_peak_strain = float(np.max(active_strains)) if len(active_strains) > 0 else 0.0

        report = {
            "penetrated": bool(final_velocity_z < -0.1 and proj.position[2] < 0.0),
            "final_velocity": float(np.linalg.norm(proj.velocity)),
            "energy_loss_pct": float(e_loss_pct),
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

    except Exception as e:
        tb = traceback.format_exc()
        queue.put(
            {
                "type": "error",
                "message": str(e),
                "traceback": tb,
            }
        )
