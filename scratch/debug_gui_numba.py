import multiprocessing

from kevlargrid.solver.worker import run_solver_process

if __name__ == "__main__":
    config = {
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
            "nx": 100,
            "ny": 100,
            "dx": 0.0050,
            "n_plies": 20,
            "t_ply": 0.002,
            "boundary_type": "non-reflecting",
        },
        "projectile": {
            "mass": 1.700,
            "velocity": [0.0, 0.0, 100.0],
            "position": [0.0, 0.0, -0.01],
            "shape_type": "sphere",
            "radius": 0.015,
        },
        "simulation": {
            "duration": 0.00200,
            "cfl_factor": 0.800,
            "damping_model": "rayleigh",
            "damping_coefficient": 0.05,
            "rayleigh_alpha": 0.0,
            "rayleigh_beta": 1e-9,
            "auto_cfl": True,
            "backend": "numba",
            "snapshot_interval": 100,
        }
    }

    # We will simulate what the runner loop does, calling the worker in a separate process
    # to catch any exception
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    parent_conn, child_conn = ctx.Pipe()

    print("Starting solver process...")
    p = ctx.Process(target=run_solver_process, args=(config, queue, child_conn), daemon=True)
    p.start()

    while p.is_alive() or not queue.empty():
        try:
            msg = queue.get(timeout=1.0)
            if msg["type"] == "error":
                print("SOLVER CRASHED:")
                print(msg["message"])
                print(msg.get("traceback", ""))
                break
            elif msg["type"] == "completed":
                print("SOLVER COMPLETED SUCCESSFULLY!")
                break
            else:
                print(f"Received message type: {msg['type']}")
        except Exception:
            continue
    p.join()
