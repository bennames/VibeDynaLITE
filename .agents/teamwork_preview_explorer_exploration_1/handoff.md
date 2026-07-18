# Exploration Report: Solver & Multi-Ply Mesh Setup

## 1. Observation

### Explicit Leapfrog Solver Loop
The explicit time integration is implemented across two optimized backends under `src/kevlargrid/solver/` and orchestrated via a background subprocess in `src/kevlargrid/solver/worker.py`.
- **CPU/Numba Backend**: The loop is defined in `src/kevlargrid/solver/fused.py` as:
  ```python
  def fused_leapfrog_loop(
      positions: np.ndarray,
      velocities: np.ndarray,
      grid_springs: np.ndarray,
      grid_stiffnesses: np.ndarray,
      grid_rest_lengths: np.ndarray,
      grid_failed: np.ndarray,
      grid_masses: np.ndarray,
      grid_tension_only: np.ndarray,
      boundary_mask: np.ndarray,
      nodal_external_forces: np.ndarray,
      proj_position: np.ndarray,
      proj_velocity: np.ndarray,
      proj_mass: float,
      proj_blade_width: float,
      proj_edge_thickness: float,
      n_plies: int,
      n_nodes_per_layer: int,
      t_ply: float,
      dx: float,
      k_penalty: float,
      rayleigh_alpha: float,
      rayleigh_beta: float,
      failure_strain: float,
      damage_onset_strain: float,
      fracture_energy_multiplier: float,
      dt: float,
      n_steps: int,
      save_interval: int,
      damp_dissipated_init: float,
      failure_dissipated_init: float,
      clamp_dissipated_init: float,
      t_sim_init: float,
      strike_direction: float,
      node_initial_springs: np.ndarray,
      node_spring_offsets: np.ndarray,
      node_spring_ids: np.ndarray,
      node_spring_signs: np.ndarray,
      use_viscous: bool = False,
      cfl_factor: float = -1.0,
      grid_damage: np.ndarray | None = None,
      ...
  )
  ```
- **GPU/Taichi Backend**: The loop is defined in `src/kevlargrid/solver/taichi_solver.py` as:
  ```python
  def taichi_leapfrog_loop(
      positions: np.ndarray,
      velocities: np.ndarray,
      grid_springs: np.ndarray,
      grid_stiffnesses: np.ndarray,
      grid_rest_lengths: np.ndarray,
      grid_failed: np.ndarray,
      grid_masses: np.ndarray,
      grid_tension_only: np.ndarray,
      boundary_mask: np.ndarray,
      nodal_external_forces: np.ndarray,
      proj_position: np.ndarray,
      proj_velocity: np.ndarray,
      proj_mass: float,
      proj_blade_width: float,
      proj_edge_thickness: float,
      n_plies: int,
      n_nodes_per_layer: int,
      t_ply: float,
      dx: float,
      k_penalty: float,
      rayleigh_alpha: float,
      rayleigh_beta: float,
      failure_strain: float,
      damage_onset_strain: float,
      fracture_energy_multiplier: float,
      dt: float,
      n_steps: int,
      save_interval: int,
      damp_dissipated_init: float,
      failure_dissipated_init: float,
      clamp_dissipated_init: float,
      t_sim_init: float,
      strike_direction: float,
      node_initial_springs: np.ndarray,
      node_spring_offsets: np.ndarray | None = None,
      node_spring_ids: np.ndarray | None = None,
      node_spring_signs: np.ndarray | None = None,
      use_viscous: bool = False,
      cfl_factor: float = -1.0,
      grid_damage: np.ndarray | None = None,
      ...
  )
  ```

### Multi-Ply Mesh Generation
Mesh generation is handled in `src/kevlargrid/solver/grid.py` under the function:
```python
def generate_rectangular_grid(
    nx: int,
    ny: int,
    dx: float,
    material: dict,
    n_plies: int = 1,
    t_ply: float | None = None,
) -> Grid
```
When `t_ply is not None` and `n_plies > 1`, Mode B (Checkout Mode) is activated:
- Each ply has its node coordinates stacked along the Z-axis (line 243):
  ```python
  ply_nodes[:, 2] = ply * t_ply
  ```
- Springs are offset by `ply * n_nodes_per_layer` (line 248):
  ```python
  ply_springs = base_springs + offset
  ```
- Interply penalty contact forces are calculated in `src/kevlargrid/solver/forces.py` as (lines 299-302):
  ```python
  gap = np.abs(z_n - z_n1)
  penetration = t_ply - gap
  penetrating = penetration > 0.0
  ```

### Clamped Boundary Conditions
Boundary mask setup in `src/kevlargrid/solver/worker.py` (lines 60-66):
```python
        if grid_cfg["boundary_type"] == "fixed":
            for ply in range(n_layers):
                offset = ply * n_nodes_per_layer
                for i in range(nx):
                    for j in range(ny):
                        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                            boundary_mask[offset + i * ny + j] = 1
```

### Material Property Mapping
`src/kevlargrid/solver/grid.py` map material properties as (lines 215-228):
```python
    tensile_modulus_gpa = material.get("tensile_modulus_gpa", 71.0)
    fiber_density_gcc = material.get("fiber_density_gcc", 1.44)
    shear_ratio = material.get("shear_ratio", 0.0004)

    # Thickness t = areal_density / density
    t = areal_density / (fiber_density_gcc * 1000.0)
    e_mod = tensile_modulus_gpa * 1e9

    k_scale = 1.0 if is_mode_b else float(n_plies)
    k_ortho = k_scale * e_mod * t
    k_shear = k_ortho * shear_ratio

    base_stiffnesses = np.where(base_tension_only, k_ortho, k_shear)
    base_rest_lengths = np.where(base_tension_only, dx, np.sqrt(2.0) * dx)
```

---

## 2. Logic Chain

1. **Integration Call & Setup**: `fused_leapfrog_loop` or `taichi_leapfrog_loop` takes pre-allocated state arrays and physics/geometry parameters. Physical constants mapped to springs and masses are fed to the integration loop without requiring runtime lookups.
2. **Layer Separation & Spacing**: In a shell representation, plies have zero coordinate thickness in the Z-dimension. The distance between adjacent layer nodes is directly initialized to `t_ply`. Under contact, penetration occurs when `gap < t_ply`. To separate the layers by a physical gap of 0.1 mm, we set `t_ply = 0.0001` (0.1 mm).
3. **Layer Isolation & ID Offsets**: Mode B stacks and offsets coordinate nodes and spring connections by `ply * n_nodes_per_layer`. Thus, nodes across different plies have distinct IDs, and springs only connect nodes within the same ply.
4. **Boundary Clamping**: Perimeter nodes of each ply are flagged with `boundary_mask = 1` if `"boundary_type": "fixed"` is selected. The solver forces velocity and accelerations to zero for all nodes having a mask of 1, effectively clamping all 4 edges.
5. **Grid Resolution**: The projectile diameter is $d = 5.46 \text{ mm} = 0.00546 \text{ m}$. To resolve this with at least 3 to 4 nodes (2 to 3 elements):
   - For exactly 4 nodes spanning the diameter (3 elements):
     $$dx \le \frac{5.46 \text{ mm}}{3} = 1.82 \text{ mm} = 0.00182 \text{ m}$$
   - For a patch of $W = 250 \text{ mm} = 0.25 \text{ m}$, this results in $nx = ny = \text{round}(0.25 / 0.00182) + 1 = 138$ nodes per direction.
6. **Property Processing**: The inputs `tensile_modulus_gpa`, `fiber_density_gcc`, `areal_density_kgm2`, and `shear_ratio` are parsed in `config.py` and processed in `grid.py` to produce spring stiffnesses (`grid.stiffnesses` mapped via $k_{ortho}$ and $k_{shear}$) and nodal masses (`grid.masses` scaled by tributary cell area).

---

## 3. Caveats

- **Shell Thickness Assumption**: This model represents layers as planar grids of nodes with zero Z-thickness. The inter-ply spacing parameter `t_ply` acts both as the geometric coordinate offset and as the threshold for interply penetration contact.
- **Normal Impact Constraint**: Inter-ply contact is evaluated strictly along the Z-axis, which is accurate for normal ($0^\circ$) impact center strikes, but may introduce inaccuracies for high-obliquity angles.
- **Rigid Projectile**: The projectile is treated as non-deformable.

---

## 4. Conclusion

To model the dry Kevlar 29 Style 713 ballistic limit benchmark with 13 plies, the simulation config dictionary should be set up as:
1. **Material Properties**:
   ```python
   material = {
       "tensile_modulus_gpa": 70.5,
       "tensile_strength_gpa": 2.92,
       "fiber_density_gcc": 1.44,
       "areal_density_kgm2": 0.475,
       "shear_ratio": 0.0004,
       "failure_strain": 0.036, # or up to 0.040
   }
   ```
2. **Grid Geometry & Boundaries**:
   - `n_plies = 13`
   - `t_ply = 0.0001` (0.1 mm spacing)
   - `dx = 0.00182` (1.82 mm element size, providing exactly 4 nodes spanning the 5.46 mm projectile diameter)
   - `nx = 138`, `ny = 138` (representing a 250 mm x 250 mm patch)
   - `boundary_type = "fixed"` (automatically clamps all 4 edges on every ply)
3. **Solver Backend**:
   - Set `"backend": "numba"` or `"taichi"` in `simulation` config to execute.

---

## 5. Verification Method

1. **Unit Test Verification**: Run the standard test suite to ensure no code regressions:
   ```bash
   make test
   ```
2. **Geometry/Stacking Verification**: Verify by compiling a test script in the workspace that imports `generate_rectangular_grid`:
   ```python
   from kevlargrid.solver.grid import generate_rectangular_grid
   grid = generate_rectangular_grid(
       nx=138, ny=138, dx=0.00182, 
       material={"tensile_modulus_gpa": 70.5, "fiber_density_gcc": 1.44, "areal_density_kgm2": 0.475, "shear_ratio": 0.0004}, 
       n_plies=13, t_ply=0.0001
   )
   # Asserts
   assert grid.n_nodes == 13 * 138 * 138 # 247,572 nodes
   assert grid.nodes[0, 2] == 0.0
   assert grid.nodes[138*138, 2] == 0.0001
   assert grid.nodes[-1, 2] == 12 * 0.0001
   ```
3. **Boundary Condition Verification**: Inspect that the boundary mask has 1 for boundaries:
   ```python
   # Nodes per ply = 138 * 138 = 19044
   # Corners and edges must be set to 1
   ```
