# Implementation Plan: 3D Node-to-Surface Contact & Physical Mesh Locking

## Executive Summary
Our recent mesh refinement study identified an artificial strain singularity ("punch-through") at fine resolutions ($dx \le 1.25\text{ mm}$). This is caused by point-loading the discrete mass-spring network, which forces unphysical single-node deflections. 

To resolve this and introduce realistic tumbling dynamics for arbitrary threats (bullets, irregular fragments), we are discarding continuum-based strain scaling. Instead, we will:
1. **Upgrade the projectile to a 6-DOF 3D Rigid Body.**
2. **Implement Node-to-Surface Penalty Contact** using Signed Distance Fields (SDFs) to distribute the load geometrically.
3. **Lock the grid resolution** to the physical length-scale of the Kevlar yarns.

---

## Phase 1: 6-DOF Rigid Body Projectile Representation
**Target File:** `src/kevlargrid/solver/projectile.py`

Transition the projectile from a 2D bounding box or point mass into a rigid 3D body capable of translation, rotation, and representing a physical surface.

### Tasks:
*   **Expand Projectile State:** Update the Taichi data structures (`@ti.dataclass`) to track a full 6-DOF state:
    *   `pos`: 3D position of the Center of Mass (CoM).
    *   `vel`: 3D linear velocity.
    *   `quat`: 4D Quaternion for 3D orientation (to avoid gimbal lock).
    *   `omega`: 3D angular velocity.
    *   `mass`: Scalar mass.
    *   `inertia_inv`: $3 \times 3$ inverse inertia tensor (computed once at initialization based on mass and geometry).
*   **Define Shape Primitives via SDFs:** Implement parameterized 3D shapes using Signed Distance Fields (SDFs). SDFs evaluate penetration depth analytically and are highly efficient on GPUs (perfect for Taichi).
    *   *Chamfered Cylinder (FSP / Bullet base):* Parameterized by `radius`, `half_length`, and `edge_radius`. The edge radius softens the sharp edges to eliminate "cookie-cutter" numerical shearing.
    *   *Sphere/Hemisphere:* Parameterized by `radius`.

---

## Phase 2: Node-to-Surface Penalty Contact
**Target Files:** `src/kevlargrid/solver/forces.py`, `src/kevlargrid/solver/taichi_solver.py`

Replace the "Center-Node Release" point-load logic with a distributed, physics-based contact algorithm.

### Tasks:
*   **Global Node Search (`@ti.kernel`):** Every timestep, loop over all active fabric nodes. Transform each node's world position into the projectile's local coordinate system using the projectile's `pos` and inverse `quat`.
*   **Penetration Evaluation:** Pass the local node coordinates into the projectile's SDF function. 
    *   If $SDF(x_{loc}, y_{loc}, z_{loc}) < 0$, the node is penetrating the projectile volume. Penetration depth is $\delta = |SDF|$.
*   **Penalty Force Calculation:** Apply a restoring spring-damper penalty force to the penetrating node:
    *   $F_{contact} = (k_{penalty} \cdot \delta + c_{damping} \cdot \dot{\delta}) \cdot \hat{n}_{world}$
    *   *Where $\hat{n}_{world}$ is the local SDF gradient rotated back into world space, and $\dot{\delta}$ is the relative normal velocity.*
*   **Two-Way Coupling (Translation & Tumbling):** Apply the equal and opposite force to the projectile accumulators using `ti.atomic_add` to prevent race conditions during parallel execution:
    *   Linear Force: $F_{proj} \mathrel{+}= -F_{contact}$
    *   Torque: $\tau_{proj} \mathrel{+}= r_{contact} \times (-F_{contact})$ *(where $r_{contact}$ is the vector from the projectile CoM to the node).*

---

## Phase 3: 6-DOF Explicit Integration
**Target File:** `src/kevlargrid/solver/integrator.py`

Update the main integration loop to step the projectile forward in both translation and rotation.

### Tasks:
*   **Translational Integration:** Standard explicit integration for `pos` and `vel` using the accumulated $F_{proj}$ and $mass$.
*   **Rotational Integration:**
    *   Update angular velocity: $\omega_{new} = \omega_{old} + dt \cdot (inertia\_inv \cdot \tau_{proj})$ (assuming a diagonalized inertia tensor for simplicity, or full tensor if necessary).
    *   Update quaternion using the angular velocity: $\dot{q} = \frac{1}{2} [0, \omega_x, \omega_y, \omega_z] \otimes q_{old}$.
    *   Normalize the quaternion after integration to prevent numerical drift ($q_{new} = q_{new} / \|q_{new}\|$).

---

## Phase 4: Physical Grid Locking
**Target Files:** `src/kevlargrid/solver/grid.py`, `src/kevlargrid/io/config.py`

In a discrete mass-spring model, $dx$ represents physical yarns. We cannot divide a $1.25\text{ mm}$ yarn into ten $0.125\text{ mm}$ springs without corrupting the physical structural response.

### Tasks:
*   **Implement a Resolution Floor:** Define `physical_yarn_width` in the material library (`src/kevlargrid/materials/library.py`). For Kevlar 29, this is typically $1.0\text{ mm}$ to $1.25\text{ mm}$.
*   **Configuration Validation:** In `config.py`, validate the requested grid $dx$. If the user attempts to load a config with a $dx$ smaller than the physical yarn width, raise a `ValueError` stating that sub-yarn discretization violates the discrete structural assumptions of the solver.
*   **Remove Old Hacks:** Strip out any logic in `failure.py` that scales failure strain dynamically based on mesh size. Ensure $\epsilon_{fail}$ is strictly tied to the physical material property (e.g., 3.6%).

---

## Phase 5: Configuration Updates and Testing
**Target Files:** `configs/examples/*.json` (or `.toml`), `tests/unit/test_projectile.py`

### Tasks:
*   **Update Config Schemas:** Update the input loaders to accept the new 3D definitions. 
    ```toml
    [projectile]
    shape = "chamfered_cylinder"
    mass_g = 1.1
    radius_mm = 2.78
    half_length_mm = 3.0
    edge_radius_mm = 0.5
    initial_vel = [0.0, 0.0, -300.0]
    initial_yaw_deg = 15.0  # Induce initial tumbling
    ```
*   **Write Tumbling Unit Test:** Add a test verifying that an off-center node impact correctly generates a non-zero torque ($\tau_{proj}$) on the rigid body, resulting in angular acceleration.
*   **Validate V50 Convergence:** Rerun the benchmark suite at the $1.25\text{ mm}$ limit. The $V_{50}$ should now be stable, as a 5.56 mm FSP will appropriately recruit 16-20 nodes simultaneously, preventing the premature "zipper" tear.