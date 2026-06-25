"""Projectile model module.

Defines the rigid-body projectile representation and routines for evolving
contact detection and inverse-distance-weighted contact force distribution
onto the spring–mass grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from kevlargrid.solver.grid import Grid


def q_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Perform quaternion multiplication q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def q_conjugate(q: np.ndarray) -> np.ndarray:
    """Compute the conjugate of a quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector v by the orientation quaternion q."""
    q_v = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    q_rot = q_mul(q_mul(q, q_v), q_conjugate(q))
    return q_rot[1:]


class Projectile:
    """Rigid-body 6-DOF projectile interacting with a Kevlar grid.

    Attributes
    ----------
    mass : float
        Projectile mass (kg).
    vel : np.ndarray
        Current 3D velocity vector (m/s), shape ``(3,)``.
    pos : np.ndarray
        Current 3D center-of-mass position (m), shape ``(3,)``.
    quat : np.ndarray
        Orientation quaternion, shape ``(4,)``.
    omega : np.ndarray
        Angular velocity vector (rad/s), shape ``(3,)``.
    shape_type : str
        Geometric shape profile ('sphere', 'cylinder', 'bullet', 'propeller', 'box').
    volume : float
        Total volume of the rigid body (m^3).
    inertia : np.ndarray
        3x3 principal moments of inertia diagonal tensor (kg*m^2).
    inertia_inv : np.ndarray
        3x3 inverse principal moments of inertia diagonal tensor (1/(kg*m^2)).
    """

    mass: float
    vel: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    shape_type: str
    volume: float
    inertia: np.ndarray
    inertia_inv: np.ndarray

    # Shape parameters
    radius: float
    length: float
    edge_radius: float
    ogive_multiplier: float
    span: float
    root_chord: float
    tip_chord: float
    twist: float
    thickness_ratio: float
    tip_radius: float
    blade_width: float
    edge_thickness: float
    contact_nodes: np.ndarray

    def __init__(
        self,
        mass: float,
        velocity: np.ndarray | list[float],
        position: np.ndarray | list[float],
        shape_type: str = "box",
        # Legacy box parameters S5.6
        blade_width: float = 0.02,
        edge_thickness: float = 0.005,
        # Shape specific parameters
        radius: float = 0.005,
        length: float = 0.01,
        edge_radius: float = 0.0,
        ogive_multiplier: float = 2.0,
        span: float = 0.05,
        root_chord: float = 0.01,
        tip_chord: float = 0.005,
        twist: float = 15.0,
        thickness_ratio: float = 12.0,
        tip_radius: float = 0.002,
        omega: np.ndarray | list[float] | None = None,
        quat: np.ndarray | list[float] | None = None,
    ) -> None:
        self.mass = float(mass)
        self.vel = np.array(velocity, dtype=np.float64)
        self.pos = np.array(position, dtype=np.float64)
        self.quat = np.array(quat if quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.omega = np.array(omega if omega is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self.shape_type = str(shape_type).lower()

        # Save shape parameters
        self.radius = float(radius)
        self.length = float(length)
        self.edge_radius = float(edge_radius)
        self.ogive_multiplier = float(ogive_multiplier)
        self.span = float(span)
        self.root_chord = float(root_chord)
        self.tip_chord = float(tip_chord)
        self.twist = float(twist)
        self.thickness_ratio = float(thickness_ratio)
        self.tip_radius = float(tip_radius)
        self.blade_width = float(blade_width)
        self.edge_thickness = float(edge_thickness)
        
        self.contact_nodes = np.array([], dtype=np.int32)

        # Compute volume and inertia tensor based on shape
        self._initialize_inertia()

    @property
    def position(self) -> np.ndarray:
        """Backward compatibility property for COM position."""
        return self.pos

    @position.setter
    def position(self, val: np.ndarray) -> None:
        self.pos = np.array(val, dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        """Backward compatibility property for linear velocity."""
        return self.vel

    @velocity.setter
    def velocity(self, val: np.ndarray) -> None:
        self.vel = np.array(val, dtype=np.float64)

    def _initialize_inertia(self) -> None:
        """Calculate volume and exact moments of inertia assuming uniform density."""
        m = self.mass
        I = np.zeros(3)  # Principal moments: [Ixx, Iyy, Izz]

        if self.shape_type == "sphere":
            R = self.radius
            self.volume = (4.0 / 3.0) * np.pi * (R ** 3)
            I[0] = I[1] = I[2] = 0.4 * m * (R ** 2)

        elif self.shape_type == "cylinder":
            R = self.radius
            L = self.length
            self.volume = np.pi * (R ** 2) * L
            I[2] = 0.5 * m * (R ** 2)
            I[0] = I[1] = (1.0 / 12.0) * m * (3.0 * (R ** 2) + L ** 2)

        elif self.shape_type == "bullet":
            R0 = self.radius
            R_og = R0 * self.ogive_multiplier
            L_nose = np.sqrt(2.0 * R_og * R0 - R0 ** 2)
            L_body = max(0.0, self.length - L_nose)

            # Numerical integration along bullet Z-axis
            N = 500
            zs = np.linspace(-L_body, L_nose, N)
            dz = (L_body + L_nose) / N
            dV_sum = 0.0
            z_dV_sum = 0.0

            rs = np.zeros(N)
            for idx, z in enumerate(zs):
                if z < 0:
                    r = R0
                else:
                    r = R0 - R_og + np.sqrt(R_og**2 - z**2)
                rs[idx] = r
                dV = np.pi * (r ** 2) * dz
                dV_sum += dV
                z_dV_sum += z * dV

            self.volume = dV_sum
            z_com = z_dV_sum / dV_sum if dV_sum > 0 else 0.0
            self.z_com = z_com
            rho = m / dV_sum if dV_sum > 0 else 0.0

            # Sum moments of inertia about the computed CoM
            I_xx_sum = 0.0
            I_zz_sum = 0.0
            for idx, z in enumerate(zs):
                r = rs[idx]
                dV = np.pi * (r ** 2) * dz
                dm = rho * dV
                dI_zz = 0.5 * dm * (r ** 2)
                dI_xx = (1.0 / 12.0) * dm * (3.0 * (r ** 2) + dz ** 2) + dm * ((z - z_com) ** 2)
                I_xx_sum += dI_xx
                I_zz_sum += dI_zz

            I[0] = I_xx_sum
            I[1] = I_xx_sum
            I[2] = I_zz_sum

        elif self.shape_type == "propeller":
            S = self.span
            c_r = self.root_chord
            c_t = self.tip_chord
            theta_t = np.radians(self.twist)
            tau = self.thickness_ratio / 100.0

            N = 200
            ys = np.linspace(0.0, S, N)
            dy = S / N
            dV_sum = 0.0
            y_dV_sum = 0.0

            areas = np.zeros(N)
            chords = np.zeros(N)
            thicknesses = np.zeros(N)
            for idx, y in enumerate(ys):
                c = c_r + (y / S) * (c_t - c_r)
                t = c * tau
                area = 0.60 * (c ** 2) * tau
                areas[idx] = area
                chords[idx] = c
                thicknesses[idx] = t
                dV = area * dy
                dV_sum += dV
                y_dV_sum += y * dV

            self.volume = dV_sum
            y_com = y_dV_sum / dV_sum if dV_sum > 0 else 0.0
            self.y_com = y_com
            rho = m / dV_sum if dV_sum > 0 else 0.0

            I_xx_sum = 0.0
            I_yy_sum = 0.0
            I_zz_sum = 0.0
            for idx, y in enumerate(ys):
                c = chords[idx]
                t = thicknesses[idx]
                theta = theta_t * (y / S)
                dV = areas[idx] * dy
                dm = rho * dV

                I_y_sec = (1.0 / 12.0) * dm * (c ** 2)
                I_x_sec = (1.0 / 12.0) * dm * (t ** 2)

                dI_xx = (I_x_sec * (np.cos(theta)**2) + I_y_sec * (np.sin(theta)**2)) + dm * ((y - y_com) ** 2)
                dI_yy = I_y_sec + I_x_sec
                dI_zz = (I_x_sec * (np.sin(theta)**2) + I_y_sec * (np.cos(theta)**2)) + dm * ((y - y_com) ** 2)

                I_xx_sum += dI_xx
                I_yy_sum += dI_yy
                I_zz_sum += dI_zz

            I[0] = I_xx_sum
            I[1] = I_yy_sum
            I[2] = I_zz_sum

        else:
            w_h = self.blade_width / 2.0
            t_h = self.edge_thickness / 2.0
            h_h = 0.005
            self.volume = 8.0 * w_h * t_h * h_h
            I[0] = (1.0 / 12.0) * m * ((2.0 * t_h)**2 + (2.0 * h_h)**2)
            I[1] = (1.0 / 12.0) * m * ((2.0 * w_h)**2 + (2.0 * h_h)**2)
            I[2] = (1.0 / 12.0) * m * ((2.0 * w_h)**2 + (2.0 * t_h)**2)

        self.inertia = np.diag(I)
        I_inv = np.zeros(3)
        for i in range(3):
            I_inv[i] = 1.0 / I[i] if I[i] > 0.0 else 0.0
        self.inertia_inv = np.diag(I_inv)

    def sdf(self, p: np.ndarray) -> float:
        """Evaluate the signed distance field at query point p (local frame)."""
        shape = self.shape_type
        if shape == "sphere":
            return np.linalg.norm(p) - self.radius
        elif shape == "cylinder":
            R = self.radius
            L = self.length
            R_e = self.edge_radius
            d_cyl = np.sqrt(p[0]**2 + p[1]**2) - (R - R_e)
            d_len = np.abs(p[2]) - (L/2.0 - R_e)
            ext_d = np.sqrt(max(0.0, d_cyl)**2 + max(0.0, d_len)**2)
            int_d = min(0.0, max(d_cyl, d_len))
            return ext_d + int_d - R_e
        elif shape == "bullet":
            R0 = self.radius
            R_og = R0 * self.ogive_multiplier
            L_nose = np.sqrt(2.0 * R_og * R0 - R0 ** 2)
            L_body = max(0.0, self.length - L_nose)
            z_com = getattr(self, "z_com", 0.0)
            
            x, y, z = p[0], p[1], p[2]
            z_geom = z + z_com
            r = np.sqrt(x**2 + y**2)
            
            if z_geom < 0.0:
                d_cyl = r - R0
                d_cap = -z_geom - L_body
                return max(d_cyl, d_cap)
            else:
                if z_geom > L_nose:
                    return np.sqrt(r**2 + (z_geom - L_nose)**2)
                else:
                    r_c = R0 - R_og
                    dist_to_center = np.sqrt((r - r_c)**2 + z_geom**2)
                    return dist_to_center - R_og
        elif shape == "propeller":
            S = self.span
            c_r = self.root_chord
            c_t = self.tip_chord
            twist_deg = self.twist
            thickness_ratio = self.thickness_ratio
            R_tip = self.tip_radius
            y_com = getattr(self, "y_com", 0.0)
            
            x, y, z = p[0], p[1], p[2]
            y_geom = y + y_com
            
            if y_geom > S - R_tip:
                return np.sqrt(x**2 + (y_geom - (S - R_tip))**2 + z**2) - R_tip
            elif y_geom < 0.0:
                # Flat root cap
                u = (x + c_r/2.0) / c_r
                u_clamped = min(max(u, 0.0), 1.0)
                t = 5.0 * (thickness_ratio / 100.0) * (
                    0.2969 * np.sqrt(u_clamped)
                    - 0.1260 * u_clamped
                    - 0.3516 * (u_clamped**2)
                    + 0.2843 * (u_clamped**3)
                    - 0.1015 * (u_clamped**4)
                ) * c_r
                half_t = max(t / 2.0, R_tip)
                d_slice = np.abs(z) - half_t
                return max(-y_geom, d_slice)
            else:
                c = c_r + (y_geom / S) * (c_t - c_r)
                theta = np.radians(twist_deg) * (y_geom / S)
                xr = x * np.cos(theta) + z * np.sin(theta)
                zr = -x * np.sin(theta) + z * np.cos(theta)
                
                u = (xr + c/2.0) / c
                u_clamped = min(max(u, 0.0), 1.0)
                t = 5.0 * (thickness_ratio / 100.0) * (
                    0.2969 * np.sqrt(u_clamped)
                    - 0.1260 * u_clamped
                    - 0.3516 * (u_clamped**2)
                    + 0.2843 * (u_clamped**3)
                    - 0.1015 * (u_clamped**4)
                ) * c
                half_t = max(t / 2.0, R_tip)
                
                if xr < -c/2.0 + R_tip:
                    dist_le = np.sqrt((xr - (-c/2.0 + R_tip))**2 + zr**2)
                    return dist_le - R_tip
                elif xr > c/2.0 - R_tip:
                    dist_te = np.sqrt((xr - (c/2.0 - R_tip))**2 + zr**2)
                    return dist_te - R_tip
                else:
                    return np.abs(zr) - half_t
        else:
            w_h = self.blade_width / 2.0
            t_h = self.edge_thickness / 2.0
            x_proj = np.clip(p[0], -w_h, w_h)
            y_proj = np.clip(p[1], -t_h, t_h)
            z_proj = 0.0
            return np.sqrt((p[0] - x_proj) ** 2 + (p[1] - y_proj) ** 2 + (p[2] - z_proj) ** 2)

    def sdf_normal(self, p: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """Compute the unit normal vector (SDF gradient) at query point p using central differences."""
        grad = np.zeros(3)
        for i in range(3):
            p_plus = p.copy()
            p_plus[i] += h
            p_minus = p.copy()
            p_minus[i] -= h
            grad[i] = (self.sdf(p_plus) - self.sdf(p_minus)) / (2.0 * h)
        norm = np.linalg.norm(grad)
        return grad / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])


def update_contact_zone(
    projectile: Projectile,
    grid: Grid,
    proximity_threshold: float,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Determine which grid nodes fall within the projectile's contact zone.

    Uses a proximity-based detection that evolves as the projectile
    penetrates the fabric.
    """
    pos = positions if positions is not None else grid.nodes

    p_rel = pos - projectile.pos
    q_inv = q_conjugate(projectile.quat)
    p_loc = np.zeros_like(p_rel)
    for i in range(len(p_rel)):
        p_loc[i] = q_rotate_vector(q_inv, p_rel[i])

    contact_mask = np.zeros(len(p_loc), dtype=bool)
    
    if projectile.shape_type == "sphere":
        dists = np.sqrt(np.sum(p_loc**2, axis=1))
        contact_mask = dists <= (projectile.radius + proximity_threshold)

    elif projectile.shape_type == "cylinder":
        R = projectile.radius
        L = projectile.length
        d_cyl = np.sqrt(p_loc[:, 0]**2 + p_loc[:, 1]**2) - R
        d_len = np.abs(p_loc[:, 2]) - L/2.0
        dists = np.maximum(d_cyl, d_len)
        contact_mask = dists <= proximity_threshold

    elif projectile.shape_type == "bullet":
        R0 = projectile.radius
        R_og = R0 * projectile.ogive_multiplier
        L_nose = np.sqrt(2.0 * R_og * R0 - R0 ** 2)
        L_body = max(0.0, projectile.length - L_nose)

        for i in range(len(p_loc)):
            z = p_loc[i, 2]
            r = np.sqrt(p_loc[i, 0]**2 + p_loc[i, 1]**2)
            if z < 0:
                dist = r - R0
            else:
                dist = r - (R0 - R_og + np.sqrt(max(0.0, R_og**2 - z**2)))
            contact_mask[i] = dist <= proximity_threshold

    elif projectile.shape_type == "propeller":
        S = projectile.span
        c_r = projectile.root_chord
        c_t = projectile.tip_chord
        theta_t = np.radians(projectile.twist)
        tau = projectile.thickness_ratio / 100.0
        R_tip = projectile.tip_radius

        for i in range(len(p_loc)):
            x, y, z = p_loc[i]
            if y > S - R_tip:
                dist = np.sqrt(x**2 + (y - (S - R_tip))**2 + z**2) - R_tip
            else:
                c = c_r + (y / S) * (c_t - c_r)
                theta = theta_t * (y / S)
                xr = x * np.cos(theta) + z * np.sin(theta)
                zr = -x * np.sin(theta) + z * np.cos(theta)
                u = xr / c
                t = 5.0 * tau * (0.2969*np.sqrt(max(0.0, u)) - 0.1260*u - 0.3516*(u**2) + 0.2843*(u**3) - 0.1015*(u**4)) * c
                dist = np.abs(zr) - t
            contact_mask[i] = dist <= proximity_threshold

    else:
        w_h = projectile.blade_width / 2.0
        t_h = projectile.edge_thickness / 2.0
        x_proj = np.clip(p_loc[:, 0], -w_h, w_h)
        y_proj = np.clip(p_loc[:, 1], -t_h, t_h)
        z_proj = 0.0
        dists = np.sqrt((p_loc[:, 0] - x_proj) ** 2 + (p_loc[:, 1] - y_proj) ** 2 + (p_loc[:, 2] - z_proj) ** 2)
        contact_mask = dists <= proximity_threshold

    contact_nodes = np.flatnonzero(contact_mask).astype(np.int32)
    projectile.contact_nodes = contact_nodes
    return contact_nodes


def distribute_contact_forces(
    projectile: Projectile,
    grid: Grid,
    positions: np.ndarray | None = None,
    k_contact: float | None = None,
) -> np.ndarray:
    """Distribute projectile contact force onto grid nodes.

    Uses inverse-distance weighting so that nodes closer to the projectile
    centre receive proportionally larger forces.
    """
    pos = positions if positions is not None else grid.nodes
    forces = np.zeros_like(pos)

    contact_nodes = projectile.contact_nodes
    if len(contact_nodes) == 0:
        return forces

    direction = np.sign(projectile.vel[2]) if projectile.vel[2] != 0.0 else 1.0
    c_pos = pos[contact_nodes]

    if projectile.shape_type == "box":
        x_p = projectile.pos[0]
        y_p = projectile.pos[1]
        z_p = projectile.pos[2]
        w_h = projectile.blade_width / 2.0
        t_h = projectile.edge_thickness / 2.0
        x_proj = np.clip(c_pos[:, 0], x_p - w_h, x_p + w_h)
        y_proj = np.clip(c_pos[:, 1], y_p - t_h, y_p + t_h)
        z_proj = z_p
        d_i = np.sqrt(
            (c_pos[:, 0] - x_proj) ** 2 + (c_pos[:, 1] - y_proj) ** 2 + (c_pos[:, 2] - z_proj) ** 2
        )
    else:
        d_i = np.sqrt(np.sum((c_pos - projectile.pos)**2, axis=1))
    penetration = np.maximum(0.0, (projectile.pos[2] - c_pos[:, 2]) * direction)

    k_val = k_contact if k_contact is not None else 10.0 * np.mean(grid.stiffnesses)

    w_i = 1.0 / np.maximum(d_i, 1e-4)
    w_mean = np.mean(w_i) if len(w_i) > 0 else 1.0
    w_normalized = w_i / w_mean if w_mean > 0.0 else np.ones_like(w_i)

    f_i = k_val * w_normalized * penetration

    forces_c = np.zeros((len(contact_nodes), 3))
    forces_c[:, 2] = f_i * direction
    forces[contact_nodes] = forces_c

    return forces


def check_termination(
    projectile: Projectile,
    grid: Grid,
    positions: np.ndarray,
    t_current: float,
    t_max: float,
    initial_velocity_z: float,
) -> str | None:
    """Check if the simulation should terminate."""
    if (
        np.sign(projectile.vel[2]) != np.sign(initial_velocity_z)
        or projectile.vel[2] == 0.0
    ):
        return "arrest"

    if t_current >= t_max:
        return "timeout"

    initial_grid_z = 0.0
    direction = np.sign(initial_velocity_z) if initial_velocity_z != 0.0 else 1.0
    has_passed = (projectile.pos[2] - initial_grid_z) * direction > 0.0

    if has_passed:
        contact_nodes = projectile.contact_nodes
        failed_np = np.asarray(grid.failed)
        if len(contact_nodes) > 0:
            contact_mask = np.zeros(grid.n_nodes, dtype=bool)
            contact_mask[contact_nodes] = True
            connected_to_contact = (
                contact_mask[grid.springs[:, 0]] | contact_mask[grid.springs[:, 1]]
            )
            if np.any(connected_to_contact):
                if np.all(failed_np[connected_to_contact]):
                    return "penetration"
        else:
            if np.all(failed_np):
                return "penetration"

    return None


def generate_impact_report(
    projectile: Projectile,
    initial_ke: float,
    termination_reason: str,
) -> dict[str, Any]:
    """Generate a summary report of the impact event."""
    residual_ke = 0.5 * projectile.mass * np.sum(projectile.vel**2)
    exit_velocity = float(np.sqrt(np.sum(projectile.vel**2)))
    energy_absorbed = initial_ke - residual_ke

    is_penetration = termination_reason == "penetration"
    is_arrest = termination_reason == "arrest"

    return {
        "arrested": is_arrest,
        "penetration": is_penetration,
        "timeout": termination_reason == "timeout",
        "exit_velocity_m_s": exit_velocity if is_penetration else 0.0,
        "residual_ke_j": residual_ke if is_penetration else 0.0,
        "energy_absorbed_j": float(energy_absorbed),
        "termination_reason": termination_reason,
    }
