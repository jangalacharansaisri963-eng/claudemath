"""Vector operations module for claudemath.

Pure-Python implementation of n-dimensional vector arithmetic, vector norms,
projections, Gram-Schmidt orthogonalization, 3D spatial rotations (Rodrigues, Quaternions, Euler),
distances, and spherical linear interpolation.
"""

from typing import List, Tuple, Optional, Callable
import math


# ----------------------------------------------------
# 1. Fundamental Vector Arithmetic (n-D)
# ----------------------------------------------------

def vector_add(u: List[float], v: List[float]) -> List[float]:
    """Element-wise addition u + v."""
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match")
    return [a + b for a, b in zip(u, v)]


def vector_sub(u: List[float], v: List[float]) -> List[float]:
    """Element-wise subtraction u - v."""
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match")
    return [a - b for a, b in zip(u, v)]


def vector_scalar_mul(v: List[float], s: float) -> List[float]:
    """Scalar multiplication s * v."""
    return [x * s for x in v]


def vector_scalar_div(v: List[float], s: float) -> List[float]:
    """Scalar division v / s."""
    if s == 0:
        raise ZeroDivisionError("Division by scalar zero")
    return [x / s for x in v]


def vector_negate(v: List[float]) -> List[float]:
    """Negate vector -v."""
    return [-x for x in v]


def vector_dot(u: List[float], v: List[float]) -> float:
    """Inner (dot) product u . v = sum(u_i * v_i)."""
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match")
    return sum(a * b for a, b in zip(u, v))


def vector_hadamard(u: List[float], v: List[float]) -> List[float]:
    """Element-wise (Hadamard) product of vectors."""
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match")
    return [a * b for a, b in zip(u, v)]


# ----------------------------------------------------
# 2. Vector Norms and Normalization
# ----------------------------------------------------

def vector_norm_l1(v: List[float]) -> float:
    """Manhattan (L1) norm sum |v_i|."""
    return sum(abs(x) for x in v)


def vector_norm_l2(v: List[float]) -> float:
    """Euclidean (L2) norm sqrt(sum v_i^2)."""
    return math.sqrt(sum(x * x for x in v))


def vector_norm_squared(v: List[float]) -> float:
    """Squared Euclidean norm sum v_i^2."""
    return sum(x * x for x in v)


def vector_norm_inf(v: List[float]) -> float:
    """Chebyshev / Infinity (L_inf) norm max |v_i|."""
    if not v:
        return 0.0
    return max(abs(x) for x in v)


def vector_norm_lp(v: List[float], p: float) -> float:
    """Minkowski (Lp) norm (sum |v_i|^p)^(1/p)."""
    if p <= 0:
        raise ValueError("p must be positive")
    return (sum(abs(x) ** p for x in v)) ** (1.0 / p)


def vector_normalize(v: List[float]) -> List[float]:
    """Return unit vector v / ||v||_2."""
    norm = vector_norm_l2(v)
    if norm == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / norm for x in v]


def is_unit_vector(v: List[float], tol: float = 1e-9) -> bool:
    """Check if vector has unit Euclidean length."""
    return abs(vector_norm_l2(v) - 1.0) < tol


# ----------------------------------------------------
# 3. 3D Specific Vector Products
# ----------------------------------------------------

def vector_cross_3d(u: List[float], v: List[float]) -> List[float]:
    """3D Cross product u x v."""
    if len(u) != 3 or len(v) != 3:
        raise ValueError("Cross product defined for 3D vectors")
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ]


def scalar_triple_product_3d(u: List[float], v: List[float], w: List[float]) -> float:
    """Scalar triple product u . (v x w) = volume of parallelepiped."""
    return vector_dot(u, vector_cross_3d(v, w))


def vector_triple_product_3d(u: List[float], v: List[float], w: List[float]) -> List[float]:
    """Vector triple product u x (v x w) = (u . w) v - (u . v) w (Lagrange formula)."""
    return vector_cross_3d(u, vector_cross_3d(v, w))


# ----------------------------------------------------
# 4. Projections, Angles, and Orthogonality
# ----------------------------------------------------

def vector_project(u: List[float], v: List[float]) -> List[float]:
    """Vector projection of u onto v: (u . v / ||v||^2) * v."""
    v_sq = vector_norm_squared(v)
    if v_sq == 0:
        raise ValueError("Cannot project onto zero vector")
    scale = vector_dot(u, v) / v_sq
    return [x * scale for x in v]


def vector_reject(u: List[float], v: List[float]) -> List[float]:
    """Vector rejection of u orthogonal to v: u - proj_v(u)."""
    return vector_sub(u, vector_project(u, v))


def vector_angle(u: List[float], v: List[float]) -> float:
    """Angle theta in radians between u and v in [0, pi]."""
    nu = vector_norm_l2(u)
    nv = vector_norm_l2(v)
    if nu == 0 or nv == 0:
        raise ValueError("Cannot compute angle with zero vector")
    cos_theta = vector_dot(u, v) / (nu * nv)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def cosine_similarity(u: List[float], v: List[float]) -> float:
    """Cosine similarity cos(theta) = (u . v) / (||u|| * ||v||)."""
    nu = vector_norm_l2(u)
    nv = vector_norm_l2(v)
    if nu == 0 or nv == 0:
        return 0.0
    return vector_dot(u, v) / (nu * nv)


def is_orthogonal(u: List[float], v: List[float], tol: float = 1e-9) -> bool:
    """Check if vectors are orthogonal (u . v == 0)."""
    return abs(vector_dot(u, v)) < tol


def is_parallel(u: List[float], v: List[float], tol: float = 1e-9) -> bool:
    """Check if vectors are parallel (|cos(theta)| == 1)."""
    return abs(abs(cosine_similarity(u, v)) - 1.0) < tol


def vector_reflect(v: List[float], normal: List[float]) -> List[float]:
    """Reflect vector v across surface with unit normal n: v - 2*(v . n)*n."""
    n_hat = vector_normalize(normal)
    factor = 2.0 * vector_dot(v, n_hat)
    return vector_sub(v, [factor * x for x in n_hat])


def vector_refract(v: List[float], normal: List[float], eta: float) -> Optional[List[float]]:
    """Snell's refraction of incident vector v across surface with unit normal n. Returns None if total internal reflection."""
    i = vector_normalize(v)
    n = vector_normalize(normal)
    cos_i = -vector_dot(n, i)
    sin_t2 = eta * eta * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return None  # Total internal reflection
    cos_t = math.sqrt(1.0 - sin_t2)
    # Refracted vector = eta * i + (eta * cos_i - cos_t) * n
    term1 = [eta * x for x in i]
    term2 = [(eta * cos_i - cos_t) * x for x in n]
    return vector_add(term1, term2)


# ----------------------------------------------------
# 5. Distances Between Vectors
# ----------------------------------------------------

def distance_euclidean(u: List[float], v: List[float]) -> float:
    """Euclidean distance ||u - v||_2."""
    return vector_norm_l2(vector_sub(u, v))


def distance_squared_euclidean(u: List[float], v: List[float]) -> float:
    """Squared Euclidean distance sum (u_i - v_i)^2."""
    return vector_norm_squared(vector_sub(u, v))


def distance_manhattan(u: List[float], v: List[float]) -> float:
    """Manhattan distance sum |u_i - v_i|."""
    return vector_norm_l1(vector_sub(u, v))


def distance_chebyshev(u: List[float], v: List[float]) -> float:
    """Chebyshev distance max |u_i - v_i|."""
    return vector_norm_inf(vector_sub(u, v))


def distance_minkowski(u: List[float], v: List[float], p: float) -> float:
    """Minkowski distance (sum |u_i - v_i|^p)^(1/p)."""
    return vector_norm_lp(vector_sub(u, v), p)


def distance_canberra(u: List[float], v: List[float]) -> float:
    """Canberra distance sum |u_i - v_i| / (|u_i| + |v_i|)."""
    if len(u) != len(v):
        raise ValueError("Vector dimensions must match")
    total = 0.0
    for a, b in zip(u, v):
        den = abs(a) + abs(b)
        if den > 1e-15:
            total += abs(a - b) / den
    return total


def distance_bray_curtis(u: List[float], v: List[float]) -> float:
    """Bray-Curtis dissimilarity sum |u_i - v_i| / sum |u_i + v_i|."""
    num = sum(abs(a - b) for a, b in zip(u, v))
    den = sum(abs(a + b) for a, b in zip(u, v))
    if den == 0:
        return 0.0
    return num / den


# ----------------------------------------------------
# 6. Gram-Schmidt Orthogonalization
# ----------------------------------------------------

def gram_schmidt_classical(basis: List[List[float]]) -> List[List[float]]:
    """Classical Gram-Schmidt orthogonalization."""
    ortho = []
    for v in basis:
        w = list(v)
        for u in ortho:
            proj = vector_project(v, u)
            w = vector_sub(w, proj)
        if vector_norm_l2(w) > 1e-12:
            ortho.append(w)
    return ortho


def gram_schmidt_modified(basis: List[List[float]]) -> List[List[float]]:
    """Modified Gram-Schmidt with orthonormal basis outputs."""
    k = len(basis)
    V = [list(row) for row in basis]
    orthonormal = []
    for i in range(k):
        norm_v = vector_norm_l2(V[i])
        if norm_v < 1e-12:
            continue
        q = vector_scalar_mul(V[i], 1.0 / norm_v)
        orthonormal.append(q)
        for j in range(i + 1, k):
            proj_scale = vector_dot(V[j], q)
            V[j] = vector_sub(V[j], vector_scalar_mul(q, proj_scale))
    return orthonormal


# ----------------------------------------------------
# 7. 3D Rotations and Quaternions
# ----------------------------------------------------

def rodrigues_rotation(v: List[float], axis: List[float], theta: float) -> List[float]:
    """Rotate 3D vector v around unit axis by angle theta using Rodrigues formula:
    v_rot = v*cos(theta) + (k x v)*sin(theta) + k*(k . v)*(1 - cos(theta))."""
    k = vector_normalize(axis)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    k_cross_v = vector_cross_3d(k, v)
    k_dot_v = vector_dot(k, v)

    term1 = [x * cos_t for x in v]
    term2 = [x * sin_t for x in k_cross_v]
    term3 = [x * (k_dot_v * (1.0 - cos_t)) for x in k]
    return vector_add(vector_add(term1, term2), term3)


# Quaternion representation: [w, x, y, z] = w + x*i + y*j + z*k
def quaternion_mul(q1: List[float], q2: List[float]) -> List[float]:
    """Hamilton quaternion product q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ]


def quaternion_conjugate(q: List[float]) -> List[float]:
    """Quaternion conjugate q* = [w, -x, -y, -z]."""
    return [q[0], -q[1], -q[2], -q[3]]


def quaternion_norm(q: List[float]) -> float:
    """Quaternion norm ||q|| = sqrt(w^2 + x^2 + y^2 + z^2)."""
    return math.sqrt(sum(x * x for x in q))


def quaternion_normalize(q: List[float]) -> List[float]:
    """Return unit quaternion."""
    n = quaternion_norm(q)
    if n == 0:
        raise ValueError("Cannot normalize zero quaternion")
    return [x / n for x in q]


def quaternion_inverse(q: List[float]) -> List[float]:
    """Quaternion multiplicative inverse q^{-1} = q* / ||q||^2."""
    n_sq = sum(x * x for x in q)
    if n_sq == 0:
        raise ZeroDivisionError("Cannot invert zero quaternion")
    conj = quaternion_conjugate(q)
    return [x / n_sq for x in conj]


def quaternion_from_axis_angle(axis: List[float], theta: float) -> List[float]:
    """Create unit quaternion from rotation axis and angle."""
    k = vector_normalize(axis)
    half = theta * 0.5
    s = math.sin(half)
    return [math.cos(half), k[0] * s, k[1] * s, k[2] * s]


def quaternion_to_axis_angle(q: List[float]) -> Tuple[List[float], float]:
    """Extract rotation axis and angle from unit quaternion."""
    q_u = quaternion_normalize(q)
    w = max(-1.0, min(1.0, q_u[0]))
    theta = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return [1.0, 0.0, 0.0], 0.0
    return [q_u[1] / s, q_u[2] / s, q_u[3] / s], theta


def quaternion_rotate_vector(q: List[float], v: List[float]) -> List[float]:
    """Rotate 3D vector v using unit quaternion: q * [0, v] * q^{-1}."""
    q_u = quaternion_normalize(q)
    v_quat = [0.0, v[0], v[1], v[2]]
    rotated_q = quaternion_mul(quaternion_mul(q_u, v_quat), quaternion_conjugate(q_u))
    return [rotated_q[1], rotated_q[2], rotated_q[3]]


def quaternion_slerp(q0: List[float], q1: List[float], t: float) -> List[float]:
    """Spherical linear interpolation between two unit quaternions."""
    q0_u = quaternion_normalize(q0)
    q1_u = quaternion_normalize(q1)
    dot = sum(a * b for a, b in zip(q0_u, q1_u))
    if dot < 0.0:
        q1_u = [-x for x in q1_u]
        dot = -dot
    dot = min(1.0, dot)
    if dot > 0.9995:
        # Linear interpolation for very close angles
        res = [q0_u[i] + t * (q1_u[i] - q0_u[i]) for i in range(4)]
        return quaternion_normalize(res)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return [s0 * q0_u[i] + s1 * q1_u[i] for i in range(4)]


# ----------------------------------------------------
# 8. Interpolation and Barycentric Coordinates
# ----------------------------------------------------

def vector_lerp(u: List[float], v: List[float], t: float) -> List[float]:
    """Linear interpolation (1 - t)*u + t*v."""
    return [a + t * (b - a) for a, b in zip(u, v)]


def vector_slerp(u: List[float], v: List[float], t: float) -> List[float]:
    """Spherical linear interpolation between two unit vectors."""
    u_unit = vector_normalize(u)
    v_unit = vector_normalize(v)
    omega = vector_angle(u_unit, v_unit)
    if abs(omega) < 1e-9:
        return u_unit
    sin_om = math.sin(omega)
    s0 = math.sin((1.0 - t) * omega) / sin_om
    s1 = math.sin(t * omega) / sin_om
    return vector_add(vector_scalar_mul(u_unit, s0), vector_scalar_mul(v_unit, s1))
def euler_angles_to_rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """Construct 3x3 rotation matrix from Euler angles (roll=X, pitch=Y, yaw=Z)."""
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cz, -cy * sz, sy],
        [sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy],
        [-cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy]
    ]


def rotation_matrix_apply(R: List[List[float]], v: List[float]) -> List[float]:
    """Apply 3x3 rotation matrix R to 3D vector v."""
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def quaternion_to_rotation_matrix(q: List[float]) -> List[List[float]]:
    """Convert unit quaternion [w, x, y, z] to 3x3 orthogonal rotation matrix."""
    w, x, y, z = quaternion_normalize(q)
    return [
        [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w), 2.0*(x*z + y*w)],
        [2.0*(x*y + z*w), 1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
        [2.0*(x*z - y*w), 2.0*(y*z + x*w), 1.0 - 2.0*(x*x + y*y)]
    ]


def rotation_matrix_to_quaternion(R: List[List[float]]) -> List[float]:
    """Convert 3x3 rotation matrix to unit quaternion."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return [0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s]
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        return [(R[2][1] - R[1][2]) / s, 0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s]
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        return [(R[0][2] - R[2][0]) / s, (R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s]
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        return [(R[1][0] - R[0][1]) / s, (R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s]


def triangle_barycentric_coordinates_2d(p: List[float], a: List[float],
                                        b: List[float], c: List[float]) -> Tuple[float, float, float]:
    """Compute barycentric weights (w_a, w_b, w_c) of 2D point p with respect to triangle abc."""
    det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(det) < 1e-14:
        raise ValueError("Degenerate triangle")
    w_a = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / det
    w_b = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / det
    w_c = 1.0 - w_a - w_b
    return w_a, w_b, w_c


def point_in_triangle_2d(p: List[float], a: List[float], b: List[float], c: List[float], tol: float = 1e-9) -> bool:
    """Check if point p lies inside 2D triangle abc."""
    wa, wb, wc = triangle_barycentric_coordinates_2d(p, a, b, c)
    return wa >= -tol and wb >= -tol and wc >= -tol


def vector_cross_7d(u: List[float], v: List[float]) -> List[float]:
    """7-dimensional cross product based on octonion multiplication table."""
    if len(u) != 7 or len(v) != 7:
        raise ValueError("Vectors must have dimension 7")
    # Standard Fano plane multiplication rules:
    # (1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5) using 0-based indices:
    # 0,1,2; 0,3,4; 0,6,5; 1,3,5; 1,4,6; 2,3,6; 2,5,4
    w = [0.0] * 7
    # e123
    w[2] += u[0]*v[1] - u[1]*v[0]
    w[0] += u[1]*v[2] - u[2]*v[1]
    w[1] += u[2]*v[0] - u[0]*v[2]
    # e145
    w[4] += u[0]*v[3] - u[3]*v[0]
    w[0] += u[3]*v[4] - u[4]*v[3]
    w[3] += u[4]*v[0] - u[0]*v[4]
    # e176
    w[5] += u[0]*v[6] - u[6]*v[0]
    w[0] += u[6]*v[5] - u[5]*v[6]
    w[6] += u[5]*v[0] - u[0]*v[5]
    # e246
    w[5] += u[1]*v[3] - u[3]*v[1]
    w[1] += u[3]*v[5] - u[5]*v[3]
    w[3] += u[5]*v[1] - u[1]*v[5]
    # e257
    w[6] += u[1]*v[4] - u[4]*v[1]
    w[1] += u[4]*v[6] - u[6]*v[4]
    w[4] += u[6]*v[1] - u[1]*v[6]
    # e347
    w[6] += u[2]*v[3] - u[3]*v[2]
    w[2] += u[3]*v[6] - u[6]*v[3]
    w[3] += u[6]*v[2] - u[2]*v[6]
    # e365
    w[4] += u[2]*v[5] - u[5]*v[2]
    w[2] += u[5]*v[4] - u[4]*v[5]
    w[5] += u[4]*v[2] - u[2]*v[4]
    return w


def distance_cosine(u: List[float], v: List[float]) -> float:
    """Cosine distance: 1 - cosine_similarity(u, v)."""
    return 1.0 - cosine_similarity(u, v)


def vector_midpoint(u: List[float], v: List[float]) -> List[float]:
    """Midpoint between two vectors: 0.5 * (u + v)."""
    return vector_scalar_mul(vector_add(u, v), 0.5)


def vector_clamp(v: List[float], min_val: float, max_val: float) -> List[float]:
    """Clamp all vector components to interval [min_val, max_val]."""
    return [max(min_val, min(max_val, x)) for x in v]


def vector_area_parallelogram_3d(u: List[float], v: List[float]) -> float:
    """Area of 3D parallelogram spanned by vectors u and v: ||u x v||."""
    return vector_norm_l2(vector_cross_3d(u, v))


def vector_volume_tetrahedron_3d(a: List[float], b: List[float], c: List[float], d: List[float]) -> float:
    """Volume of 3D tetrahedron with vertices a, b, c, d: |(b - a) . ((c - a) x (d - a))| / 6."""
    u = vector_sub(b, a)
    v = vector_sub(c, a)
    w = vector_sub(d, a)
    return abs(scalar_triple_product_3d(u, v, w)) / 6.0


def vector_distance_to_line_3d(point: List[float], line_point: List[float], line_dir: List[float]) -> float:
    """Perpendicular distance from point to 3D line."""
    d_hat = vector_normalize(line_dir)
    diff = vector_sub(point, line_point)
    cross = vector_cross_3d(diff, d_hat)
    return vector_norm_l2(cross)


def vector_distance_to_plane_3d(point: List[float], plane_point: List[float], plane_normal: List[float]) -> float:
    """Signed perpendicular distance from point to 3D plane."""
    n_hat = vector_normalize(plane_normal)
    return vector_dot(vector_sub(point, plane_point), n_hat)
