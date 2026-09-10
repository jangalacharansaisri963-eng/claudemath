"""Calculus module for claudemath.

Pure-Python implementations of differentiation, integration, vector calculus,
limits, curve differential geometry, optimization, Taylor expansions, and special mathematical functions.
"""

from typing import Callable, List, Tuple, Optional
import math
import cmath

# ----------------------------------------------------
# 1. Numerical Differentiation (Single Variable)
# ----------------------------------------------------

def forward_difference(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Compute first derivative using forward finite difference: (f(x + h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h

def backward_difference(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Compute first derivative using backward finite difference: (f(x) - f(x - h)) / h."""
    return (f(x) - f(x - h)) / h

def central_difference(f: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """Compute first derivative using central finite difference: (f(x + h) - f(x - h)) / (2*h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)

def central_difference_5point(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Compute first derivative using 5-point stencil (O(h^4) accuracy)."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12.0 * h)

def central_difference_7point(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Compute first derivative using 7-point stencil (O(h^6) accuracy)."""
    return (-f(x - 3 * h) + 9 * f(x - 2 * h) - 45 * f(x - h) + 45 * f(x + h) - 9 * f(x + 2 * h) + f(x + 3 * h)) / (60.0 * h)

def complex_step_derivative(f: Callable[[complex], complex], x: float, h: float = 1e-20) -> float:
    """Compute derivative using complex-step differentiation (virtually zero subtractive cancellation)."""
    return f(complex(x, h)).imag / h

def second_derivative(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Compute second derivative using central difference: (f(x+h) - 2*f(x) + f(x-h)) / h^2."""
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)

def second_derivative_5point(f: Callable[[float], float], x: float, h: float = 1e-3) -> float:
    """Compute second derivative using 5-point stencil (O(h^4) error)."""
    return (-f(x + 2 * h) + 16 * f(x + h) - 30 * f(x) + 16 * f(x - h) - f(x - 2 * h)) / (12.0 * h * h)

def third_derivative(f: Callable[[float], float], x: float, h: float = 1e-3) -> float:
    """Compute third derivative using central difference."""
    return (f(x + 2 * h) - 2.0 * f(x + h) + 2.0 * f(x - h) - f(x - 2 * h)) / (2.0 * (h ** 3))

def fourth_derivative(f: Callable[[float], float], x: float, h: float = 1e-2) -> float:
    """Compute fourth derivative using central difference."""
    return (f(x + 2 * h) - 4.0 * f(x + h) + 6.0 * f(x) - 4.0 * f(x - h) + f(x - 2 * h)) / (h ** 4)

def nth_derivative(f: Callable[[float], float], x: float, n: int, h: float = 1e-3) -> float:
    """Compute n-th numerical derivative using finite differences."""
    if n < 0:
        raise ValueError("Derivative order n must be non-negative")
    if n == 0:
        return f(x)
    if n == 1:
        return central_difference(f, x, h)
    if n == 2:
        return second_derivative(f, x, h)
    val = 0.0
    for k in range(n + 1):
        coeff = math.comb(n, k) * ((-1) ** (n - k))
        val += coeff * f(x + k * h)
    return val / (h ** n)

# ----------------------------------------------------
# 2. Multivariable Differentiation & Vector Calculus
# ----------------------------------------------------

def partial_derivative(f: Callable[[List[float]], float], point: List[float], var_idx: int, h: float = 1e-6) -> float:
    """Compute partial derivative of f with respect to variable at var_idx."""
    p_plus = list(point)
    p_minus = list(point)
    p_plus[var_idx] += h
    p_minus[var_idx] -= h
    return (f(p_plus) - f(p_minus)) / (2.0 * h)

def second_partial_derivative(f: Callable[[List[float]], float], point: List[float], var_idx: int, h: float = 1e-4) -> float:
    """Compute second partial derivative d^2 f / d x_i^2."""
    p_plus = list(point)
    p_minus = list(point)
    p_plus[var_idx] += h
    p_minus[var_idx] -= h
    return (f(p_plus) - 2.0 * f(point) + f(p_minus)) / (h * h)

def mixed_partial_derivative(f: Callable[[List[float]], float], point: List[float], idx_i: int, idx_j: int, h: float = 1e-4) -> float:
    """Compute mixed partial derivative d^2 f / (d x_i d x_j)."""
    if idx_i == idx_j:
        return second_partial_derivative(f, point, idx_i, h)
    pp, pm, mp, mm = list(point), list(point), list(point), list(point)
    pp[idx_i] += h; pp[idx_j] += h
    pm[idx_i] += h; pm[idx_j] -= h
    mp[idx_i] -= h; mp[idx_j] += h
    mm[idx_i] -= h; mm[idx_j] -= h
    return (f(pp) - f(pm) - f(mp) + f(mm)) / (4.0 * h * h)

def numerical_gradient(f: Callable[[List[float]], float], point: List[float], h: float = 1e-6) -> List[float]:
    """Compute the numerical gradient vector of scalar field f at point."""
    return [partial_derivative(f, point, i, h) for i in range(len(point))]

def directional_derivative(f: Callable[[List[float]], float], point: List[float], direction: List[float], h: float = 1e-6) -> float:
    """Compute directional derivative of f at point in specified direction."""
    norm = math.sqrt(sum(d * d for d in direction))
    if norm == 0:
        raise ValueError("Direction vector cannot be zero")
    unit_dir = [d / norm for d in direction]
    grad = numerical_gradient(f, point, h)
    return sum(g * u for g, u in zip(grad, unit_dir))

def jacobian_matrix(funcs: List[Callable[[List[float]], float]], point: List[float], h: float = 1e-6) -> List[List[float]]:
    """Compute the Jacobian matrix J_ij = d f_i / d x_j."""
    return [[partial_derivative(f, point, j, h) for j in range(len(point))] for f in funcs]

def hessian_matrix(f: Callable[[List[float]], float], point: List[float], h: float = 1e-4) -> List[List[float]]:
    """Compute the symmetric Hessian matrix H_ij = d^2 f / (d x_i d x_j)."""
    n = len(point)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            val = mixed_partial_derivative(f, point, i, j, h)
            H[i][j] = val
            H[j][i] = val
    return H

def divergence_2d(f_x: Callable[[List[float]], float], f_y: Callable[[List[float]], float], point: List[float], h: float = 1e-6) -> float:
    """Compute 2D divergence div F = d F_x / dx + d F_y / dy."""
    return partial_derivative(f_x, point, 0, h) + partial_derivative(f_y, point, 1, h)

def divergence_3d(f_x: Callable[[List[float]], float], f_y: Callable[[List[float]], float],
                  f_z: Callable[[List[float]], float], point: List[float], h: float = 1e-6) -> float:
    """Compute 3D divergence div F."""
    return (partial_derivative(f_x, point, 0, h) +
            partial_derivative(f_y, point, 1, h) +
            partial_derivative(f_z, point, 2, h))

def curl_2d(f_x: Callable[[List[float]], float], f_y: Callable[[List[float]], float], point: List[float], h: float = 1e-6) -> float:
    """Compute scalar 2D curl: d F_y / dx - d F_x / dy."""
    return partial_derivative(f_y, point, 0, h) - partial_derivative(f_x, point, 1, h)

def curl_3d(f_x: Callable[[List[float]], float], f_y: Callable[[List[float]], float],
            f_z: Callable[[List[float]], float], point: List[float], h: float = 1e-6) -> List[float]:
    """Compute 3D vector curl."""
    c_x = partial_derivative(f_z, point, 1, h) - partial_derivative(f_y, point, 2, h)
    c_y = partial_derivative(f_x, point, 2, h) - partial_derivative(f_z, point, 0, h)
    c_z = partial_derivative(f_y, point, 0, h) - partial_derivative(f_x, point, 1, h)
    return [c_x, c_y, c_z]

def laplacian_2d(f: Callable[[List[float]], float], point: List[float], h: float = 1e-4) -> float:
    """Compute 2D Laplacian Delta f = d^2 f / dx^2 + d^2 f / dy^2."""
    return second_partial_derivative(f, point, 0, h) + second_partial_derivative(f, point, 1, h)

def laplacian_3d(f: Callable[[List[float]], float], point: List[float], h: float = 1e-4) -> float:
    """Compute 3D Laplacian Delta f."""
    return (second_partial_derivative(f, point, 0, h) +
            second_partial_derivative(f, point, 1, h) +
            second_partial_derivative(f, point, 2, h))

def biharmonic_2d(f: Callable[[List[float]], float], point: List[float], h: float = 1e-3) -> float:
    """Compute 2D biharmonic operator Delta^2 f."""
    lap = lambda p: laplacian_2d(f, p, h)
    return laplacian_2d(lap, point, h)

def vector_laplacian_3d(f_x: Callable[[List[float]], float], f_y: Callable[[List[float]], float],
                        f_z: Callable[[List[float]], float], point: List[float], h: float = 1e-4) -> List[float]:
    """Compute vector Laplacian Delta F = [Delta F_x, Delta F_y, Delta F_z]."""
    return [laplacian_3d(f_x, point, h), laplacian_3d(f_y, point, h), laplacian_3d(f_z, point, h)]

# ----------------------------------------------------
# 3. Numerical Integration (Quadrature)
# ----------------------------------------------------

def trapezoidal_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Single-interval trapezoidal rule: (b - a) * (f(a) + f(b)) / 2."""
    return (b - a) * (f(a) + f(b)) * 0.5

def composite_trapezoid(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite trapezoidal rule with n intervals."""
    if n <= 0:
        raise ValueError("n must be positive")
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h

def simpson_13_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Single Simpson's 1/3 rule."""
    mid = (a + b) * 0.5
    return ((b - a) / 6.0) * (f(a) + 4.0 * f(mid) + f(b))

def composite_simpson_13(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite Simpson's 1/3 rule with n subintervals (rounded up to even)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * f(x)
    return (h / 3.0) * total

def simpson_38_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Single Simpson's 3/8 rule."""
    h = (b - a) / 3.0
    return (3.0 * h / 8.0) * (f(a) + 3.0 * f(a + h) + 3.0 * f(a + 2 * h) + f(b))

def composite_simpson_38(f: Callable[[float], float], a: float, b: float, n: int = 999) -> float:
    """Composite Simpson's 3/8 rule."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 3 != 0:
        n = (n // 3 + 1) * 3
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        coeff = 2.0 if (i % 3 == 0) else 3.0
        total += coeff * f(a + i * h)
    return (3.0 * h / 8.0) * total

def boole_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Single Boole's rule (5-point Newton-Cotes)."""
    h = (b - a) / 4.0
    return (2.0 * h / 45.0) * (7 * f(a) + 32 * f(a + h) + 12 * f(a + 2 * h) + 32 * f(a + 3 * h) + 7 * f(b))

def composite_boole(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite Boole's rule."""
    if n % 4 != 0:
        n = (n // 4 + 1) * 4
    h = (b - a) / n
    total = 0.0
    for i in range(0, n, 4):
        total += boole_rule(f, a + i * h, a + (i + 4) * h)
    return total

def midpoint_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Single midpoint rule."""
    return (b - a) * f((a + b) * 0.5)

def composite_midpoint(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite midpoint rule with n intervals."""
    if n <= 0:
        raise ValueError("n must be positive")
    h = (b - a) / n
    total = sum(f(a + (i + 0.5) * h) for i in range(n))
    return h * total

def romberg_integration(f: Callable[[float], float], a: float, b: float, max_steps: int = 8, tol: float = 1e-10) -> float:
    """Romberg integration."""
    R = [[0.0] * (max_steps + 1) for _ in range(max_steps + 1)]
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))
    for i in range(1, max_steps + 1):
        h *= 0.5
        sum_pts = sum(f(a + (2 * k - 1) * h) for k in range(1, (1 << (i - 1)) + 1))
        R[i][0] = 0.5 * R[i - 1][0] + h * sum_pts
        for j in range(1, i + 1):
            factor = 4 ** j
            R[i][j] = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0)
        if i > 2 and abs(R[i][i] - R[i - 1][i - 1]) < tol:
            return R[i][i]
    return R[max_steps][max_steps]

def adaptive_simpson_quadrature(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_depth: int = 25) -> float:
    """Adaptive Simpson quadrature."""
    def _asr(fa, fb, fc, a_val, b_val, c_val, cur_tol, whole, depth):
        d = (a_val + c_val) * 0.5
        e = (c_val + b_val) * 0.5
        fd, fe = f(d), f(e)
        left = ((c_val - a_val) / 6.0) * (fa + 4.0 * fd + fc)
        right = ((b_val - c_val) / 6.0) * (fc + 4.0 * fe + fb)
        delta = left + right - whole
        if depth <= 0 or abs(delta) <= 15.0 * cur_tol:
            return left + right + delta / 15.0
        return (_asr(fa, fc, fd, a_val, c_val, d, cur_tol * 0.5, left, depth - 1) +
                _asr(fc, fb, fe, c_val, b_val, e, cur_tol * 0.5, right, depth - 1))
    c = (a + b) * 0.5
    fa, fb, fc = f(a), f(b), f(c)
    whole = ((b - a) / 6.0) * (fa + 4.0 * fc + fb)
    return _asr(fa, fb, fc, a, b, c, tol, whole, max_depth)

def gauss_legendre_2pt(f: Callable[[float], float], a: float, b: float) -> float:
    """2-point Gauss-Legendre quadrature."""
    c = 1.0 / math.sqrt(3.0)
    mid = 0.5 * (b + a)
    half_w = 0.5 * (b - a)
    return half_w * (f(mid - half_w * c) + f(mid + half_w * c))

def gauss_legendre_3pt(f: Callable[[float], float], a: float, b: float) -> float:
    """3-point Gauss-Legendre quadrature."""
    c = math.sqrt(3.0 / 5.0)
    mid = 0.5 * (b + a)
    half_w = 0.5 * (b - a)
    return half_w * ((5.0 / 9.0) * f(mid - half_w * c) +
                     (8.0 / 9.0) * f(mid) +
                     (5.0 / 9.0) * f(mid + half_w * c))

def gauss_legendre_4pt(f: Callable[[float], float], a: float, b: float) -> float:
    """4-point Gauss-Legendre quadrature."""
    x1 = math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
    x2 = math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0)
    w1 = (18.0 + math.sqrt(30.0)) / 36.0
    w2 = (18.0 - math.sqrt(30.0)) / 36.0
    mid = 0.5 * (b + a)
    half_w = 0.5 * (b - a)
    return half_w * (w1 * f(mid - half_w * x1) + w1 * f(mid + half_w * x1) +
                     w2 * f(mid - half_w * x2) + w2 * f(mid + half_w * x2))

def gauss_legendre_5pt(f: Callable[[float], float], a: float, b: float) -> float:
    """5-point Gauss-Legendre quadrature."""
    x1 = (1.0 / 3.0) * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0))
    x2 = (1.0 / 3.0) * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0))
    w0 = 128.0 / 225.0
    w1 = (322.0 + 13.0 * math.sqrt(70.0)) / 900.0
    w2 = (322.0 - 13.0 * math.sqrt(70.0)) / 900.0
    mid = 0.5 * (b + a)
    half_w = 0.5 * (b - a)
    return half_w * (w0 * f(mid) +
                     w1 * f(mid - half_w * x1) + w1 * f(mid + half_w * x1) +
                     w2 * f(mid - half_w * x2) + w2 * f(mid + half_w * x2))

def monte_carlo_integral_1d(f: Callable[[float], float], a: float, b: float, samples: int = 50000, seed: int = 42) -> float:
    """1D Monte Carlo integration over [a, b]."""
    state = seed
    total = 0.0
    for _ in range(samples):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        u = state / 4294967296.0
        x = a + u * (b - a)
        total += f(x)
    return (b - a) * (total / samples)

def double_integral_rect(f: Callable[[float, float], float], x_min: float, x_max: float,
                         y_min: float, y_max: float, nx: int = 100, ny: int = 100) -> float:
    """Double integral over rectangle [x_min, x_max] x [y_min, y_max]."""
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny
    total = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * hx
        for j in range(ny):
            y = y_min + (j + 0.5) * hy
            total += f(x, y)
    return total * hx * hy

def double_integral_general(f: Callable[[float, float], float], x_min: float, x_max: float,
                            y_lower: Callable[[float], float], y_upper: Callable[[float], float],
                            nx: int = 80, ny: int = 80) -> float:
    """Double integral with variable y-bounds: integral_{x_min}^{x_max} dx integral_{y_lower(x)}^{y_upper(x)} f(x, y) dy."""
    hx = (x_max - x_min) / nx
    total = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * hx
        yl = y_lower(x)
        yu = y_upper(x)
        if yu > yl:
            hy = (yu - yl) / ny
            inner_sum = sum(f(x, yl + (j + 0.5) * hy) for j in range(ny))
            total += inner_sum * hy
    return total * hx

def triple_integral_box(f: Callable[[float, float, float], float],
                        x_min: float, x_max: float,
                        y_min: float, y_max: float,
                        z_min: float, z_max: float,
                        nx: int = 40, ny: int = 40, nz: int = 40) -> float:
    """Triple integral over rectangular prism."""
    hx = (x_max - x_min) / nx
    hy = (y_max - y_min) / ny
    hz = (z_max - z_min) / nz
    total = 0.0
    for i in range(nx):
        x = x_min + (i + 0.5) * hx
        for j in range(ny):
            y = y_min + (j + 0.5) * hy
            for k in range(nz):
                z = z_min + (k + 0.5) * hz
                total += f(x, y, z)
    return total * hx * hy * hz

def improper_integral_upper_inf(f: Callable[[float], float], a: float) -> float:
    """Improper integral integral_a^inf f(x) dx via substitution t = 1/(x - a + 1)."""
    g = lambda t: f(a + (1.0 - t) / t) / (t * t) if t > 1e-12 else 0.0
    return composite_simpson_13(g, 1e-7, 1.0, 2000)

def improper_integral_both_inf(f: Callable[[float], float]) -> float:
    """Improper integral over (-inf, inf) via transformation x = t / (1 - t^2)."""
    g = lambda t: f(t / (1.0 - t * t)) * (1.0 + t * t) / ((1.0 - t * t) ** 2) if abs(t) < 0.999999 else 0.0
    return composite_simpson_13(g, -0.9999, 0.9999, 3000)

def line_integral_2d(f_x: Callable[[float, float], float], f_y: Callable[[float, float], float],
                     gamma_x: Callable[[float], float], gamma_y: Callable[[float], float],
                     t_start: float, t_end: float, n: int = 1000) -> float:
    """Line integral of 2D vector field along parametric curve gamma(t)."""
    dt = (t_end - t_start) / n
    total = 0.0
    for i in range(n):
        t = t_start + (i + 0.5) * dt
        x, y = gamma_x(t), gamma_y(t)
        dx_dt = central_difference(gamma_x, t, 1e-5)
        dy_dt = central_difference(gamma_y, t, 1e-5)
        total += (f_x(x, y) * dx_dt + f_y(x, y) * dy_dt)
    return total * dt

def line_integral_scalar_2d(f: Callable[[float, float], float],
                            gamma_x: Callable[[float], float], gamma_y: Callable[[float], float],
                            t_start: float, t_end: float, n: int = 1000) -> float:
    """Line integral of scalar field f(x, y) ds along curve gamma(t)."""
    dt = (t_end - t_start) / n
    total = 0.0
    for i in range(n):
        t = t_start + (i + 0.5) * dt
        x, y = gamma_x(t), gamma_y(t)
        speed = math.sqrt(central_difference(gamma_x, t, 1e-5) ** 2 + central_difference(gamma_y, t, 1e-5) ** 2)
        total += f(x, y) * speed
    return total * dt

def surface_flux_rect(f_z: Callable[[float, float, float], float],
                      x_min: float, x_max: float, y_min: float, y_max: float, z_const: float) -> float:
    """Flux of field [0, 0, f_z] through horizontal planar rectangle at z = z_const."""
    return double_integral_rect(lambda x, y: f_z(x, y, z_const), x_min, x_max, y_min, y_max)

# ----------------------------------------------------
# 4. Limits and Asymptotics
# ----------------------------------------------------

def limit_right(f: Callable[[float], float], x0: float, steps: int = 15) -> float:
    """Right-hand limit lim_{x -> x0+} f(x)."""
    h = 0.1
    vals = [f(x0 + h * (0.1 ** i)) for i in range(steps)]
    return vals[-1]

def limit_left(f: Callable[[float], float], x0: float, steps: int = 15) -> float:
    """Left-hand limit lim_{x -> x0-} f(x)."""
    h = 0.1
    vals = [f(x0 - h * (0.1 ** i)) for i in range(steps)]
    return vals[-1]

def limit_two_sided(f: Callable[[float], float], x0: float, tol: float = 1e-5) -> Optional[float]:
    """Two-sided limit if left and right limits agree."""
    r = limit_right(f, x0)
    l = limit_left(f, x0)
    if abs(r - l) <= tol:
        return (r + l) * 0.5
    return None

def limit_infinity(f: Callable[[float], float], steps: int = 8) -> float:
    """Limit lim_{x -> inf} f(x)."""
    vals = [f(100.0 * (10.0 ** i)) for i in range(steps)]
    return vals[-1]

def limit_neg_infinity(f: Callable[[float], float], steps: int = 8) -> float:
    """Limit lim_{x -> -inf} f(x)."""
    vals = [f(-100.0 * (10.0 ** i)) for i in range(steps)]
    return vals[-1]

def richardson_extrapolation(estimates: List[float], step_ratio: float = 2.0, order: int = 2) -> float:
    """Richardson extrapolation."""
    if len(estimates) < 2:
        return estimates[0] if estimates else 0.0
    factor = (step_ratio ** order)
    return (factor * estimates[-1] - estimates[-2]) / (factor - 1.0)

# ----------------------------------------------------
# 5. Curves, Geometry, and Applications
# ----------------------------------------------------

def tangent_line(f: Callable[[float], float], x0: float) -> Tuple[float, float]:
    """Return (slope m, intercept b) of tangent line y = m*x + b at x0."""
    m = central_difference(f, x0)
    b = f(x0) - m * x0
    return m, b

def normal_line(f: Callable[[float], float], x0: float) -> Tuple[float, float]:
    """Return (slope m_norm, intercept b_norm) of normal line at x0."""
    m = central_difference(f, x0)
    if abs(m) < 1e-15:
        raise ValueError("Tangent is horizontal; normal line is vertical")
    m_norm = -1.0 / m
    b_norm = f(x0) - m_norm * x0
    return m_norm, b_norm

def arc_length_numerical_1d(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Arc length of y = f(x) over [a, b]."""
    integrand = lambda x: math.sqrt(1.0 + (central_difference(f, x) ** 2))
    return composite_simpson_13(integrand, a, b, n)

def arc_length_param_2d(x_t: Callable[[float], float], y_t: Callable[[float], float],
                        t_start: float, t_end: float, n: int = 1000) -> float:
    """Arc length of 2D parametric curve."""
    integrand = lambda t: math.sqrt(central_difference(x_t, t) ** 2 + central_difference(y_t, t) ** 2)
    return composite_simpson_13(integrand, t_start, t_end, n)

def curvature_2d(f: Callable[[float], float], x: float) -> float:
    """Signed curvature kappa = f''(x) / (1 + f'(x)^2)^(3/2)."""
    dy = central_difference(f, x)
    d2y = second_derivative(f, x)
    return d2y / ((1.0 + dy * dy) ** 1.5)

def radius_of_curvature_2d(f: Callable[[float], float], x: float) -> float:
    """Radius of curvature R = 1 / |kappa|."""
    k = curvature_2d(f, x)
    if abs(k) < 1e-15:
        return float('inf')
    return 1.0 / abs(k)

def center_of_curvature_2d(f: Callable[[float], float], x: float) -> Tuple[float, float]:
    """Center of curvature (x_c, y_c) at x."""
    dy = central_difference(f, x)
    d2y = second_derivative(f, x)
    if abs(d2y) < 1e-15:
        raise ValueError("Zero curvature; center at infinity")
    xc = x - dy * (1.0 + dy * dy) / d2y
    yc = f(x) + (1.0 + dy * dy) / d2y
    return xc, yc

def curvature_param_2d(x_t: Callable[[float], float], y_t: Callable[[float], float], t: float) -> float:
    """Curvature kappa(t) of plane parametric curve."""
    xp = central_difference(x_t, t)
    yp = central_difference(y_t, t)
    xpp = second_derivative(x_t, t)
    ypp = second_derivative(y_t, t)
    speed_cubed = (xp * xp + yp * yp) ** 1.5
    if speed_cubed < 1e-15:
        return 0.0
    return abs(xp * ypp - yp * xpp) / speed_cubed

def surface_area_revolution_x(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Surface area of revolution about x-axis: 2*pi * integral |f(x)| * sqrt(1 + f'(x)^2) dx."""
    integrand = lambda x: 2.0 * math.pi * abs(f(x)) * math.sqrt(1.0 + central_difference(f, x) ** 2)
    return composite_simpson_13(integrand, a, b, n)

def surface_area_revolution_y(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Surface area of revolution about y-axis: 2*pi * integral |x| * sqrt(1 + f'(x)^2) dx."""
    integrand = lambda x: 2.0 * math.pi * abs(x) * math.sqrt(1.0 + central_difference(f, x) ** 2)
    return composite_simpson_13(integrand, a, b, n)

def volume_revolution_disk(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Volume of revolution about x-axis using disk method: pi * integral (f(x))^2 dx."""
    integrand = lambda x: math.pi * (f(x) ** 2)
    return composite_simpson_13(integrand, a, b, n)

def volume_revolution_shell(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Volume of revolution about y-axis using cylindrical shell method: 2*pi * integral x * f(x) dx."""
    integrand = lambda x: 2.0 * math.pi * x * abs(f(x))
    return composite_simpson_13(integrand, a, b, n)

def centroid_x(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Centroid x-coordinate of area under curve y = f(x): (1 / Area) * integral x * f(x) dx."""
    area = composite_simpson_13(f, a, b, n)
    if area == 0:
        raise ZeroDivisionError("Area is zero")
    mx = composite_simpson_13(lambda x: x * f(x), a, b, n)
    return mx / area

def centroid_y(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Centroid y-coordinate of area under curve y = f(x): (1 / (2*Area)) * integral (f(x))^2 dx."""
    area = composite_simpson_13(f, a, b, n)
    if area == 0:
        raise ZeroDivisionError("Area is zero")
    my = composite_simpson_13(lambda x: 0.5 * (f(x) ** 2), a, b, n)
    return my / area

def moment_of_inertia_x(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Second moment of area about x-axis: (1/3) * integral (f(x))^3 dx."""
    return (1.0 / 3.0) * composite_simpson_13(lambda x: f(x) ** 3, a, b, n)

def moment_of_inertia_y(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Second moment of area about y-axis: integral x^2 * f(x) dx."""
    return composite_simpson_13(lambda x: (x ** 2) * f(x), a, b, n)

def average_value(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Average value of f on [a, b]: (1 / (b - a)) * integral_a^b f(x) dx."""
    if a == b:
        return f(a)
    return composite_simpson_13(f, a, b, n) / (b - a)

def work_1d(force: Callable[[float], float], x0: float, x1: float, n: int = 1000) -> float:
    """Work done by 1D force: integral_{x0}^{x1} F(x) dx."""
    return composite_simpson_13(force, x0, x1, n)

# ----------------------------------------------------
# 6. Optimization, Critical Points, and Root Finding
# ----------------------------------------------------

def is_critical_point(f: Callable[[float], float], x: float, tol: float = 1e-5) -> bool:
    """Return True if x is a critical point (f'(x) ~ 0)."""
    return abs(central_difference(f, x)) < tol

def is_convex_at(f: Callable[[float], float], x: float) -> bool:
    """Check if function is strictly convex (f''(x) > 0) at x."""
    return second_derivative(f, x) > 0.0

def is_concave_at(f: Callable[[float], float], x: float) -> bool:
    """Check if function is strictly concave (f''(x) < 0) at x."""
    return second_derivative(f, x) < 0.0

def is_inflection_point(f: Callable[[float], float], x: float, tol: float = 1e-4) -> bool:
    """Check if x is an inflection point."""
    d2_left = second_derivative(f, x - 1e-3)
    d2_right = second_derivative(f, x + 1e-3)
    return (d2_left * d2_right < 0.0) or abs(second_derivative(f, x)) < tol

def golden_section_search(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6) -> float:
    """Find local minimum of f on [a, b] using Golden Section Search."""
    invphi = (math.sqrt(5.0) - 1.0) * 0.5
    invphi2 = (3.0 - math.sqrt(5.0)) * 0.5
    h = b - a
    if h <= tol:
        return (a + b) * 0.5
    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)
    while abs(b - a) > tol:
        if yc < yd:
            b, d, yd = d, c, yc
            h = b - a
            c = a + invphi2 * h
            yc = f(c)
        else:
            a, c, yc = c, d, yd
            h = b - a
            d = a + invphi * h
            yd = f(d)
    return (a + b) * 0.5

def brent_minimize_1d(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 100) -> float:
    """Brent's method for 1D local minimization."""
    gold = (3.0 - math.sqrt(5.0)) * 0.5
    x = w = v = a + gold * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        if abs(x - mid) <= (tol2 - 0.5 * (b - a)):
            return x
        p = q = r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0:
                p = -p
            q = abs(q)
            r = e
            e = d
        if abs(p) < abs(0.5 * q * r) and (p > q * (a - x)) and (p < q * (b - x)):
            d = p / q
            u = x + d
            if (u - a) < tol2 or (b - u) < tol2:
                d = tol1 if (mid - x >= 0) else -tol1
        else:
            e = b - x if x < mid else a - x
            d = gold * e
        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w, fv, fw = w, u, fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x

def newton_raphson_minimize_1d(f: Callable[[float], float], x0: float, tol: float = 1e-6, max_iter: int = 50) -> float:
    """Minimize 1D function using Newton's method."""
    curr = x0
    for _ in range(max_iter):
        df = central_difference(f, curr)
        d2f = second_derivative(f, curr)
        if abs(d2f) < 1e-12:
            break
        step = df / d2f
        curr -= step
        if abs(step) < tol:
            return curr
    return curr

def gradient_descent_1d(f: Callable[[float], float], x0: float, lr: float = 0.01, max_iter: int = 500, tol: float = 1e-6) -> float:
    """1D gradient descent minimization."""
    curr = x0
    for _ in range(max_iter):
        grad = central_difference(f, curr)
        step = lr * grad
        curr -= step
        if abs(step) < tol:
            break
    return curr

def gradient_descent_nd(f: Callable[[List[float]], float], x0: List[float], lr: float = 0.01,
                        max_iter: int = 500, tol: float = 1e-6) -> List[float]:
    """Multidimensional gradient descent."""
    curr = list(x0)
    for _ in range(max_iter):
        grad = numerical_gradient(f, curr)
        norm_grad = math.sqrt(sum(g * g for g in grad))
        if norm_grad < tol:
            break
        for i in range(len(curr)):
            curr[i] -= lr * grad[i]
    return curr

def root_bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Find root of f in [a, b] using bisection method."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if (b - a) * 0.5 < tol or f(mid) == 0.0:
            return mid
        if fa * f(mid) < 0:
            b = mid
        else:
            a = mid
            fa = f(mid)
    return 0.5 * (a + b)

def root_newton(f: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """Find root using Newton-Raphson."""
    curr = x0
    for _ in range(max_iter):
        val = f(curr)
        if abs(val) < tol:
            return curr
        df = central_difference(f, curr)
        if abs(df) < 1e-15:
            raise ZeroDivisionError("Derivative near zero")
        curr -= val / df
    return curr

def root_secant(f: Callable[[float], float], x0: float, x1: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """Find root using Secant method."""
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if abs(f1) < tol:
            return x1
        denom = f1 - f0
        if abs(denom) < 1e-15:
            break
        x_next = x1 - f1 * (x1 - x0) / denom
        x0, x1 = x1, x_next
        f0, f1 = f1, f(x1)
    return x1

def root_halley(f: Callable[[float], float], x0: float, tol: float = 1e-8, max_iter: int = 50) -> float:
    """Find root using Halley's method."""
    curr = x0
    for _ in range(max_iter):
        y = f(curr)
        if abs(y) < tol:
            return curr
        dy = central_difference(f, curr)
        d2y = second_derivative(f, curr)
        denom = 2.0 * dy * dy - y * d2y
        if abs(denom) < 1e-15:
            break
        curr -= (2.0 * y * dy) / denom
    return curr

def root_ridder(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = 60) -> float:
    """Find root using Ridders' method."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        denom = math.sqrt(fc * fc - fa * fb)
        if denom == 0:
            return c
        dx = (c - a) * fc / denom
        x_new = c + (-dx if fa >= fb else dx)
        fx_new = f(x_new)
        if abs(fx_new) < tol:
            return x_new
        if fc * fx_new <= 0:
            a, b, fa, fb = c, x_new, fc, fx_new
        elif fa * fx_new <= 0:
            b, fb = x_new, fx_new
        else:
            a, fa = x_new, fx_new
        if abs(b - a) < tol:
            return 0.5 * (a + b)
    return 0.5 * (a + b)

# ----------------------------------------------------
# 7. Taylor Series, Polynomials, and Approximations
# ----------------------------------------------------

def taylor_coefficient(f: Callable[[float], float], a: float, k: int) -> float:
    """Compute k-th Taylor coefficient c_k = f^(k)(a) / k!."""
    return nth_derivative(f, a, k) / math.factorial(k)

def taylor_polynomial_eval(f: Callable[[float], float], a: float, degree: int, x: float) -> float:
    """Evaluate degree-N Taylor polynomial of f centered at a at point x."""
    res = 0.0
    dx = x - a
    term = 1.0
    for k in range(degree + 1):
        coeff = taylor_coefficient(f, a, k)
        res += coeff * term
        term *= dx
    return res

def maclaurin_polynomial_eval(f: Callable[[float], float], degree: int, x: float) -> float:
    """Evaluate degree-N Maclaurin polynomial (centered at 0) of f at x."""
    return taylor_polynomial_eval(f, 0.0, degree, x)

# ----------------------------------------------------
# 8. Special Mathematical Functions and Integrals
# ----------------------------------------------------

def erf_integral(x: float) -> float:
    """Error function erf(x)."""
    if x == 0.0:
        return 0.0
    sign = 1.0 if x >= 0 else -1.0
    val = composite_simpson_13(lambda t: math.exp(-t * t), 0.0, abs(x), 500)
    return sign * (2.0 / math.sqrt(math.pi)) * val

def erfc_integral(x: float) -> float:
    """Complementary error function erfc(x) = 1 - erf(x)."""
    return 1.0 - erf_integral(x)

def fresnel_s(x: float) -> float:
    """Fresnel sine integral S(x)."""
    return composite_simpson_13(lambda t: math.sin(0.5 * math.pi * t * t), 0.0, x, 600)

def fresnel_c(x: float) -> float:
    """Fresnel cosine integral C(x)."""
    return composite_simpson_13(lambda t: math.cos(0.5 * math.pi * t * t), 0.0, x, 600)

def gamma_lanczos(z: float) -> float:
    """Gamma function Gamma(z) via Lanczos approximation."""
    if z <= 0.0 and z == math.floor(z):
        raise ValueError("Gamma function undefined at non-positive integers")
    if z < 0.5:
        return math.pi / (math.sin(math.pi * z) * gamma_lanczos(1.0 - z))
    z -= 1.0
    p = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109583111094, 9.9843695780195716e-6, 1.5056327351493116e-7
    ]
    x = p[0]
    for i in range(1, len(p)):
        x += p[i] / (z + i)
    t = z + len(p) - 1.5
    return math.sqrt(2.0 * math.pi) * (t ** (z + 0.5)) * math.exp(-t) * x

def log_gamma(z: float) -> float:
    """ln|Gamma(z)|."""
    return math.log(abs(gamma_lanczos(z)))

def beta_function(x: float, y: float) -> float:
    """Beta function B(x, y)."""
    return (gamma_lanczos(x) * gamma_lanczos(y)) / gamma_lanczos(x + y)

def digamma(x: float, h: float = 1e-6) -> float:
    """Digamma function psi(x)."""
    return central_difference(log_gamma, x, h)

def polygamma(n: int, x: float, h: float = 1e-3) -> float:
    """Polygamma function psi^(n)(x)."""
    if n < 0:
        raise ValueError("Order n must be non-negative")
    return nth_derivative(log_gamma, x, n + 1, h)

def exponential_integral_e1(x: float) -> float:
    """Exponential integral E_1(x)."""
    if x <= 0:
        raise ValueError("x must be positive for E_1")
    return improper_integral_upper_inf(lambda t: math.exp(-t) / t, x)

def sine_integral_si(x: float) -> float:
    """Sine integral Si(x)."""
    if x == 0.0:
        return 0.0
    f = lambda t: (math.sin(t) / t) if abs(t) > 1e-12 else 1.0
    return composite_simpson_13(f, 0.0, x, 600)

def cosine_integral_ci(x: float) -> float:
    """Cosine integral Ci(x)."""
    if x <= 0:
        raise ValueError("x must be positive for Ci")
    euler_mascheroni = 0.57721566490153286
    f = lambda t: ((math.cos(t) - 1.0) / t) if abs(t) > 1e-12 else 0.0
    return euler_mascheroni + math.log(x) + composite_simpson_13(f, 0.0, x, 600)

def elliptic_integral_k(m: float, n: int = 1000) -> float:
    """Complete elliptic integral of first kind K(m)."""
    if m >= 1.0:
        raise ValueError("Parameter m must be < 1")
    f = lambda theta: 1.0 / math.sqrt(1.0 - m * (math.sin(theta) ** 2))
    return composite_simpson_13(f, 0.0, math.pi * 0.5, n)

def elliptic_integral_e(m: float, n: int = 1000) -> float:
    """Complete elliptic integral of second kind E(m)."""
    if m > 1.0:
        raise ValueError("Parameter m must be <= 1")
    f = lambda theta: math.sqrt(1.0 - m * (math.sin(theta) ** 2))
    return composite_simpson_13(f, 0.0, math.pi * 0.5, n)

def legendre_polynomial_eval(n: int, x: float) -> float:
    """Evaluate Legendre polynomial P_n(x) using Bonnet's recurrence."""
    if n < 0:
        raise ValueError("Degree n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return x
    p0, p1 = 1.0, x
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p1 - k * p0) / (k + 1)
        p0, p1 = p1, p_next
    return p1

def hermite_polynomial_eval(n: int, x: float) -> float:
    """Evaluate physicists' Hermite polynomial H_n(x) via recurrence: H_{n+1} = 2x H_n - 2n H_{n-1}."""
    if n < 0:
        raise ValueError("Degree n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h0, h1 = 1.0, 2.0 * x
    for k in range(1, n):
        h_next = 2.0 * x * h1 - 2.0 * k * h0
        h0, h1 = h1, h_next
    return h1

def laguerre_polynomial_eval(n: int, x: float) -> float:
    """Evaluate Laguerre polynomial L_n(x) via recurrence."""
    if n < 0:
        raise ValueError("Degree n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 - x
    l0, l1 = 1.0, 1.0 - x
    for k in range(1, n):
        l_next = ((2 * k + 1 - x) * l1 - k * l0) / (k + 1)
        l0, l1 = l1, l_next
    return l1

def chebyshev_t_eval(n: int, x: float) -> float:
    """Evaluate Chebyshev polynomial of first kind T_n(x) via recurrence T_{n+1} = 2x T_n - T_{n-1}."""
    if n < 0:
        raise ValueError("Degree n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return x
    t0, t1 = 1.0, x
    for _ in range(1, n):
        t0, t1 = t1, 2.0 * x * t1 - t0
    return t1

def chebyshev_u_eval(n: int, x: float) -> float:
    """Evaluate Chebyshev polynomial of second kind U_n(x)."""
    if n < 0:
        raise ValueError("Degree n must be non-negative")
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    u0, u1 = 1.0, 2.0 * x
    for _ in range(1, n):
        u0, u1 = u1, 2.0 * x * u1 - u0
    return u1

def bessel_j0(x: float) -> float:
    """Bessel function of first kind of order 0: J_0(x) via integral definition (1/pi)*integral_0^pi cos(x*sin(theta)) dtheta."""
    return composite_simpson_13(lambda theta: math.cos(x * math.sin(theta)), 0.0, math.pi, 500) / math.pi

def bessel_j1(x: float) -> float:
    """Bessel function of first kind of order 1: J_1(x) via integral definition (1/pi)*integral_0^pi cos(theta - x*sin(theta)) dtheta."""
    return composite_simpson_13(lambda theta: math.cos(theta - x * math.sin(theta)), 0.0, math.pi, 500) / math.pi

def bessel_jn(n: int, x: float) -> float:
    """Bessel function of first kind of integer order n: J_n(x)."""
    return composite_simpson_13(lambda theta: math.cos(n * theta - x * math.sin(theta)), 0.0, math.pi, 600) / math.pi
