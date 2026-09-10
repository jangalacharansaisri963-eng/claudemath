"""Complex numbers module for claudemath.

Pure-Python implementation of complex arithmetic, polar and exponential representations,
trigonometric and hyperbolic complex functions, conformal mappings, Mobius transforms,
complex polynomials, and fractal generation (Mandelbrot and Julia sets).
"""

from typing import List, Tuple, Optional, Callable
import math
import cmath


# ----------------------------------------------------
# 1. Fundamental Complex Arithmetic & Representations
# ----------------------------------------------------

def complex_add(z1: complex, z2: complex) -> complex:
    """Addition of two complex numbers z1 + z2."""
    return z1 + z2


def complex_sub(z1: complex, z2: complex) -> complex:
    """Subtraction of two complex numbers z1 - z2."""
    return z1 - z2


def complex_mul(z1: complex, z2: complex) -> complex:
    """Multiplication of two complex numbers z1 * z2."""
    return z1 * z2


def complex_div(z1: complex, z2: complex) -> complex:
    """Division of two complex numbers z1 / z2."""
    if z2 == 0:
        raise ZeroDivisionError("Division by zero in complex division")
    return z1 / z2


def complex_conjugate(z: complex) -> complex:
    """Complex conjugate z* = a - b*i."""
    return complex(z.real, -z.imag)


def complex_modulus(z: complex) -> float:
    """Modulus (magnitude / absolute value) |z| = sqrt(a^2 + b^2)."""
    return math.hypot(z.real, z.imag)


def complex_modulus_squared(z: complex) -> float:
    """Square of modulus |z|^2 = a^2 + b^2."""
    return z.real * z.real + z.imag * z.imag


def complex_argument(z: complex) -> float:
    """Principal argument Arg(z) in (-pi, pi]."""
    return math.atan2(z.imag, z.real)


def to_polar(z: complex) -> Tuple[float, float]:
    """Convert Cartesian complex to polar coordinates (r, theta)."""
    return complex_modulus(z), complex_argument(z)


def from_polar(r: float, theta: float) -> complex:
    """Create complex number from polar coordinates r * e^(i * theta)."""
    if r < 0:
        raise ValueError("Polar radius r must be non-negative")
    return complex(r * math.cos(theta), r * math.sin(theta))


def complex_signum(z: complex) -> complex:
    """Signum of complex number z / |z| (or 0 if z == 0)."""
    r = complex_modulus(z)
    if r == 0:
        return complex(0.0, 0.0)
    return z / r


def complex_reciprocal(z: complex) -> complex:
    """Multiplicative inverse 1 / z."""
    if z == 0:
        raise ZeroDivisionError("Cannot invert complex zero")
    return 1.0 / z


def complex_distance(z1: complex, z2: complex) -> float:
    """Euclidean distance |z1 - z2| between two complex numbers in the Argand plane."""
    return complex_modulus(z1 - z2)


# ----------------------------------------------------
# 2. Powers, Roots, and Exponentials
# ----------------------------------------------------

def complex_pow(base: complex, exponent: complex) -> complex:
    """Complex exponentiation base^exponent."""
    return cmath.exp(exponent * cmath.log(base)) if base != 0 else complex(0.0, 0.0)


def complex_sqrt(z: complex) -> complex:
    """Principal branch square root sqrt(z)."""
    return cmath.sqrt(z)


def complex_nth_roots(z: complex, n: int) -> List[complex]:
    """Return all n complex n-th roots of z using De Moivre's theorem."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if z == 0:
        return [complex(0.0, 0.0)] * n
    r, theta = to_polar(z)
    r_root = r ** (1.0 / n)
    roots = []
    for k in range(n):
        angle = (theta + 2.0 * math.pi * k) / n
        roots.append(from_polar(r_root, angle))
    return roots


def roots_of_unity(n: int) -> List[complex]:
    """Return all n distinct n-th roots of unity e^(2*pi*i*k / n)."""
    return complex_nth_roots(complex(1.0, 0.0), n)


def primitive_roots_of_unity(n: int) -> List[complex]:
    """Return primitive n-th roots of unity (gcd(k, n) == 1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    res = []
    for k in range(n):
        if math.gcd(k, n) == 1:
            res.append(from_polar(1.0, 2.0 * math.pi * k / n))
    return res


def complex_exp(z: complex) -> complex:
    """Complex exponential e^z = e^x * (cos(y) + i*sin(y))."""
    return cmath.exp(z)


def complex_log(z: complex) -> complex:
    """Principal branch of natural complex logarithm Ln(z) = ln|z| + i*Arg(z)."""
    if z == 0:
        raise ValueError("Logarithm undefined at z = 0")
    return cmath.log(z)


def complex_log_base(z: complex, base: complex) -> complex:
    """Complex logarithm with arbitrary complex base."""
    return complex_log(z) / complex_log(base)


# ----------------------------------------------------
# 3. Trigonometric and Hyperbolic Complex Functions
# ----------------------------------------------------

def complex_sin(z: complex) -> complex:
    """Complex sine sin(z)."""
    return cmath.sin(z)


def complex_cos(z: complex) -> complex:
    """Complex cosine cos(z)."""
    return cmath.cos(z)


def complex_tan(z: complex) -> complex:
    """Complex tangent tan(z)."""
    return cmath.tan(z)


def complex_cot(z: complex) -> complex:
    """Complex cotangent cot(z) = 1 / tan(z)."""
    return 1.0 / cmath.tan(z)


def complex_sec(z: complex) -> complex:
    """Complex secant sec(z) = 1 / cos(z)."""
    return 1.0 / cmath.cos(z)


def complex_csc(z: complex) -> complex:
    """Complex cosecant csc(z) = 1 / sin(z)."""
    return 1.0 / cmath.sin(z)


def complex_asin(z: complex) -> complex:
    """Complex arcsine asin(z)."""
    return cmath.asin(z)


def complex_acos(z: complex) -> complex:
    """Complex arccosine acos(z)."""
    return cmath.acos(z)


def complex_atan(z: complex) -> complex:
    """Complex arctangent atan(z)."""
    return cmath.atan(z)


def complex_sinh(z: complex) -> complex:
    """Complex hyperbolic sine sinh(z)."""
    return cmath.sinh(z)


def complex_cosh(z: complex) -> complex:
    """Complex hyperbolic cosine cosh(z)."""
    return cmath.cosh(z)


def complex_tanh(z: complex) -> complex:
    """Complex hyperbolic tangent tanh(z)."""
    return cmath.tanh(z)


def complex_coth(z: complex) -> complex:
    """Complex hyperbolic cotangent coth(z)."""
    return 1.0 / cmath.tanh(z)


def complex_sech(z: complex) -> complex:
    """Complex hyperbolic secant sech(z)."""
    return 1.0 / cmath.cosh(z)


def complex_csch(z: complex) -> complex:
    """Complex hyperbolic cosecant csch(z)."""
    return 1.0 / cmath.sinh(z)


def complex_asinh(z: complex) -> complex:
    """Complex inverse hyperbolic sine asinh(z)."""
    return cmath.asinh(z)


def complex_acosh(z: complex) -> complex:
    """Complex inverse hyperbolic cosine acosh(z)."""
    return cmath.acosh(z)


def complex_atanh(z: complex) -> complex:
    """Complex inverse hyperbolic tangent atanh(z)."""
    return cmath.atanh(z)


# ----------------------------------------------------
# 4. Geometry and Conformal Mappings in C
# ----------------------------------------------------

def mobius_transform(z: complex, a: complex, b: complex, c: complex, d: complex) -> complex:
    """Mobius (fractional linear) transformation f(z) = (a*z + b) / (c*z + d)."""
    det = a * d - b * c
    if abs(det) < 1e-15:
        raise ValueError("Degenerate Mobius transform (ad - bc == 0)")
    denom = c * z + d
    if denom == 0:
        return complex(float('inf'), float('inf'))
    return (a * z + b) / denom


def mobius_inverse(a: complex, b: complex, c: complex, d: complex) -> Tuple[complex, complex, complex, complex]:
    """Return coefficients of inverse Mobius transform: (d, -b, -c, a)."""
    return d, -b, -c, a


def cross_ratio(z1: complex, z2: complex, z3: complex, z4: complex) -> complex:
    """Cross-ratio (z1, z2; z3, z4) = ((z1 - z3)*(z2 - z4)) / ((z2 - z3)*(z1 - z4))."""
    num = (z1 - z3) * (z2 - z4)
    den = (z2 - z3) * (z1 - z4)
    if den == 0:
        raise ZeroDivisionError("Denominator in cross ratio is zero")
    return num / den


def joukowsky_transform(z: complex) -> complex:
    """Joukowsky conformal transform f(z) = 0.5 * (z + 1/z)."""
    if z == 0:
        raise ZeroDivisionError("Joukowsky map has pole at z = 0")
    return 0.5 * (z + 1.0 / z)


def cayley_transform(z: complex) -> complex:
    """Cayley transform mapping upper half-plane to unit disk: f(z) = (z - i) / (z + i)."""
    if z == -1j:
        raise ZeroDivisionError("Cayley transform undefined at z = -i")
    return (z - 1j) / (z + 1j)


def inverse_cayley_transform(w: complex) -> complex:
    """Inverse Cayley transform mapping unit disk to upper half-plane: f(w) = -i*(w + 1) / (w - 1)."""
    if w == 1:
        raise ZeroDivisionError("Inverse Cayley map undefined at w = 1")
    return -1j * (w + 1.0) / (w - 1.0)


def stereographic_projection_to_sphere(z: complex) -> Tuple[float, float, float]:
    """Stereographic projection from complex plane to Riemann sphere (X, Y, Z)."""
    mod_sq = complex_modulus_squared(z)
    denom = 1.0 + mod_sq
    X = 2.0 * z.real / denom
    Y = 2.0 * z.imag / denom
    Z = (mod_sq - 1.0) / denom
    return X, Y, Z


def stereographic_projection_from_sphere(X: float, Y: float, Z: float) -> complex:
    """Inverse stereographic projection from Riemann sphere to complex plane."""
    if abs(1.0 - Z) < 1e-14:
        return complex(float('inf'), float('inf'))
    return complex(X / (1.0 - Z), Y / (1.0 - Z))


def chordal_distance(z1: complex, z2: complex) -> float:
    """Chordal distance on Riemann sphere between z1 and z2."""
    p1 = stereographic_projection_to_sphere(z1)
    p2 = stereographic_projection_to_sphere(z2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


# ----------------------------------------------------
# 5. Complex Polynomials and Root Solvers
# ----------------------------------------------------

def complex_poly_eval(coeffs: List[complex], z: complex) -> complex:
    """Evaluate polynomial sum(coeffs[i] * z^i) for complex z via Horner's rule."""
    if not coeffs:
        return complex(0.0, 0.0)
    res = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        res = res * z + c
    return res


def complex_poly_derivative(coeffs: List[complex]) -> List[complex]:
    """Formal derivative of complex polynomial."""
    if len(coeffs) <= 1:
        return [complex(0.0, 0.0)]
    return [coeffs[i] * i for i in range(1, len(coeffs))]


def durand_kerner_roots(coeffs: List[complex], max_iter: int = 100, tol: float = 1e-10) -> List[complex]:
    """Find all complex roots of polynomial simultaneously via Weierstrass/Durand-Kerner method."""
    deg = len(coeffs) - 1
    if deg <= 0:
        return []
    lead = coeffs[-1]
    monic = [c / lead for c in coeffs]
    # Initialize initial guesses evenly spaced around circle
    r0 = 0.4 + 0.9j
    roots = [r0 ** i for i in range(deg)]
    for _ in range(max_iter):
        max_diff = 0.0
        new_roots = list(roots)
        for i in range(deg):
            p_val = complex_poly_eval(monic, roots[i])
            denom = 1.0
            for j in range(deg):
                if i != j:
                    denom *= (roots[i] - roots[j])
            if abs(denom) > 1e-15:
                delta = p_val / denom
                new_roots[i] = roots[i] - delta
                max_diff = max(max_diff, abs(delta))
        roots = new_roots
        if max_diff < tol:
            break
    return roots


# ----------------------------------------------------
# 6. Fractals and Dynamics in the Complex Plane
# ----------------------------------------------------

def mandelbrot_escape_time(c: complex, max_iter: int = 100) -> int:
    """Return number of iterations before |z| > 2 for z_{n+1} = z_n^2 + c."""
    z = complex(0.0, 0.0)
    for i in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return i
        z = z * z + c
    return max_iter


def julia_escape_time(z: complex, c: complex, max_iter: int = 100) -> int:
    """Return number of iterations before |z| > 2 for Julia set with parameter c."""
    for i in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return i
        z = z * z + c
    return max_iter


def burning_ship_escape_time(c: complex, max_iter: int = 100) -> int:
    """Return escape time for Burning Ship fractal z_{n+1} = (|Re(z)| + i|Im(z)|)^2 + c."""
    z = complex(0.0, 0.0)
    for i in range(max_iter):
        if z.real * z.real + z.imag * z.imag > 4.0:
            return i
        z = complex(abs(z.real), abs(z.imag)) ** 2 + c
    return max_iter


def complex_cauchy_integral_approx(f: Callable[[complex], complex],
                                  center: complex,
                                  radius: float,
                                  steps: int = 100) -> complex:
    """Approximate contour integral oint f(z) dz along circle |z - center| = radius."""
    total = complex(0.0, 0.0)
    dt = 2.0 * math.pi / steps
    for k in range(steps):
        t = k * dt
        z = center + radius * complex(math.cos(t), math.sin(t))
        dz = radius * complex(-math.sin(t), math.cos(t)) * dt
        total += f(z) * dz
    return total
def complex_midpoint(z1: complex, z2: complex) -> complex:
    """Midpoint between two complex numbers: (z1 + z2) / 2."""
    return (z1 + z2) * 0.5


def complex_dot_product(z1: complex, z2: complex) -> float:
    """Real dot product Re(z1 * conj(z2)) = x1*x2 + y1*y2."""
    return z1.real * z2.real + z1.imag * z2.imag


def complex_cross_product(z1: complex, z2: complex) -> float:
    """2D cross product Im(conj(z1) * z2) = x1*y2 - y1*x2."""
    return z1.real * z2.imag - z1.imag * z2.real


def complex_parallel(z1: complex, z2: complex, tol: float = 1e-9) -> bool:
    """Check if complex numbers represent parallel vectors."""
    return abs(complex_cross_product(z1, z2)) < tol


def complex_perpendicular(z1: complex, z2: complex, tol: float = 1e-9) -> bool:
    """Check if complex numbers represent perpendicular vectors."""
    return abs(complex_dot_product(z1, z2)) < tol


def complex_rotate(z: complex, angle_radians: float) -> complex:
    """Rotate complex number by angle in radians."""
    rot = complex(math.cos(angle_radians), math.sin(angle_radians))
    return z * rot


def complex_scale(z: complex, factor: float) -> complex:
    """Scale complex number by real factor."""
    return z * factor


def complex_unit_circle_project(z: complex) -> complex:
    """Project non-zero complex number onto unit circle."""
    mod = complex_modulus(z)
    if mod == 0:
        raise ValueError("Cannot project zero onto unit circle")
    return z / mod


def complex_gamma_stirling(z: complex) -> complex:
    """Stirling approximation for complex Gamma function Gamma(z)."""
    if z.real <= 0 and z.imag == 0 and z.real == int(z.real):
        raise ValueError("Gamma undefined at non-positive integers")
    # Reflection formula for Re(z) < 0.5: Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
    if z.real < 0.5:
        return math.pi / (cmath.sin(math.pi * z) * complex_gamma_stirling(1.0 - z))
    z_sub = z - 1.0
    return cmath.sqrt(2.0 * math.pi * z_sub) * ((z_sub / math.e) ** z_sub)


def cauchy_riemann_check(u: Callable[[float, float], float],
                         v: Callable[[float, float], float],
                         x: float, y: float, h: float = 1e-6) -> Tuple[bool, float]:
    """Verify Cauchy-Riemann equations du/dx = dv/dy and du/dy = -dv/dx numerically."""
    du_dx = (u(x + h, y) - u(x - h, y)) / (2.0 * h)
    du_dy = (u(x, y + h) - u(x, y - h)) / (2.0 * h)
    dv_dx = (v(x + h, y) - v(x - h, y)) / (2.0 * h)
    dv_dy = (v(x, y + h) - v(x, y - h)) / (2.0 * h)
    err1 = abs(du_dx - dv_dy)
    err2 = abs(du_dy + dv_dx)
    max_err = max(err1, err2)
    return max_err < 1e-4, max_err
