"""Numerical analysis module for claudemath.

Pure-Python implementation of root-finding algorithms (Brent, Halley, Muller, Newton),
numerical optimization (Nelder-Mead, Golden Section, BFGS), interpolation (Splines,
Chebyshev, Barycentric, Hermite), numerical differentiation, adaptive quadrature,
and iterative linear system solvers (Jacobi, Gauss-Seidel, SOR, Conjugate Gradient).
"""

from typing import List, Tuple, Callable, Optional, Dict, Any
import math


# ----------------------------------------------------
# 1. 1D Root Finding Algorithms
# ----------------------------------------------------

def bisection_method(f: Callable[[float], float], a: float, b: float,
                     tol: float = 1e-10, max_iter: int = 100) -> float:
    """Bisection method for bracketed root f(x) = 0."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) * 0.5 < tol:
            return m
        if fa * fm < 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return 0.5 * (a + b)


def regula_falsi(f: Callable[[float], float], a: float, b: float,
                 tol: float = 1e-10, max_iter: int = 100) -> float:
    """False position (Regula Falsi) root finding."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        s = (a * fb - b * fa) / (fb - fa)
        fs = f(s)
        if abs(fs) < tol:
            return s
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
    return s


def newton_raphson(f: Callable[[float], float], df: Callable[[float], float],
                   x0: float, tol: float = 1e-10, max_iter: int = 100) -> float:
    """Newton-Raphson iteration: x_{k+1} = x_k - f(x_k)/f'(x_k)."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Derivative near zero in Newton-Raphson")
        x_next = x - fx / dfx
        if abs(x_next - x) < tol or abs(fx) < tol:
            return x_next
        x = x_next
    return x


def secant_method(f: Callable[[float], float], x0: float, x1: float,
                  tol: float = 1e-10, max_iter: int = 100) -> float:
    """Secant method (derivative-free Newton)."""
    f0 = f(x0)
    f1 = f(x1)
    for _ in range(max_iter):
        if abs(f1 - f0) < 1e-14:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol or abs(f(x2)) < tol:
            return x2
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    return x1


def halley_method(f: Callable[[float], float], df: Callable[[float], float],
                  d2f: Callable[[float], float], x0: float,
                  tol: float = 1e-10, max_iter: int = 100) -> float:
    """Halley's method (cubic convergence using second derivative)."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        denom = 2.0 * dfx * dfx - fx * d2fx
        if abs(denom) < 1e-14:
            break
        x_next = x - (2.0 * fx * dfx) / denom
        if abs(x_next - x) < tol or abs(fx) < tol:
            return x_next
        x = x_next
    return x


def steffensen_method(f: Callable[[float], float], x0: float,
                      tol: float = 1e-10, max_iter: int = 100) -> float:
    """Steffensen's method: quadratic convergence without derivatives."""
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        gx = (f(x + fx) - fx) / fx if fx != 0 else 0
        if abs(gx) < 1e-14:
            break
        x_next = x - fx / gx
        if abs(x_next - x) < tol or abs(fx) < tol:
            return x_next
        x = x_next
    return x


def fixed_point_iteration(g: Callable[[float], float], x0: float,
                          tol: float = 1e-10, max_iter: int = 100) -> float:
    """Fixed-point iteration x_{k+1} = g(x_k)."""
    x = x0
    for _ in range(max_iter):
        x_next = g(x)
        if abs(x_next - x) < tol:
            return x_next
        x = x_next
    return x


def brent_root(f: Callable[[float], float], a: float, b: float,
               tol: float = 1e-10, max_iter: int = 100) -> float:
    """Brent-Dekker root finding algorithm combining bisection, secant, and inverse quadratic interpolation."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Root must be bracketed")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    mflag = True
    d = 0.0

    for _ in range(max_iter):
        if abs(fb) < tol:
            return b
        if abs(fa - fc) > 1e-14 and abs(fb - fc) > 1e-14:
            # Inverse quadratic interpolation
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +                 (b * fa * fc) / ((fb - fa) * (fb - fc)) +                 (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant rule
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3.0 * a + b) / 4.0 <= s <= b or b <= s <= (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) * 0.5
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) * 0.5
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c = b
        fc = fb
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        if abs(b - a) < tol:
            return b
    return b


# ----------------------------------------------------
# 2. 1D and Multidimensional Optimization
# ----------------------------------------------------

def golden_section_search(f: Callable[[float], float], a: float, b: float,
                          tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, float]:
    """Golden-section search for scalar minimum in bracket [a, b]. Returns (x_min, f_min)."""
    invphi = (math.sqrt(5.0) - 1.0) * 0.5
    invphi2 = (3.0 - math.sqrt(5.0)) * 0.5
    h = b - a
    if h <= tol:
        return 0.5 * (a + b), f(0.5 * (a + b))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c)
    yd = f(d)

    for _ in range(max_iter):
        if h < tol:
            break
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d)

    x_min = 0.5 * (a + b)
    return x_min, f(x_min)


def gradient_descent_1d(f: Callable[[float], float], df: Callable[[float], float],
                        x0: float, lr: float = 0.01, tol: float = 1e-8,
                        max_iter: int = 500) -> float:
    """Gradient descent for 1D function."""
    x = x0
    for _ in range(max_iter):
        grad = df(x)
        if abs(grad) < tol:
            break
        x -= lr * grad
    return x


def newton_optimization_1d(df: Callable[[float], float], d2f: Callable[[float], float],
                           x0: float, tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton's method for finding local extremum (zero of df)."""
    return newton_raphson(df, d2f, x0, tol=tol, max_iter=max_iter)


# ----------------------------------------------------
# 3. Interpolation and Polynomial Approximation
# ----------------------------------------------------

def lagrange_interpolate(xs: List[float], ys: List[float], x: float) -> float:
    """Lagrange polynomial interpolation at point x."""
    n = len(xs)
    total = 0.0
    for i in range(n):
        term = ys[i]
        for j in range(n):
            if i != j:
                term *= (x - xs[j]) / (xs[i] - xs[j])
        total += term
    return total


def newton_divided_differences(xs: List[float], ys: List[float]) -> List[float]:
    """Compute coefficients of Newton divided difference form polynomial."""
    n = len(xs)
    coef = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])
    return coef


def newton_polynomial_eval(coef: List[float], xs: List[float], x: float) -> float:
    """Evaluate Newton form polynomial using Horner-like scheme."""
    n = len(coef)
    res = coef[-1]
    for i in range(n - 2, -1, -1):
        res = res * (x - xs[i]) + coef[i]
    return res


def neville_interpolation(xs: List[float], ys: List[float], x: float) -> float:
    """Neville's algorithm for evaluating interpolation polynomial at x."""
    n = len(xs)
    p = list(ys)
    for m in range(1, n):
        for i in range(n - m):
            p[i] = ((x - xs[i + m]) * p[i] + (xs[i] - x) * p[i + 1]) / (xs[i] - xs[i + m])
    return p[0]


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> List[float]:
    """Compute n Chebyshev nodes in interval [a, b] to minimize Runge phenomenon."""
    nodes = []
    for k in range(1, n + 1):
        # Roots of T_n(x): cos((2k - 1)*pi / (2n))
        root = math.cos((2.0 * k - 1.0) * math.pi / (2.0 * n))
        # Map from [-1, 1] to [a, b]
        mapped = 0.5 * (a + b) + 0.5 * (b - a) * root
        nodes.append(mapped)
    return nodes


def barycentric_weights(xs: List[float]) -> List[float]:
    """Compute barycentric weights for stable polynomial interpolation."""
    n = len(xs)
    weights = [1.0] * n
    for j in range(n):
        w = 1.0
        for k in range(n):
            if k != j:
                w *= (xs[j] - xs[k])
        weights[j] = 1.0 / w
    return weights


def barycentric_interpolate(xs: List[float], ys: List[float], weights: List[float], x: float) -> float:
    """Barycentric formula for fast O(n) interpolation."""
    for i, xi in enumerate(xs):
        if abs(x - xi) < 1e-14:
            return ys[i]
    num = 0.0
    den = 0.0
    for xi, yi, wi in zip(xs, ys, weights):
        term = wi / (x - xi)
        num += term * yi
        den += term
    return num / den


# ----------------------------------------------------
# 4. Numerical Differentiation and Richardson Extrapolation
# ----------------------------------------------------

def forward_difference_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """First-order forward difference: (f(x + h) - f(x)) / h."""
    return (f(x + h) - f(x)) / h


def backward_difference_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """First-order backward difference: (f(x) - f(x - h)) / h."""
    return (f(x) - f(x - h)) / h


def central_difference_derivative(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """Second-order central difference: (f(x + h) - f(x - h)) / (2h)."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def second_derivative_central(f: Callable[[float], float], x: float, h: float = 1e-4) -> float:
    """Second derivative central difference: (f(x+h) - 2f(x) + f(x-h)) / h^2."""
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)


def richardson_extrapolation_derivative(f: Callable[[float], float], x: float, h: float = 0.05) -> float:
    """Richardson extrapolation on central differences yielding O(h^4) accuracy."""
    d1 = central_difference_derivative(f, x, h)
    d2 = central_difference_derivative(f, x, h * 0.5)
    return (4.0 * d2 - d1) / 3.0


# ----------------------------------------------------
# 5. Numerical Quadrature (Integration)
# ----------------------------------------------------

def trapezoidal_rule(f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
    """Composite trapezoidal rule."""
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        total += f(a + i * h)
    return total * h


def simpson_rule(f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
    """Composite Simpson's 1/3 rule (n must be even)."""
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 == 1 else 2.0) * f(x)
    return total * (h / 3.0)


def simpson_38_rule(f: Callable[[float], float], a: float, b: float, n: int = 99) -> float:
    """Composite Simpson's 3/8 rule (n multiple of 3)."""
    if n % 3 != 0:
        n += (3 - (n % 3))
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        weight = 2.0 if i % 3 == 0 else 3.0
        total += weight * f(a + i * h)
    return total * (3.0 * h / 8.0)


def boole_rule(f: Callable[[float], float], a: float, b: float) -> float:
    """Boole's 5-point quadrature rule."""
    h = (b - a) / 4.0
    return (2.0 * h / 45.0) * (7.0 * f(a) + 32.0 * f(a + h) + 12.0 * f(a + 2.0 * h) + 32.0 * f(a + 3.0 * h) + 7.0 * f(b))


def romberg_integration(f: Callable[[float], float], a: float, b: float, max_steps: int = 6) -> float:
    """Romberg integration applying Richardson extrapolation to trapezoidal approximations."""
    R = [[0.0] * (max_steps + 1) for _ in range(max_steps + 1)]
    h = b - a
    R[0][0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_steps + 1):
        h *= 0.5
        sum_f = sum(f(a + (2 * k - 1) * h) for k in range(1, (1 << (i - 1)) + 1))
        R[i][0] = 0.5 * R[i - 1][0] + h * sum_f
        for j in range(1, i + 1):
            factor = 4.0 ** j
            R[i][j] = (factor * R[i][j - 1] - R[i - 1][j - 1]) / (factor - 1.0)
    return R[max_steps][max_steps]


def gauss_legendre_quadrature_2(f: Callable[[float], float], a: float, b: float) -> float:
    """2-point Gauss-Legendre quadrature."""
    c = 1.0 / math.sqrt(3.0)
    x1 = 0.5 * ((b - a) * (-c) + (a + b))
    x2 = 0.5 * ((b - a) * c + (a + b))
    return 0.5 * (b - a) * (f(x1) + f(x2))


def gauss_legendre_quadrature_3(f: Callable[[float], float], a: float, b: float) -> float:
    """3-point Gauss-Legendre quadrature."""
    c = math.sqrt(3.0 / 5.0)
    x1 = 0.5 * ((b - a) * (-c) + (a + b))
    x2 = 0.5 * (a + b)
    x3 = 0.5 * ((b - a) * c + (a + b))
    w1, w2, w3 = 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0
    return 0.5 * (b - a) * (w1 * f(x1) + w2 * f(x2) + w3 * f(x3))


# ----------------------------------------------------
# 6. Iterative Linear Solvers
# ----------------------------------------------------

def jacobi_iteration_solve(A: List[List[float]], b: List[float],
                           x0: Optional[List[float]] = None,
                           tol: float = 1e-9, max_iter: int = 500) -> List[float]:
    """Jacobi iteration for diagonally dominant system A * x = b."""
    n = len(b)
    x = list(x0) if x0 is not None else [0.0] * n
    for _ in range(max_iter):
        x_new = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            if abs(A[i][i]) < 1e-14:
                raise ZeroDivisionError("Zero on diagonal in Jacobi")
            x_new[i] = (b[i] - s) / A[i][i]
        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new
        if diff < tol:
            break
    return x


def gauss_seidel_iteration_solve(A: List[List[float]], b: List[float],
                                 x0: Optional[List[float]] = None,
                                 tol: float = 1e-9, max_iter: int = 500) -> List[float]:
    """Gauss-Seidel iterative linear solver."""
    n = len(b)
    x = list(x0) if x0 is not None else [0.0] * n
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new = (b[i] - s) / A[i][i]
            max_diff = max(max_diff, abs(x_new - x[i]))
            x[i] = x_new
        if max_diff < tol:
            break
    return x


def successive_over_relaxation_solve(A: List[List[float]], b: List[float], omega: float,
                                     x0: Optional[List[float]] = None,
                                     tol: float = 1e-9, max_iter: int = 500) -> List[float]:
    """Successive Over-Relaxation (SOR) solver with relaxation parameter omega in (0, 2)."""
    n = len(b)
    x = list(x0) if x0 is not None else [0.0] * n
    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_gs = (b[i] - s) / A[i][i]
            x_new = (1.0 - omega) * x[i] + omega * x_gs
            max_diff = max(max_diff, abs(x_new - x[i]))
            x[i] = x_new
        if max_diff < tol:
            break
    return x


def conjugate_gradient_solve(A: List[List[float]], b: List[float],
                             x0: Optional[List[float]] = None,
                             tol: float = 1e-9, max_iter: Optional[int] = None) -> List[float]:
    """Conjugate Gradient method for symmetric positive-definite linear system A * x = b."""
    n = len(b)
    if max_iter is None:
        max_iter = 2 * n
    x = list(x0) if x0 is not None else [0.0] * n
    Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    r = [b[i] - Ax[i] for i in range(n)]
    p = list(r)
    rsold = sum(ri * ri for ri in r)

    for _ in range(max_iter):
        if math.sqrt(rsold) < tol:
            break
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        pAp = sum(p[i] * Ap[i] for i in range(n))
        if abs(pAp) < 1e-15:
            break
        alpha = rsold / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rsnew = sum(ri * ri for ri in r)
        if math.sqrt(rsnew) < tol:
            break
        beta = rsnew / rsold
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        rsold = rsnew
    return x
def muller_root(f: Callable[[float], float], x0: float, x1: float, x2: float,
                tol: float = 1e-10, max_iter: int = 100) -> float:
    """Muller's method for root finding using quadratic parabola through 3 points."""
    for _ in range(max_iter):
        f0, f1, f2 = f(x0), f(x1), f(x2)
        h1 = x1 - x0
        h2 = x2 - x1
        if abs(h1) < 1e-14 or abs(h2) < 1e-14:
            break
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        a = (d2 - d1) / (h2 + h1)
        b = a * h2 + d2
        c = f2
        disc = b * b - 4.0 * a * c
        rad = math.sqrt(max(0.0, disc))
        den1 = b + rad
        den2 = b - rad
        den = den1 if abs(den1) > abs(den2) else den2
        if abs(den) < 1e-14:
            break
        dx = -2.0 * c / den
        x3 = x2 + dx
        if abs(dx) < tol or abs(f(x3)) < tol:
            return x3
        x0, x1, x2 = x1, x2, x3
    return x2


def ridder_root(f: Callable[[float], float], a: float, b: float,
                tol: float = 1e-10, max_iter: int = 100) -> float:
    """Ridders' root finding method using exponential weighting."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Root must be bracketed")
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fmid = f(mid)
        s = math.sqrt(fmid * fmid - fa * fb)
        if s == 0:
            return mid
        sign = 1.0 if fa - fb > 0 else -1.0
        x_new = mid + (mid - a) * sign * fmid / s
        fnew = f(x_new)
        if abs(fnew) < tol or abs(b - a) < tol:
            return x_new
        if fmid * fnew < 0:
            a, b = mid, x_new
            fa, fb = fmid, fnew
        elif fa * fnew < 0:
            b = x_new
            fb = fnew
        else:
            a = x_new
            fa = fnew
    return 0.5 * (a + b)


def fibonacci_search_1d(f: Callable[[float], float], a: float, b: float, n: int = 25) -> Tuple[float, float]:
    """1D optimization via Fibonacci search."""
    fib = [1, 1]
    for _ in range(n + 2):
        fib.append(fib[-1] + fib[-2])
    k = n
    c = a + (fib[k - 1] / fib[k + 1]) * (b - a)
    d = a + (fib[k] / fib[k + 1]) * (b - a)
    fc = f(c)
    fd = f(d)
    for i in range(1, n):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + (fib[n - i] / fib[n - i + 2]) * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + (fib[n - i + 1] / fib[n - i + 2]) * (b - a)
            fd = f(d)
    x_opt = 0.5 * (a + b)
    return x_opt, f(x_opt)


def hermite_interpolate(x0: float, y0: float, dy0: float,
                        x1: float, y1: float, dy1: float, x: float) -> float:
    """Cubic Hermite interpolation polynomial matching endpoints and derivatives."""
    h = x1 - x0
    t = (x - x0) / h
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return h00 * y0 + h10 * h * dy0 + h01 * y1 + h11 * h * dy1


def piecewise_linear_interpolate(xs: List[float], ys: List[float], x: float) -> float:
    """Piecewise linear interpolation at query point x."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # Binary search
    lo, hi = 0, len(xs) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return (1.0 - t) * ys[lo] + t * ys[hi]


def bilinear_interpolate(q11: float, q12: float, q21: float, q22: float,
                         x1: float, x2: float, y1: float, y2: float,
                         x: float, y: float) -> float:
    """Bilinear 2D interpolation on rectangle [x1, x2] x [y1, y2]."""
    tx = (x - x1) / (x2 - x1)
    ty = (y - y1) / (y2 - y1)
    r1 = (1.0 - tx) * q11 + tx * q21
    r2 = (1.0 - tx) * q12 + tx * q22
    return (1.0 - ty) * r1 + ty * r2


def adaptive_simpson_quadrature(f: Callable[[float], float], a: float, b: float,
                                tol: float = 1e-8, max_depth: int = 30) -> float:
    """Adaptive Simpson quadrature with error control."""
    def simpson_step(f, a, b):
        c = 0.5 * (a + b)
        return (b - a) / 6.0 * (f(a) + 4.0 * f(c) + f(b))

    def recursive_asr(f, a, b, tol, whole, depth):
        c = 0.5 * (a + b)
        left = simpson_step(f, a, c)
        right = simpson_step(f, c, b)
        if depth <= 0 or abs(left + right - whole) <= 15.0 * tol:
            return left + right + (left + right - whole) / 15.0
        return (recursive_asr(f, a, c, tol * 0.5, left, depth - 1) +
                recursive_asr(f, c, b, tol * 0.5, right, depth - 1))

    whole = simpson_step(f, a, b)
    return recursive_asr(f, a, b, tol, whole, max_depth)


def monte_carlo_integrate_1d(f: Callable[[float], float], a: float, b: float,
                             num_samples: int = 10000) -> float:
    """Monte Carlo numerical integration with deterministic pseudo-random sequence."""
    total = 0.0
    # Linear congruential generator
    state = 123456789
    for _ in range(num_samples):
        state = (1103515245 * state + 12345) & 0x7fffffff
        u = state / 2147483648.0
        x = a + (b - a) * u
        total += f(x)
    return (b - a) * (total / num_samples)


def midpoint_composite_rule(f: Callable[[float], float], a: float, b: float, n: int = 100) -> float:
    """Composite midpoint numerical integration."""
    h = (b - a) / n
    total = sum(f(a + (i + 0.5) * h) for i in range(n))
    return total * h


def five_point_stencil_derivative(f: Callable[[float], float], x: float, h: float = 1e-3) -> float:
    """Five-point central stencil for first derivative O(h^4)."""
    return (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h)


def five_point_stencil_second_derivative(f: Callable[[float], float], x: float, h: float = 1e-3) -> float:
    """Five-point central stencil for second derivative O(h^4)."""
    return (-f(x + 2.0 * h) + 16.0 * f(x + h) - 30.0 * f(x) + 16.0 * f(x - h) - f(x - 2.0 * h)) / (12.0 * h * h)


def laplacian_2d_finite_difference(f: Callable[[float, float], float],
                                   x: float, y: float, h: float = 1e-3) -> float:
    """Standard 5-point finite difference discrete Laplacian: d^2f/dx^2 + d^2f/dy^2."""
    center = f(x, y)
    neighbors = f(x + h, y) + f(x - h, y) + f(x, y + h) + f(x, y - h)
    return (neighbors - 4.0 * center) / (h * h)


def power_iteration_dominant_eigenvalue(A: List[List[float]], max_iter: int = 100,
                                       tol: float = 1e-9) -> Tuple[float, List[float]]:
    """Power iteration to find dominant eigenvalue and associated eigenvector."""
    n = len(A)
    v = [1.0 / math.sqrt(n)] * n
    lam = 0.0
    for _ in range(max_iter):
        Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in Av))
        if norm < 1e-15:
            break
        v_next = [x / norm for x in Av]
        # Rayleigh quotient for eigenvalue
        Av_next = [sum(A[i][j] * v_next[j] for j in range(n)) for i in range(n)]
        lam_next = sum(v_next[i] * Av_next[i] for i in range(n))
        if abs(lam_next - lam) < tol:
            return lam_next, v_next
        lam = lam_next
        v = v_next
    return lam, v


def tridiagonal_thomas_algorithm(a: List[float], b: List[float], c: List[float], d: List[float]) -> List[float]:
    """Thomas algorithm (TDMA) for tridiagonal system. a: sub, b: main, c: super, d: RHS."""
    n = len(d)
    c_prime = [0.0] * n
    d_prime = [0.0] * n
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i - 1] * c_prime[i - 1]
        c_prime[i] = (c[i] / denom) if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i - 1] * d_prime[i - 1]) / denom
    x = [0.0] * n
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def cubic_spline_second_derivatives(xs: List[float], ys: List[float]) -> List[float]:
    """Compute natural cubic spline second derivatives via tridiagonal solve."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    # Set up system for interior nodes
    sub = [h[i] for i in range(1, n - 2)]
    main = [2.0 * (h[i - 1] + h[i]) for i in range(1, n - 1)]
    super_diag = [h[i] for i in range(1, n - 2)]
    rhs = [6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]) for i in range(1, n - 1)]
    if n == 2:
        return [0.0, 0.0]
    M_interior = tridiagonal_thomas_algorithm(sub, main, super_diag, rhs)
    return [0.0] + M_interior + [0.0]


def cubic_spline_evaluate(xs: List[float], ys: List[float], M: List[float], x: float) -> float:
    """Evaluate cubic spline at query x given second derivatives M."""
    n = len(xs)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    # Locate interval
    i = 0
    while i < n - 2 and xs[i + 1] < x:
        i += 1
    h = xs[i + 1] - xs[i]
    dx1 = xs[i + 1] - x
    dx0 = x - xs[i]
    val = (M[i] * (dx1 ** 3) + M[i + 1] * (dx0 ** 3)) / (6.0 * h) +           (ys[i] - M[i] * h * h / 6.0) * (dx1 / h) +           (ys[i + 1] - M[i + 1] * h * h / 6.0) * (dx0 / h)
    return val
def rayleigh_quotient(A: List[List[float]], v: List[float]) -> float:
    """Rayleigh quotient R(A, v) = (v^T A v) / (v^T v)."""
    n = len(v)
    Av = [sum(A[i][j] * v[j] for j in range(n)) for i in range(n)]
    vAv = sum(v[i] * Av[i] for i in range(n))
    vv = sum(vi * vi for vi in v)
    if vv == 0:
        raise ValueError("Vector must be non-zero")
    return vAv / vv


def aitken_delta_squared_acceleration(seq: List[float]) -> List[float]:
    """Aitken's Delta^2 process to accelerate convergence of a sequence."""
    accelerated = []
    for i in range(len(seq) - 2):
        x0, x1, x2 = seq[i], seq[i + 1], seq[i + 2]
        d1 = x1 - x0
        d2 = x2 - 2.0 * x1 + x0
        if abs(d2) > 1e-15:
            accelerated.append(x0 - (d1 * d1) / d2)
        else:
            accelerated.append(x1)
    return accelerated


def numerical_gradient_vector(f: Callable[[List[float]], float], x: List[float], h: float = 1e-5) -> List[float]:
    """Compute numerical gradient of multivariable scalar function using central differences."""
    n = len(x)
    grad = [0.0] * n
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * h)
    return grad


def numerical_jacobian_matrix(f: Callable[[List[float]], List[float]],
                              x: List[float], h: float = 1e-5) -> List[List[float]]:
    """Compute m x n Jacobian matrix of vector-valued function."""
    n = len(x)
    f0 = f(x)
    m = len(f0)
    J = [[0.0] * n for _ in range(m)]
    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += h
        x_minus[j] -= h
        f_plus = f(x_plus)
        f_minus = f(x_minus)
        for i in range(m):
            J[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * h)
    return J


def numerical_hessian_matrix(f: Callable[[List[float]], float],
                             x: List[float], h: float = 1e-4) -> List[List[float]]:
    """Compute n x n numerical Hessian matrix."""
    n = len(x)
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                xp = list(x)
                xm = list(x)
                xp[i] += h
                xm[i] -= h
                H[i][i] = (f(xp) - 2.0 * f(x) + f(xm)) / (h * h)
            else:
                xpp, xpm, xmp, xmm = list(x), list(x), list(x), list(x)
                xpp[i] += h; xpp[j] += h
                xpm[i] += h; xpm[j] -= h
                xmp[i] -= h; xmp[j] += h
                xmm[i] -= h; xmm[j] -= h
                val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h)
                H[i][j] = val
                H[j][i] = val
    return H


def rosenbrock_function_2d(x: float, y: float, a: float = 1.0, b: float = 100.0) -> float:
    """Rosenbrock benchmark optimization function f(x, y) = (a - x)^2 + b*(y - x^2)^2."""
    return (a - x) ** 2 + b * ((y - x * x) ** 2)


def rosenbrock_gradient_2d(x: float, y: float, a: float = 1.0, b: float = 100.0) -> Tuple[float, float]:
    """Analytical gradient of 2D Rosenbrock function."""
    df_dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x)
    df_dy = 2.0 * b * (y - x * x)
    return df_dx, df_dy


def himmelblau_function_2d(x: float, y: float) -> float:
    """Himmelblau benchmark function (x^2 + y - 11)^2 + (x + y^2 - 7)^2."""
    return (x * x + y - 11.0) ** 2 + (x + y * y - 7.0) ** 2


def ackley_function_2d(x: float, y: float) -> float:
    """Ackley benchmark optimization function in 2D."""
    term1 = -20.0 * math.exp(-0.2 * math.sqrt(0.5 * (x * x + y * y)))
    term2 = -math.exp(0.5 * (math.cos(2.0 * math.pi * x) + math.cos(2.0 * math.pi * y)))
    return term1 + term2 + math.e + 20.0


def rastrigin_function_2d(x: float, y: float, A: float = 10.0) -> float:
    """Rastrigin non-convex benchmark function."""
    return 2.0 * A + (x * x - A * math.cos(2.0 * math.pi * x)) + (y * y - A * math.cos(2.0 * math.pi * y))


def brent_minimize_1d(f: Callable[[float], float], ax: float, bx: float, cx: float,
                      tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, float]:
    """Brent's 1D minimization using parabolic interpolation and golden section."""
    a = min(ax, cx)
    b = max(ax, cx)
    x = w = v = bx
    fw = fv = fx = f(x)
    cgold = 0.3819660
    d = e = 0.0

    for _ in range(max_iter):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d
            if abs(p) >= abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x):
                e = (a - x) if x >= xm else (b - x)
                d = cgold * e
            else:
                d = p / q
                u = x + d
                if (u - a) < tol2 or (b - u) < tol2:
                    d = tol1 if (xm - x) >= 0 else -tol1
        else:
            e = (a - x) if x >= xm else (b - x)
            d = cgold * e
        u = (x + d) if abs(d) >= tol1 else (x + (tol1 if d > 0 else -tol1))
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
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu
    return x, fx


def qr_algorithm_eigenvalues_symmetric(A: List[List[float]], max_iter: int = 60,
                                       tol: float = 1e-8) -> List[float]:
    """Compute eigenvalues of symmetric matrix using QR algorithm."""
    n = len(A)
    Ak = [row[:] for row in A]
    for _ in range(max_iter):
        # Gram-Schmidt QR decomposition of Ak
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]
        for j in range(n):
            v = [Ak[i][j] for i in range(n)]
            for i in range(j):
                q_i = [Q[k][i] for k in range(n)]
                R[i][j] = sum(q_i[k] * Ak[k][j] for k in range(n))
                for k in range(n):
                    v[k] -= R[i][j] * q_i[k]
            norm_v = math.sqrt(sum(x * x for x in v))
            R[j][j] = norm_v
            if norm_v > 1e-14:
                for k in range(n):
                    Q[k][j] = v[k] / norm_v
        # Next Ak = R * Q
        Ak_next = [[sum(R[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        # Check off-diagonal convergence
        off_diag = sum(abs(Ak_next[i][j]) for i in range(n) for j in range(n) if i != j)
        Ak = Ak_next
        if off_diag < tol:
            break
    return [Ak[i][i] for i in range(n)]


def finite_difference_first_order_grid(ys: List[float], dx: float) -> List[float]:
    """Compute first-order numerical derivative across uniform 1D data grid."""
    n = len(ys)
    dy = [0.0] * n
    if n < 2:
        return dy
    dy[0] = (ys[1] - ys[0]) / dx
    for i in range(1, n - 1):
        dy[i] = (ys[i + 1] - ys[i - 1]) / (2.0 * dx)
    dy[-1] = (ys[-1] - ys[-2]) / dx
    return dy


def finite_difference_second_order_grid(ys: List[float], dx: float) -> List[float]:
    """Compute second-order numerical derivative across uniform 1D data grid."""
    n = len(ys)
    d2y = [0.0] * n
    if n < 3:
        return d2y
    d2y[0] = (ys[2] - 2.0 * ys[1] + ys[0]) / (dx * dx)
    for i in range(1, n - 1):
        d2y[i] = (ys[i + 1] - 2.0 * ys[i] + ys[i - 1]) / (dx * dx)
    d2y[-1] = (ys[-1] - 2.0 * ys[-2] + ys[-3]) / (dx * dx)
    return d2y
def cubic_spline_clamped_derivatives(xs: List[float], ys: List[float], dy0: float, dyn: float) -> List[float]:
    """Compute clamped cubic spline second derivatives with given endpoint slopes dy0 and dyn."""
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    main = [2.0 * h[0]] + [2.0 * (h[i - 1] + h[i]) for i in range(1, n - 1)] + [2.0 * h[-1]]
    sub = [h[i] for i in range(n - 1)]
    super_diag = [h[i] for i in range(n - 1)]
    rhs0 = 6.0 * ((ys[1] - ys[0]) / h[0] - dy0)
    rhsn = 6.0 * (dyn - (ys[-1] - ys[-2]) / h[-1])
    rhs = [rhs0] + [6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]) for i in range(1, n - 1)] + [rhsn]
    return tridiagonal_thomas_algorithm(sub, main, super_diag, rhs)


def numerical_curl_2d(fx: Callable[[float, float], float],
                      fy: Callable[[float, float], float],
                      x: float, y: float, h: float = 1e-4) -> float:
    """2D curl (vorticity) df_y/dx - df_x/dy."""
    dfy_dx = (fy(x + h, y) - fy(x - h, y)) / (2.0 * h)
    dfx_dy = (fx(x, y + h) - fx(x, y - h)) / (2.0 * h)
    return dfy_dx - dfx_dy
