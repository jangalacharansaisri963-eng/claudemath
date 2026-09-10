"""Polynomial operations module for claudemath.

Pure-Python implementation of polynomial algebra (dense and sparse), long division,
synthetic division, Horner evaluation, root bounding (Cauchy, Lagrange, Descartes, Sturm),
special orthogonal polynomials (Chebyshev, Legendre, Hermite, Laguerre),
polynomial GCD, discriminant, and resultant.
"""

from typing import List, Tuple, Optional, Dict, Any
import math


# ----------------------------------------------------
# 1. Fundamental Polynomial Arithmetic
# ----------------------------------------------------
# Polynomials represented as List[float] where index i is coefficient of x^i:
# [c0, c1, c2, ...] = c0 + c1*x + c2*x^2 + ...

def poly_trim(p: List[float], tol: float = 1e-14) -> List[float]:
    """Strip leading zeros from high-degree terms."""
    p_copy = list(p)
    while len(p_copy) > 1 and abs(p_copy[-1]) < tol:
        p_copy.pop()
    if len(p_copy) == 1 and abs(p_copy[0]) < tol:
        return [0.0]
    return p_copy


def poly_degree(p: List[float]) -> int:
    """Return degree of polynomial (or -1 for zero polynomial)."""
    trimmed = poly_trim(p)
    if len(trimmed) == 1 and trimmed[0] == 0.0:
        return -1
    return len(trimmed) - 1


def poly_add(p1: List[float], p2: List[float]) -> List[float]:
    """Polynomial addition p1(x) + p2(x)."""
    n = max(len(p1), len(p2))
    res = [0.0] * n
    for i in range(len(p1)):
        res[i] += p1[i]
    for i in range(len(p2)):
        res[i] += p2[i]
    return poly_trim(res)


def poly_sub(p1: List[float], p2: List[float]) -> List[float]:
    """Polynomial subtraction p1(x) - p2(x)."""
    n = max(len(p1), len(p2))
    res = [0.0] * n
    for i in range(len(p1)):
        res[i] += p1[i]
    for i in range(len(p2)):
        res[i] -= p2[i]
    return poly_trim(res)


def poly_scalar_mul(p: List[float], s: float) -> List[float]:
    """Scalar multiplication s * p(x)."""
    return poly_trim([c * s for c in p])


def poly_neg(p: List[float]) -> List[float]:
    """Negate polynomial -p(x)."""
    return [-c for c in p]


def poly_mul(p1: List[float], p2: List[float]) -> List[float]:
    """Polynomial multiplication p1(x) * p2(x) via convolution."""
    p1 = poly_trim(p1)
    p2 = poly_trim(p2)
    if (len(p1) == 1 and p1[0] == 0.0) or (len(p2) == 1 and p2[0] == 0.0):
        return [0.0]
    res = [0.0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1 != 0.0:
            for j, c2 in enumerate(p2):
                res[i + j] += c1 * c2
    return poly_trim(res)


def poly_karatsuba_mul(p1: List[float], p2: List[float]) -> List[float]:
    """Karatsuba divide-and-conquer polynomial multiplication."""
    p1 = poly_trim(p1)
    p2 = poly_trim(p2)
    n = max(len(p1), len(p2))
    if n <= 16:
        return poly_mul(p1, p2)

    # Pad to equal power of 2 length
    m = n // 2
    low1 = p1[:m]
    high1 = p1[m:]
    low2 = p2[:m]
    high2 = p2[m:]

    z0 = poly_karatsuba_mul(low1, low2)
    z2 = poly_karatsuba_mul(high1, high2)
    z1 = poly_karatsuba_mul(poly_add(low1, high1), poly_add(low2, high2))
    z1 = poly_sub(poly_sub(z1, z0), z2)

    # Reassemble z0 + z1*x^m + z2*x^(2m)
    res_len = len(z2) + 2 * m
    res = [0.0] * res_len
    for i, c in enumerate(z0):
        res[i] += c
    for i, c in enumerate(z1):
        res[i + m] += c
    for i, c in enumerate(z2):
        res[i + 2 * m] += c
    return poly_trim(res)


def poly_divmod(dividend: List[float], divisor: List[float]) -> Tuple[List[float], List[float]]:
    """Polynomial long division: dividend = quotient * divisor + remainder."""
    num = poly_trim(dividend)
    den = poly_trim(divisor)
    if len(den) == 1 and den[0] == 0.0:
        raise ZeroDivisionError("Polynomial division by zero polynomial")
    if len(num) < len(den):
        return [0.0], num

    rem = list(num)
    deg_den = len(den) - 1
    deg_num = len(num) - 1
    lead_den = den[-1]
    quot = [0.0] * (deg_num - deg_den + 1)

    for i in range(deg_num - deg_den, -1, -1):
        factor = rem[deg_den + i] / lead_den
        quot[i] = factor
        for j in range(deg_den + 1):
            rem[j + i] -= factor * den[j]
    return poly_trim(quot), poly_trim(rem)


def poly_div(p1: List[float], p2: List[float]) -> List[float]:
    """Polynomial quotient p1(x) / p2(x)."""
    q, _ = poly_divmod(p1, p2)
    return q


def poly_mod(p1: List[float], p2: List[float]) -> List[float]:
    """Polynomial remainder p1(x) % p2(x)."""
    _, r = poly_divmod(p1, p2)
    return r


def synthetic_division(p: List[float], r: float) -> Tuple[List[float], float]:
    """Synthetic division of polynomial p(x) by linear factor (x - r). Returns (quotient, remainder)."""
    p = poly_trim(p)
    n = len(p)
    if n <= 1:
        return [0.0], p[0] if p else 0.0
    # Coefficients from highest degree to lowest
    coeffs = list(reversed(p))
    quot_high = [coeffs[0]]
    for c in coeffs[1:-1]:
        quot_high.append(c + quot_high[-1] * r)
    rem = coeffs[-1] + quot_high[-1] * r
    return list(reversed(quot_high)), rem


def poly_power(p: List[float], k: int) -> List[float]:
    """Polynomial power p(x)^k using binary exponentiation."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return [1.0]
    res = [1.0]
    base = list(p)
    while k > 0:
        if k % 2 == 1:
            res = poly_mul(res, base)
        base = poly_mul(base, base)
        k //= 2
    return res


def poly_compose(p: List[float], q: List[float]) -> List[float]:
    """Polynomial composition p(q(x)) via Horner's rule."""
    p = poly_trim(p)
    if not p or (len(p) == 1 and p[0] == 0.0):
        return [0.0]
    res = [p[-1]]
    for c in reversed(p[:-1]):
        res = poly_add(poly_mul(res, q), [c])
    return res


# ----------------------------------------------------
# 2. Evaluation and Calculus on Polynomials
# ----------------------------------------------------

def poly_eval(p: List[float], x: float) -> float:
    """Horner's rule evaluation of polynomial at x: p(x) = c0 + x*(c1 + x*(c2 + ...))."""
    if not p:
        return 0.0
    res = p[-1]
    for c in reversed(p[:-1]):
        res = res * x + c
    return res


def poly_eval_derivative(p: List[float], x: float) -> Tuple[float, float]:
    """Horner's rule evaluation of both p(x) and p'(x) simultaneously."""
    if not p:
        return 0.0, 0.0
    val = p[-1]
    dval = 0.0
    for c in reversed(p[:-1]):
        dval = dval * x + val
        val = val * x + c
    return val, dval


def poly_derivative(p: List[float]) -> List[float]:
    """Formal derivative p'(x) = sum_{i=1}^n i * c_i * x^{i-1}."""
    p = poly_trim(p)
    if len(p) <= 1:
        return [0.0]
    return [i * p[i] for i in range(1, len(p))]


def poly_integral(p: List[float], c0: float = 0.0) -> List[float]:
    """Formal antiderivative int p(x) dx with integration constant c0."""
    p = poly_trim(p)
    res = [c0]
    for i, c in enumerate(p):
        res.append(c / (i + 1))
    return res


def poly_definite_integral(p: List[float], a: float, b: float) -> float:
    """Definite integral of p(x) from a to b."""
    antideriv = poly_integral(p)
    return poly_eval(antideriv, b) - poly_eval(antideriv, a)


def poly_shift(p: List[float], a: float) -> List[float]:
    """Shift variable p(x + a)."""
    return poly_compose(p, [a, 1.0])


# ----------------------------------------------------
# 3. Root Bounds and Root Counting
# ----------------------------------------------------

def poly_cauchy_bound(p: List[float]) -> float:
    """Cauchy's bound: all roots z of p(x) satisfy |z| <= 1 + max_{i < n} |c_i / c_n|."""
    p = poly_trim(p)
    if len(p) <= 1:
        return 0.0
    lead = abs(p[-1])
    max_ratio = max(abs(c) / lead for c in p[:-1])
    return 1.0 + max_ratio


def poly_lagrange_bound(p: List[float]) -> float:
    """Lagrange's bound: max(1, sum_{i=0}^{n-1} |c_i / c_n|)."""
    p = poly_trim(p)
    if len(p) <= 1:
        return 0.0
    lead = abs(p[-1])
    return max(1.0, sum(abs(c) / lead for c in p[:-1]))


def poly_descartes_sign_changes(p: List[float]) -> int:
    """Descartes' rule of signs: count sign variations in non-zero coefficients."""
    p = poly_trim(p)
    non_zeros = [c for c in reversed(p) if abs(c) > 1e-14]
    changes = 0
    for i in range(len(non_zeros) - 1):
        if (non_zeros[i] > 0 and non_zeros[i + 1] < 0) or (non_zeros[i] < 0 and non_zeros[i + 1] > 0):
            changes += 1
    return changes


def poly_sturm_sequence(p: List[float]) -> List[List[float]]:
    """Construct Sturm sequence P_0, P_1, ..., P_m."""
    p = poly_trim(p)
    p0 = p
    p1 = poly_derivative(p0)
    seq = [p0, p1]
    while True:
        _, rem = poly_divmod(seq[-2], seq[-1])
        if len(rem) == 1 and abs(rem[0]) < 1e-12:
            break
        # Negative remainder
        seq.append(poly_neg(rem))
    return seq


def poly_sturm_sign_changes(seq: List[List[float]], x: float) -> int:
    """Count sign variations in Sturm sequence evaluated at x."""
    vals = [poly_eval(p, x) for p in seq if abs(poly_eval(p, x)) > 1e-14]
    changes = 0
    for i in range(len(vals) - 1):
        if (vals[i] > 0 and vals[i + 1] < 0) or (vals[i] < 0 and vals[i + 1] > 0):
            changes += 1
    return changes


def poly_count_real_roots_sturm(p: List[float], a: float, b: float) -> int:
    """Count number of distinct real roots in interval (a, b] using Sturm's theorem."""
    seq = poly_sturm_sequence(p)
    va = poly_sturm_sign_changes(seq, a)
    vb = poly_sturm_sign_changes(seq, b)
    return max(0, va - vb)


# ----------------------------------------------------
# 4. Orthogonal and Special Polynomials
# ----------------------------------------------------

def chebyshev_polynomial_first_kind(n: int) -> List[float]:
    """Chebyshev polynomial of first kind T_n(x). Recurrence: T_0 = 1, T_1 = x, T_{n+1} = 2x*T_n - T_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    t0 = [1.0]
    t1 = [0.0, 1.0]
    for _ in range(2, n + 1):
        # 2x * t1 - t0
        two_x_t1 = poly_mul([0.0, 2.0], t1)
        t_next = poly_sub(two_x_t1, t0)
        t0, t1 = t1, t_next
    return t1


def chebyshev_polynomial_second_kind(n: int) -> List[float]:
    """Chebyshev polynomial of second kind U_n(x). Recurrence: U_0 = 1, U_1 = 2x, U_{n+1} = 2x*U_n - U_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    u0 = [1.0]
    u1 = [0.0, 2.0]
    for _ in range(2, n + 1):
        two_x_u1 = poly_mul([0.0, 2.0], u1)
        u_next = poly_sub(two_x_u1, u0)
        u0, u1 = u1, u_next
    return u1


def legendre_polynomial(n: int) -> List[float]:
    """Legendre polynomial P_n(x). Bonnet's recurrence: (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    p0 = [1.0]
    p1 = [0.0, 1.0]
    for k in range(1, n):
        # ((2k + 1) * x * p1 - k * p0) / (k + 1)
        term1 = poly_mul([0.0, float(2 * k + 1)], p1)
        term2 = poly_scalar_mul(p0, float(k))
        p_next = poly_scalar_mul(poly_sub(term1, term2), 1.0 / (k + 1))
        p0, p1 = p1, p_next
    return p1


def hermite_polynomial_physicists(n: int) -> List[float]:
    """Physicists' Hermite polynomial H_n(x): H_0 = 1, H_1 = 2x, H_{n+1} = 2x*H_n - 2n*H_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h0 = [1.0]
    h1 = [0.0, 2.0]
    for k in range(1, n):
        term1 = poly_mul([0.0, 2.0], h1)
        term2 = poly_scalar_mul(h0, 2.0 * k)
        h_next = poly_sub(term1, term2)
        h0, h1 = h1, h_next
    return h1


def hermite_polynomial_probabilists(n: int) -> List[float]:
    """Probabilists' Hermite polynomial He_n(x): He_0 = 1, He_1 = x, He_{n+1} = x*He_n - n*He_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    he0 = [1.0]
    he1 = [0.0, 1.0]
    for k in range(1, n):
        term1 = poly_mul([0.0, 1.0], he1)
        term2 = poly_scalar_mul(he0, float(k))
        he_next = poly_sub(term1, term2)
        he0, he1 = he1, he_next
    return he1


def laguerre_polynomial(n: int) -> List[float]:
    """Laguerre polynomial L_n(x): L_0 = 1, L_1 = 1 - x, (k+1) L_{k+1} = (2k+1 - x) L_k - k L_{k-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [1.0, -1.0]
    l0 = [1.0]
    l1 = [1.0, -1.0]
    for k in range(1, n):
        term1 = poly_mul([float(2 * k + 1), -1.0], l1)
        term2 = poly_scalar_mul(l0, float(k))
        l_next = poly_scalar_mul(poly_sub(term1, term2), 1.0 / (k + 1))
        l0, l1 = l1, l_next
    return l1


def bernstein_polynomial(n: int, v: int) -> List[float]:
    """Bernstein basis polynomial b_{v, n}(x) = C(n, v) * x^v * (1 - x)^{n - v}."""
    if v < 0 or v > n:
        return [0.0]
    comb = math.comb(n, v)
    # (1 - x)^{n - v}
    one_minus_x = poly_power([1.0, -1.0], n - v)
    # x^v
    x_v = [0.0] * v + [1.0]
    return poly_scalar_mul(poly_mul(x_v, one_minus_x), float(comb))


def wilkinson_polynomial(n: int = 20) -> List[float]:
    """Construct Wilkinson's polynomial prod_{i=1}^n (x - i)."""
    res = [1.0]
    for i in range(1, n + 1):
        res = poly_mul(res, [-float(i), 1.0])
    return res


# ----------------------------------------------------
# 5. Polynomial GCD, Resultant, and Discriminant
# ----------------------------------------------------

def poly_gcd(p1: List[float], p2: List[float], tol: float = 1e-10) -> List[float]:
    """Monic polynomial Greatest Common Divisor via Euclidean algorithm."""
    a = poly_trim(p1)
    b = poly_trim(p2)
    while len(b) > 1 or abs(b[0]) > tol:
        _, r = poly_divmod(a, b)
        a, b = b, r
    # Make monic
    if abs(a[-1]) > tol:
        lead = a[-1]
        return [c / lead for c in a]
    return a


def sylvester_matrix(p1: List[float], p2: List[float]) -> List[List[float]]:
    """Construct Sylvester matrix of two polynomials for resultant computation."""
    p1 = poly_trim(p1)
    p2 = poly_trim(p2)
    m = len(p1) - 1
    n = len(p2) - 1
    if m <= 0 or n <= 0:
        return [[0.0]]
    size = m + n
    S = [[0.0] * size for _ in range(size)]
    # First n rows for p1
    coeffs1 = list(reversed(p1))
    for i in range(n):
        for j, c in enumerate(coeffs1):
            S[i][i + j] = c
    # Next m rows for p2
    coeffs2 = list(reversed(p2))
    for i in range(m):
        for j, c in enumerate(coeffs2):
            S[n + i][i + j] = c
    return S


def poly_resultant(p1: List[float], p2: List[float]) -> float:
    """Resultant of two polynomials Res(p1, p2) = det(Sylvester(p1, p2))."""
    S = sylvester_matrix(p1, p2)
    # Gaussian determinant
    n = len(S)
    det = 1.0
    M = [row[:] for row in S]
    for i in range(n):
        pivot_idx = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_idx][i]) < 1e-14:
            return 0.0
        if pivot_idx != i:
            M[i], M[pivot_idx] = M[pivot_idx], M[i]
            det = -det
        pivot = M[i][i]
        det *= pivot
        for r in range(i + 1, n):
            factor = M[r][i] / pivot
            for c in range(i, n):
                M[r][c] -= factor * M[i][c]
    return det


def poly_discriminant(p: List[float]) -> float:
    """Discriminant of polynomial Disc(p) = ((-1)^{n(n-1)/2} / a_n) * Res(p, p')."""
    p = poly_trim(p)
    n = len(p) - 1
    if n <= 1:
        return 1.0
    dp = poly_derivative(p)
    res = poly_resultant(p, dp)
    sign = (-1.0) ** (n * (n - 1) // 2)
    lead = p[-1]
    return (sign / lead) * res
def poly_from_roots(roots: List[float]) -> List[float]:
    """Construct monic polynomial prod_{r in roots} (x - r)."""
    res = [1.0]
    for r in roots:
        res = poly_mul(res, [-r, 1.0])
    return res


def poly_roots_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Roots of quadratic equation a*x^2 + b*x + c = 0."""
    if a == 0:
        if b == 0:
            raise ValueError("Degenerate equation")
        return complex(-c / b, 0.0), complex(-c / b, 0.0)
    disc = b * b - 4.0 * a * c
    if disc >= 0:
        r1 = (-b + math.sqrt(disc)) / (2.0 * a)
        r2 = (-b - math.sqrt(disc)) / (2.0 * a)
        return complex(r1, 0.0), complex(r2, 0.0)
    else:
        real = -b / (2.0 * a)
        imag = math.sqrt(-disc) / (2.0 * a)
        return complex(real, imag), complex(real, -imag)


def poly_roots_cubic_cardano(a: float, b: float, c: float, d: float) -> Tuple[complex, complex, complex]:
    """Analytic cubic roots via Cardano-Tartaglia formula for a*x^3 + b*x^2 + c*x + d = 0."""
    if a == 0:
        raise ValueError("Leading coefficient must be non-zero for cubic")
    # Depressed cubic t^3 + p*t + q = 0 with x = t - b/(3a)
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * b ** 3 - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a ** 3)
    shift = -b / (3.0 * a)

    disc = (q * 0.5) ** 2 + (p / 3.0) ** 3
    if disc >= 0:
        u = (-q * 0.5 + math.sqrt(disc))
        u_val = math.copysign(abs(u) ** (1.0 / 3.0), u)
        v = (-q * 0.5 - math.sqrt(disc))
        v_val = math.copysign(abs(v) ** (1.0 / 3.0), v)
        t1 = u_val + v_val
        t2 = complex(-0.5 * (u_val + v_val), 0.5 * math.sqrt(3.0) * (u_val - v_val))
        t3 = complex(-0.5 * (u_val + v_val), -0.5 * math.sqrt(3.0) * (u_val - v_val))
        return complex(t1 + shift, 0.0), t2 + shift, t3 + shift
    else:
        # 3 real roots via trigonometric formula
        r = math.sqrt(-(p / 3.0) ** 3)
        phi = math.acos(-q / (2.0 * r))
        m = 2.0 * (-(p / 3.0)) ** 0.5
        t1 = m * math.cos(phi / 3.0)
        t2 = m * math.cos((phi + 2.0 * math.pi) / 3.0)
        t3 = m * math.cos((phi + 4.0 * math.pi) / 3.0)
        return complex(t1 + shift, 0.0), complex(t2 + shift, 0.0), complex(t3 + shift, 0.0)


def poly_deflate(p: List[float], root: float, tol: float = 1e-9) -> List[float]:
    """Deflate polynomial p(x) by dividing out root factor (x - root)."""
    quot, rem = synthetic_division(p, root)
    if abs(rem) > tol:
        raise ValueError(f"Value {root} is not an exact root (rem = {rem})")
    return quot


def fibonacci_polynomial(n: int) -> List[float]:
    """Fibonacci polynomial F_n(x): F_0 = 0, F_1 = 1, F_{n+1} = x*F_n + F_{n-1}."""
    if n <= 0:
        return [0.0]
    if n == 1:
        return [1.0]
    f0 = [0.0]
    f1 = [1.0]
    for _ in range(2, n + 1):
        x_f1 = poly_mul([0.0, 1.0], f1)
        f_next = poly_add(x_f1, f0)
        f0, f1 = f1, f_next
    return f1


def lucas_polynomial(n: int) -> List[float]:
    """Lucas polynomial L_n(x): L_0 = 2, L_1 = x, L_{n+1} = x*L_n + L_{n-1}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [2.0]
    if n == 1:
        return [0.0, 1.0]
    l0 = [2.0]
    l1 = [0.0, 1.0]
    for _ in range(2, n + 1):
        x_l1 = poly_mul([0.0, 1.0], l1)
        l_next = poly_add(x_l1, l0)
        l0, l1 = l1, l_next
    return l1


def gegenbauer_polynomial(n: int, alpha: float) -> List[float]:
    """Gegenbauer (ultraspherical) polynomial C_n^(alpha)(x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0 * alpha]
    c0 = [1.0]
    c1 = [0.0, 2.0 * alpha]
    for k in range(1, n):
        term1 = poly_mul([0.0, 2.0 * (k + alpha)], c1)
        term2 = poly_scalar_mul(c0, float(k + 2 * alpha - 1))
        c_next = poly_scalar_mul(poly_sub(term1, term2), 1.0 / (k + 1))
        c0, c1 = c1, c_next
    return c1


def jacobi_polynomial(n: int, alpha: float, beta: float) -> List[float]:
    """Jacobi orthogonal polynomial P_n^(alpha, beta)(x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.5 * (alpha - beta), 0.5 * (alpha + beta + 2.0)]
    p0 = [1.0]
    p1 = [0.5 * (alpha - beta), 0.5 * (alpha + beta + 2.0)]
    for n_curr in range(1, n):
        c = 2 * n_curr + alpha + beta
        a1 = 2 * (n_curr + 1) * (n_curr + alpha + beta + 1) * c
        a2 = (c + 1) * (alpha * alpha - beta * beta)
        a3 = (c) * (c + 1) * (c + 2)
        a4 = 2 * (n_curr + alpha) * (n_curr + beta) * (c + 2)
        # a1 * P_{n+1} = (a2 + a3*x) * P_n - a4 * P_{n-1}
        linear = [a2, a3]
        term1 = poly_mul(linear, p1)
        term2 = poly_scalar_mul(p0, a4)
        p_next = poly_scalar_mul(poly_sub(term1, term2), 1.0 / a1)
        p0, p1 = p1, p_next
    return p1


def sparse_to_dense_poly(sp: Dict[int, float]) -> List[float]:
    """Convert sparse degree:coeff dict to dense list."""
    if not sp:
        return [0.0]
    deg = max(sp.keys())
    res = [0.0] * (deg + 1)
    for d, c in sp.items():
        res[d] = c
    return poly_trim(res)


def dense_to_sparse_poly(p: List[float], tol: float = 1e-14) -> Dict[int, float]:
    """Convert dense polynomial list to sparse degree:coeff dict."""
    return {i: c for i, c in enumerate(p) if abs(c) > tol}


def sparse_poly_add(sp1: Dict[int, float], sp2: Dict[int, float]) -> Dict[int, float]:
    """Add two sparse polynomials."""
    res = dict(sp1)
    for d, c in sp2.items():
        res[d] = res.get(d, 0.0) + c
        if abs(res[d]) < 1e-14:
            del res[d]
    return res


def sparse_poly_mul(sp1: Dict[int, float], sp2: Dict[int, float]) -> Dict[int, float]:
    """Multiply two sparse polynomials."""
    res = {}
    for d1, c1 in sp1.items():
        for d2, c2 in sp2.items():
            deg = d1 + d2
            res[deg] = res.get(deg, 0.0) + c1 * c2
            if abs(res[deg]) < 1e-14:
                del res[deg]
    return res


def sparse_poly_eval(sp: Dict[int, float], x: float) -> float:
    """Evaluate sparse polynomial at x."""
    return sum(c * (x ** d) for d, c in sp.items())


def sparse_poly_derivative(sp: Dict[int, float]) -> Dict[int, float]:
    """Formal derivative of sparse polynomial."""
    res = {}
    for d, c in sp.items():
        if d > 0:
            res[d - 1] = c * d
    return res


def poly_square_free_factorization(p: List[float]) -> Tuple[List[float], List[float]]:
    """Compute square-free part = p / gcd(p, p')."""
    dp = poly_derivative(p)
    g = poly_gcd(p, dp)
    sq_free, _ = poly_divmod(p, g)
    return sq_free, g
def poly_monic(p: List[float]) -> List[float]:
    """Return monic polynomial p / leading_coef."""
    p = poly_trim(p)
    if len(p) == 1 and p[0] == 0.0:
        return [0.0]
    lead = p[-1]
    return [c / lead for c in p]


def poly_scale_x(p: List[float], a: float) -> List[float]:
    """Scale variable p(a*x)."""
    return [c * (a ** i) for i, c in enumerate(p)]


def poly_reciprocal(p: List[float]) -> List[float]:
    """Reciprocal polynomial x^deg * p(1/x)."""
    p = poly_trim(p)
    return list(reversed(p))


def poly_even_part(p: List[float]) -> List[float]:
    """Even component of polynomial (c0 + c2*x^2 + c4*x^4 + ...)."""
    return [c if i % 2 == 0 else 0.0 for i, c in enumerate(p)]


def poly_odd_part(p: List[float]) -> List[float]:
    """Odd component of polynomial (c1*x + c3*x^3 + ...)."""
    return [c if i % 2 == 1 else 0.0 for i, c in enumerate(p)]


def poly_inner_product_l2(p1: List[float], p2: List[float], a: float, b: float) -> float:
    """L2 inner product int_a^b p1(x) * p2(x) dx."""
    prod = poly_mul(p1, p2)
    return poly_definite_integral(prod, a, b)


def poly_norm_l2(p: List[float], a: float, b: float) -> float:
    """L2 norm sqrt(int_a^b p(x)^2 dx)."""
    val = poly_inner_product_l2(p, p, a, b)
    return math.sqrt(max(0.0, val))


def bernoulli_polynomial_eval(n: int, x: float) -> float:
    """Evaluate n-th Bernoulli polynomial B_n(x) = sum_{k=0}^n C(n, k) * B_k * x^{n-k}."""
    # Generate Bernoulli numbers
    b = [0.0] * (n + 1)
    for m in range(n + 1):
        b[m] = 1.0 / (m + 1)
        for j in range(m, 0, -1):
            b[j - 1] = float(j) * (b[j - 1] - b[j])
    # Now evaluate sum
    total = 0.0
    for k in range(n + 1):
        total += math.comb(n, k) * b[k] * (x ** (n - k))
    return total


def euler_polynomial_eval(n: int, x: float) -> float:
    """Evaluate n-th Euler polynomial E_n(x)."""
    total = 0.0
    for k in range(n + 1):
        # E_n(x) = (2 / (n + 1)) * (B_{n+1}(x) - 2^{n+1} * B_{n+1}(x / 2))
        pass
    # Direct formula via binomial coefficients and Euler numbers
    val = (2.0 / (n + 1)) * (bernoulli_polynomial_eval(n + 1, x) - (2.0 ** (n + 1)) * bernoulli_polynomial_eval(n + 1, x * 0.5))
    return val
