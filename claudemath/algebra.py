"""Algebra module for claudemath.

Pure-Python implementation of algebraic equation solvers (quadratic, cubic, quartic),
polynomial arithmetic and root bounds, rational expressions, group and ring theory,
exponents and radicals, and abstract algebraic structures.
"""

from typing import List, Tuple, Optional, Dict, Any, Callable
import math
import cmath


# ----------------------------------------------------
# 1. Algebraic Equations and Solvers
# ----------------------------------------------------

def solve_linear(a: float, b: float) -> float:
    """Solve linear equation a*x + b = 0."""
    if a == 0:
        raise ValueError("Cannot solve 0*x + b = 0 (no unique solution)")
    return -b / a


def quadratic_discriminant(a: float, b: float, c: float) -> float:
    """Compute discriminant Delta = b^2 - 4*a*c."""
    return b * b - 4.0 * a * c


def quadratic_vertex(a: float, b: float, c: float) -> Tuple[float, float]:
    """Find vertex (h, k) of parabola y = a*x^2 + b*x + c."""
    if a == 0:
        raise ValueError("Coefficient a cannot be zero")
    h = -b / (2.0 * a)
    k = c - (b * b) / (4.0 * a)
    return h, k


def solve_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Solve quadratic equation a*x^2 + b*x + c = 0."""
    if a == 0:
        raise ValueError("Coefficient a cannot be zero")
    disc = complex(b * b - 4.0 * a * c, 0.0)
    root_disc = cmath.sqrt(disc)
    x1 = (-b + root_disc) / (2.0 * a)
    x2 = (-b - root_disc) / (2.0 * a)
    return x1, x2


def cubic_discriminant(a: float, b: float, c: float, d: float) -> float:
    """Compute discriminant of cubic a*x^3 + b*x^2 + c*x + d = 0."""
    return (18.0 * a * b * c * d - 4.0 * (b ** 3) * d +
            (b ** 2) * (c ** 2) - 4.0 * a * (c ** 3) - 27.0 * (a ** 2) * (d ** 2))


def solve_cubic_cardano(a: float, b: float, c: float, d: float) -> Tuple[complex, complex, complex]:
    """Solve cubic equation a*x^3 + b*x^2 + c*x + d = 0 using Cardano's formula."""
    if a == 0:
        raise ValueError("Leading coefficient a cannot be zero")
    # Depress to t^3 + p*t + q = 0 by substituting x = t - b/(3a)
    p = (3.0 * a * c - b * b) / (3.0 * a * a)
    q = (2.0 * (b ** 3) - 9.0 * a * b * c + 27.0 * (a ** 2) * d) / (27.0 * (a ** 3))

    delta = complex((q / 2.0) ** 2 + (p / 3.0) ** 3, 0.0)
    sqrt_delta = cmath.sqrt(delta)

    # Cube roots
    u = (-q / 2.0 + sqrt_delta) ** (1.0 / 3.0)
    v = (-q / 2.0 - sqrt_delta) ** (1.0 / 3.0)

    shift = -b / (3.0 * a)
    omega = complex(-0.5, math.sqrt(3.0) / 2.0)
    omega2 = complex(-0.5, -math.sqrt(3.0) / 2.0)

    t1 = u + v
    t2 = omega * u + omega2 * v
    t3 = omega2 * u + omega * v

    return t1 + shift, t2 + shift, t3 + shift


def solve_quartic_ferrari(a: float, b: float, c: float, d: float, e: float) -> Tuple[complex, complex, complex, complex]:
    """Solve monic/general quartic a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 using Ferrari's method."""
    if a == 0:
        raise ValueError("Leading coefficient a cannot be zero")
    # Normalize to monic: x^4 + B*x^3 + C*x^2 + D*x + E = 0
    B, C, D, E = b / a, c / a, d / a, e / a
    # Depressed quartic: y^4 + p*y^2 + q*y + r = 0 with x = y - B/4
    p = C - 3.0 * (B ** 2) / 8.0
    q = D - B * C / 2.0 + (B ** 3) / 8.0
    r = E - B * D / 4.0 + (B ** 2) * C / 16.0 - 3.0 * (B ** 4) / 256.0

    shift = -B / 4.0

    if abs(q) < 1e-12:
        # Biquadratic: y^4 + p*y^2 + r = 0
        z1, z2 = solve_quadratic(1.0, p, r)
        y1, y2 = cmath.sqrt(z1), -cmath.sqrt(z1)
        y3, y4 = cmath.sqrt(z2), -cmath.sqrt(z2)
        return y1 + shift, y2 + shift, y3 + shift, y4 + shift

    # Resolvent cubic in m: 8*m^3 + 8*p*m^2 + (2*p^2 - 8*r)*m - q^2 = 0
    m_roots = solve_cubic_cardano(8.0, 8.0 * p, 2.0 * p * p - 8.0 * r, -q * q)
    m = m_roots[0]
    sqrt_2m = cmath.sqrt(2.0 * m)
    sqrt_term1 = cmath.sqrt(-(2.0 * p + 2.0 * m + math.sqrt(2.0) * q / sqrt_2m))
    sqrt_term2 = cmath.sqrt(-(2.0 * p + 2.0 * m - math.sqrt(2.0) * q / sqrt_2m))

    y1 = 0.5 * (sqrt_2m + sqrt_term1)
    y2 = 0.5 * (sqrt_2m - sqrt_term1)
    y3 = 0.5 * (-sqrt_2m + sqrt_term2)
    y4 = 0.5 * (-sqrt_2m - sqrt_term2)

    return y1 + shift, y2 + shift, y3 + shift, y4 + shift


# ----------------------------------------------------
# 2. Polynomial Arithmetic and Bounds
# ----------------------------------------------------

def polynomial_eval_horner(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial sum(coeffs[i] * x^i) using Horner's rule."""
    if not coeffs:
        return 0.0
    res = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        res = res * x + c
    return res


def polynomial_add(p: List[float], q: List[float]) -> List[float]:
    """Add two polynomials p(x) + q(x)."""
    length = max(len(p), len(q))
    res = [0.0] * length
    for i in range(len(p)):
        res[i] += p[i]
    for i in range(len(q)):
        res[i] += q[i]
    while len(res) > 1 and abs(res[-1]) < 1e-14:
        res.pop()
    return res


def polynomial_sub(p: List[float], q: List[float]) -> List[float]:
    """Subtract two polynomials p(x) - q(x)."""
    length = max(len(p), len(q))
    res = [0.0] * length
    for i in range(len(p)):
        res[i] += p[i]
    for i in range(len(q)):
        res[i] -= q[i]
    while len(res) > 1 and abs(res[-1]) < 1e-14:
        res.pop()
    return res


def polynomial_mul(p: List[float], q: List[float]) -> List[float]:
    """Multiply two polynomials p(x) * q(x)."""
    if not p or not q or (len(p) == 1 and p[0] == 0) or (len(q) == 1 and q[0] == 0):
        return [0.0]
    res = [0.0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            res[i + j] += a * b
    return res


def polynomial_scale(p: List[float], s: float) -> List[float]:
    """Scalar multiplication s * p(x)."""
    return [s * x for x in p]


def polynomial_derivative(p: List[float]) -> List[float]:
    """Formal derivative of polynomial p'(x)."""
    if len(p) <= 1:
        return [0.0]
    return [i * p[i] for i in range(1, len(p))]


def polynomial_integral(p: List[float], c: float = 0.0) -> List[float]:
    """Formal antiderivative of polynomial with constant of integration c."""
    res = [c]
    for i, coeff in enumerate(p):
        res.append(coeff / (i + 1))
    return res


def polynomial_divmod(dividend: List[float], divisor: List[float]) -> Tuple[List[float], List[float]]:
    """Polynomial division returning (quotient, remainder)."""
    if not divisor or all(abs(c) < 1e-14 for c in divisor):
        raise ZeroDivisionError("Cannot divide by zero polynomial")
    p = list(dividend)
    d = list(divisor)
    while len(d) > 1 and abs(d[-1]) < 1e-14:
        d.pop()
    quotient = [0.0] * max(1, len(p) - len(d) + 1)
    while len(p) >= len(d) and any(abs(c) > 1e-14 for c in p):
        lead_p = p[-1]
        lead_d = d[-1]
        deg_diff = len(p) - len(d)
        factor = lead_p / lead_d
        quotient[deg_diff] = factor
        for i in range(len(d)):
            p[i + deg_diff] -= factor * d[i]
        while p and abs(p[-1]) < 1e-14:
            p.pop()
    remainder = p if p else [0.0]
    return quotient, remainder


def synthetic_division(poly: List[float], r: float) -> Tuple[List[float], float]:
    """Divide polynomial by (x - r) using synthetic division."""
    # poly = [c_0, c_1, ..., c_n]
    coeffs = list(reversed(poly))
    n = len(coeffs)
    q = [coeffs[0]]
    for i in range(1, n - 1):
        q.append(coeffs[i] + q[-1] * r)
    rem = coeffs[-1] + q[-1] * r if n > 1 else coeffs[0]
    return list(reversed(q)), rem


def descartes_rule_of_signs_max_positive(coeffs: List[float]) -> int:
    """Maximum number of positive real roots using Descartes' Rule of Signs."""
    nonzero = [c for c in reversed(coeffs) if abs(c) > 1e-14]
    sign_changes = 0
    for i in range(len(nonzero) - 1):
        if (nonzero[i] > 0 and nonzero[i + 1] < 0) or (nonzero[i] < 0 and nonzero[i + 1] > 0):
            sign_changes += 1
    return sign_changes


def descartes_rule_of_signs_max_negative(coeffs: List[float]) -> int:
    """Maximum number of negative real roots using Descartes' Rule of Signs."""
    neg_poly = [coeffs[i] * ((-1) ** i) for i in range(len(coeffs))]
    return descartes_rule_of_signs_max_positive(neg_poly)


def cauchy_root_bound(coeffs: List[float]) -> float:
    """Cauchy upper bound for the absolute values of polynomial roots: 1 + max(|a_k / a_n|)."""
    if len(coeffs) <= 1:
        return 0.0
    an = abs(coeffs[-1])
    if an == 0:
        raise ValueError("Leading coefficient cannot be zero")
    return 1.0 + max(abs(c) / an for c in coeffs[:-1])


def rational_root_candidates(coeffs: List[int]) -> List[Tuple[int, int]]:
    """Generate potential rational roots (p, q) via Rational Root Theorem."""
    a0 = abs(coeffs[0])
    an = abs(coeffs[-1])
    if a0 == 0 or an == 0:
        return [(0, 1)]
    p_divs = [d for d in range(1, a0 + 1) if a0 % d == 0]
    q_divs = [d for d in range(1, an + 1) if an % d == 0]
    candidates = set()
    for p in p_divs:
        for q in q_divs:
            g = math.gcd(p, q)
            candidates.add((p // g, q // g))
            candidates.add((-p // g, q // g))
    return sorted(list(candidates))


# ----------------------------------------------------
# 3. Rational and Radical Expressions
# ----------------------------------------------------

def simplify_fraction(num: int, den: int) -> Tuple[int, int]:
    """Reduce fraction num / den to lowest terms with positive denominator."""
    if den == 0:
        raise ZeroDivisionError("Denominator cannot be zero")
    g = math.gcd(abs(num), abs(den))
    s = 1 if den > 0 else -1
    return s * (num // g), abs(den // g)


def add_fractions(n1: int, d1: int, n2: int, d2: int) -> Tuple[int, int]:
    """Add two rational fractions n1/d1 + n2/d2."""
    return simplify_fraction(n1 * d2 + n2 * d1, d1 * d2)


def sub_fractions(n1: int, d1: int, n2: int, d2: int) -> Tuple[int, int]:
    """Subtract two rational fractions n1/d1 - n2/d2."""
    return simplify_fraction(n1 * d2 - n2 * d1, d1 * d2)


def mul_fractions(n1: int, d1: int, n2: int, d2: int) -> Tuple[int, int]:
    """Multiply two rational fractions (n1/d1) * (n2/d2)."""
    return simplify_fraction(n1 * n2, d1 * d2)


def div_fractions(n1: int, d1: int, n2: int, d2: int) -> Tuple[int, int]:
    """Divide two rational fractions (n1/d1) / (n2/d2)."""
    if n2 == 0:
        raise ZeroDivisionError("Cannot divide by zero fraction")
    return simplify_fraction(n1 * d2, d1 * n2)


def simplify_radical_int(n: int) -> Tuple[int, int]:
    """Simplify sqrt(n) into a * sqrt(b) where b is squarefree."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0, 0
    a = 1
    d = 2
    temp = n
    while d * d <= temp:
        while temp % (d * d) == 0:
            a *= d
            temp //= (d * d)
        d += 1
    return a, temp


def integer_nth_root(n: int, k: int) -> int:
    """Exact integer floor of k-th root of n (floor(n^(1/k)))."""
    if n < 0 and k % 2 == 0:
        raise ValueError("Even root of negative number")
    if n == 0:
        return 0
    if k == 1:
        return n
    sign = -1 if n < 0 else 1
    n_abs = abs(n)
    x = int(n_abs ** (1.0 / k))
    if (x + 1) ** k <= n_abs:
        x += 1
    while (x ** k) > n_abs:
        x -= 1
    return sign * x


def log_change_of_base(x: float, new_base: float) -> float:
    """Compute log_new_base(x) using change of base formula."""
    if x <= 0 or new_base <= 0 or new_base == 1.0:
        raise ValueError("Invalid logarithm argument or base")
    return math.log(x) / math.log(new_base)


def lambert_w0(x: float, tol: float = 1e-10, max_iter: int = 50) -> float:
    """Principal branch of Lambert W function W_0(x) (solving w * e^w = x)."""
    if x < -1.0 / math.e:
        raise ValueError("x < -1/e has no real solution")
    w = 0.0 if x < 1.0 else math.log(x)
    for _ in range(max_iter):
        ew = math.exp(w)
        w_ew = w * ew
        f = w_ew - x
        if abs(f) < tol:
            return w
        f_prime = ew * (w + 1.0)
        f_double_prime = ew * (w + 2.0)
        # Halley iteration
        step = (2.0 * f * f_prime) / (2.0 * (f_prime ** 2) - f * f_double_prime)
        w -= step
    return w


# ----------------------------------------------------
# 4. Sequences, Series, and Progressions
# ----------------------------------------------------

def arithmetic_progression_term(a1: float, d: float, n: int) -> float:
    """Compute n-th term of arithmetic progression a_n = a_1 + (n - 1)*d."""
    return a1 + (n - 1) * d


def arithmetic_progression_sum(a1: float, d: float, n: int) -> float:
    """Sum of first n terms of arithmetic progression: (n / 2) * (2*a1 + (n - 1)*d)."""
    return (n * 0.5) * (2.0 * a1 + (n - 1) * d)


def geometric_progression_term(a1: float, r: float, n: int) -> float:
    """Compute n-th term of geometric progression a_n = a_1 * r^(n - 1)."""
    return a1 * (r ** (n - 1))


def geometric_progression_sum_finite(a1: float, r: float, n: int) -> float:
    """Sum of first n terms of geometric progression: a1 * (1 - r^n) / (1 - r)."""
    if r == 1.0:
        return a1 * n
    return a1 * (1.0 - r ** n) / (1.0 - r)


def geometric_progression_sum_infinite(a1: float, r: float) -> float:
    """Sum of infinite convergent geometric series: a1 / (1 - r) for |r| < 1."""
    if abs(r) >= 1.0:
        raise ValueError("Geometric series diverges for |r| >= 1")
    return a1 / (1.0 - r)


def harmonic_progression_term(a1: float, d: float, n: int) -> float:
    """Compute n-th term of harmonic progression 1 / (a1 + (n - 1)*d)."""
    denom = a1 + (n - 1) * d
    if denom == 0:
        raise ZeroDivisionError("Harmonic progression term has zero denominator")
    return 1.0 / denom


# ----------------------------------------------------
# 5. Abstract Algebra (Groups, Permutations, Operations)
# ----------------------------------------------------

def is_associative_operation(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if binary operation op is associative over elements."""
    for a in elements:
        for b in elements:
            for c in elements:
                if op(op(a, b), c) != op(a, op(b, c)):
                    return False
    return True


def is_commutative_operation(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if binary operation op is commutative over elements."""
    for a in elements:
        for b in elements:
            if op(a, b) != op(b, a):
                return False
    return True


def find_identity_element(elements: List[Any], op: Callable[[Any, Any], Any]) -> Optional[Any]:
    """Find identity element e in elements such that op(e, x) = op(x, e) = x for all x."""
    for candidate in elements:
        is_id = True
        for x in elements:
            if op(candidate, x) != x or op(x, candidate) != x:
                is_id = False
                break
        if is_id:
            return candidate
    return None


def is_group(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if elements and operation op form a valid finite group."""
    e = find_identity_element(elements, op)
    if e is None:
        return False
    if not is_associative_operation(elements, op):
        return False
    elem_set = set(elements)
    for a in elements:
        # Closure check
        for b in elements:
            if op(a, b) not in elem_set:
                return False
        # Inverse check
        has_inv = any(op(a, b) == e and op(b, a) == e for b in elements)
        if not has_inv:
            return False
    return True


def is_abelian_group(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if group is abelian (commutative)."""
    return is_group(elements, op) and is_commutative_operation(elements, op)


def permutation_compose(p: List[int], q: List[int]) -> List[int]:
    """Compose two permutations (p o q)(i) = p[q[i]] (0-indexed)."""
    return [p[q[i]] for i in range(len(p))]


def permutation_inverse(p: List[int]) -> List[int]:
    """Compute inverse of permutation p."""
    inv = [0] * len(p)
    for i, val in enumerate(p):
        inv[val] = i
    return inv


def permutation_cycles(p: List[int]) -> List[List[int]]:
    """Decompose permutation into disjoint cyclic representation."""
    n = len(p)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                cycle.append(curr)
                curr = p[curr]
            if len(cycle) > 1:
                cycles.append(cycle)
    return cycles


def permutation_sign_parity(p: List[int]) -> int:
    """Return sign of permutation sgn(sigma) = (-1)^inversions (1 for even, -1 for odd)."""
    inversions = 0
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                inversions += 1
    return -1 if inversions % 2 == 1 else 1


def cyclic_group_generators(n: int) -> List[int]:
    """Return all generators of cyclic group Z_n under addition modulo n."""
    return [k for k in range(1, n) if math.gcd(k, n) == 1]
def solve_linear_system_2x2(a1: float, b1: float, c1: float,
                            a2: float, b2: float, c2: float) -> Tuple[float, float]:
    """Solve 2x2 linear system a1*x + b1*y = c1 and a2*x + b2*y = c2 using Cramer's rule."""
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-14:
        raise ValueError("System has no unique solution (determinant is zero)")
    x = (c1 * b2 - c2 * b1) / det
    y = (a1 * c2 - a2 * c1) / det
    return x, y


def cramers_rule_2x2(a1: float, b1: float, c1: float,
                     a2: float, b2: float, c2: float) -> Tuple[float, float]:
    """Alias for solve_linear_system_2x2."""
    return solve_linear_system_2x2(a1, b1, c1, a2, b2, c2)


def complete_the_square(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Express a*x^2 + b*x + c as a*(x - h)^2 + k. Returns (a, h, k)."""
    if a == 0:
        raise ValueError("Leading coefficient cannot be zero")
    h = -b / (2.0 * a)
    k = c - (b * b) / (4.0 * a)
    return a, h, k


def vieta_formulas_quadratic(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Return sum of roots (-b/a) and product of roots (c/a)."""
    if a == 0:
        raise ValueError("Leading coefficient cannot be zero")
    return -b / a, c / a


def vieta_formulas_cubic(a: float, b: float, c: float, d: float) -> Tuple[float, float, float]:
    """Return (sum of roots, sum of pairwise products, product of roots) for cubic."""
    if a == 0:
        raise ValueError("Leading coefficient cannot be zero")
    return -b / a, c / a, -d / a


def binomial_expansion_coeffs(n: int) -> List[int]:
    """Return binomial coefficients for (a + b)^n [C(n,0), C(n,1), ..., C(n,n)]."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return [math.comb(n, k) for k in range(n + 1)]


def multinomial_coefficient(ks: List[int]) -> int:
    """Compute multinomial coefficient (sum(k_i))! / prod(k_i!)."""
    n = sum(ks)
    res = 1
    running = 0
    for k in ks:
        running += k
        res *= math.comb(running, k)
    return res


def polynomial_compose(p: List[float], q: List[float]) -> List[float]:
    """Compute composition p(q(x))."""
    if not p:
        return [0.0]
    res = [p[-1]]
    for coeff in reversed(p[:-1]):
        res = polynomial_add(polynomial_mul(res, q), [coeff])
    return res


def polynomial_power(p: List[float], n: int) -> List[float]:
    """Compute p(x)^n via repeated squaring."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    res = [1.0]
    base = list(p)
    while n > 0:
        if n % 2 == 1:
            res = polynomial_mul(res, base)
        base = polynomial_mul(base, base)
        n //= 2
    return res


def polynomial_gcd_euclidean(p: List[float], q: List[float]) -> List[float]:
    """Compute monic greatest common divisor of two polynomials."""
    while any(abs(c) > 1e-12 for c in q):
        _, rem = polynomial_divmod(p, q)
        p, q = q, rem
    lead = p[-1]
    return [c / lead for c in p] if abs(lead) > 1e-14 else p


def chebyshev_polynomial_t(n: int) -> List[float]:
    """Coefficients of Chebyshev polynomial of first kind T_n(x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    t0 = [1.0]
    t1 = [0.0, 1.0]
    for _ in range(2, n + 1):
        # 2x * T_{k-1} - T_{k-2}
        two_x_t1 = [0.0] + [2.0 * c for c in t1]
        t_next = polynomial_sub(two_x_t1, t0)
        t0, t1 = t1, t_next
    return t1


def legendre_polynomial_coeffs(n: int) -> List[float]:
    """Coefficients of Legendre polynomial P_n(x) using Bonnet's recurrence."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 1.0]
    p0 = [1.0]
    p1 = [0.0, 1.0]
    for k in range(1, n):
        # (k+1)*P_{k+1} = (2k+1)*x*P_k - k*P_{k-1}
        term1 = [0.0] + [((2.0 * k + 1.0) / (k + 1.0)) * c for c in p1]
        term2 = [(k / (k + 1.0)) * c for c in p0]
        p_next = polynomial_sub(term1, term2)
        p0, p1 = p1, p_next
    return p1


def hermite_polynomial_physicist(n: int) -> List[float]:
    """Coefficients of physicist Hermite polynomial H_n(x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h0 = [1.0]
    h1 = [0.0, 2.0]
    for k in range(1, n):
        # H_{k+1} = 2x H_k - 2k H_{k-1}
        two_x_h1 = [0.0] + [2.0 * c for c in h1]
        two_k_h0 = [2.0 * k * c for c in h0]
        h_next = polynomial_sub(two_x_h1, two_k_h0)
        h0, h1 = h1, h_next
    return h1


def laguerre_polynomial_coeffs(n: int) -> List[float]:
    """Coefficients of standard Laguerre polynomial L_n(x)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return [1.0]
    if n == 1:
        return [1.0, -1.0]
    l0 = [1.0]
    l1 = [1.0, -1.0]
    for k in range(1, n):
        # (k+1)*L_{k+1} = (2k+1 - x)*L_k - k*L_{k-1}
        factor1 = [(2.0 * k + 1.0) / (k + 1.0) * c for c in l1]
        factor_x = [0.0] + [1.0 / (k + 1.0) * c for c in l1]
        factor2 = [(k / (k + 1.0)) * c for c in l0]
        l_next = polynomial_sub(polynomial_sub(factor1, factor_x), factor2)
        l0, l1 = l1, l_next
    return l1


def lagrange_interpolation_poly(xs: List[float], ys: List[float]) -> List[float]:
    """Construct interpolating polynomial P(x) passing through (xs[i], ys[i])."""
    n = len(xs)
    if n != len(ys) or n == 0:
        raise ValueError("xs and ys must have matching non-empty lengths")
    total_poly = [0.0]
    for i in range(n):
        basis = [1.0]
        denom = 1.0
        for j in range(n):
            if i != j:
                basis = polynomial_mul(basis, [-xs[j], 1.0])
                denom *= (xs[i] - xs[j])
        scaled_basis = polynomial_scale(basis, ys[i] / denom)
        total_poly = polynomial_add(total_poly, scaled_basis)
    return total_poly


def newton_divided_differences_poly(xs: List[float], ys: List[float]) -> List[float]:
    """Construct Newton form interpolating polynomial coefficients."""
    n = len(xs)
    table = [[0.0] * n for _ in range(n)]
    for i in range(n):
        table[i][0] = ys[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (xs[i + j] - xs[i])
    # Build polynomial: c0 + c1*(x - x0) + c2*(x - x0)*(x - x1) + ...
    total = [table[0][0]]
    curr_term = [1.0]
    for j in range(1, n):
        curr_term = polynomial_mul(curr_term, [-xs[j - 1], 1.0])
        scaled = polynomial_scale(curr_term, table[0][j])
        total = polynomial_add(total, scaled)
    return total


def faulhaber_sum_powers(p: int, n: int) -> int:
    """Compute sum of p-th powers sum_{k=1}^n k^p."""
    if p < 0 or n < 0:
        raise ValueError("p and n must be non-negative")
    return sum(k ** p for k in range(1, n + 1))


def quaternion_mul(q1: Tuple[float, float, float, float],
                   q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Hamiltonian quaternion product q1 * q2 for (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return w, x, y, z


def quaternion_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Quaternion conjugate (w, -x, -y, -z)."""
    return q[0], -q[1], -q[2], -q[3]


def quaternion_norm(q: Tuple[float, float, float, float]) -> float:
    """Quaternion Euclidean norm |q| = sqrt(w^2 + x^2 + y^2 + z^2)."""
    return math.sqrt(sum(v * v for v in q))


def quaternion_inverse(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Quaternion multiplicative inverse q^{-1} = conjugate(q) / |q|^2."""
    norm_sq = sum(v * v for v in q)
    if norm_sq == 0:
        raise ZeroDivisionError("Cannot invert zero quaternion")
    w, x, y, z = quaternion_conjugate(q)
    return w / norm_sq, x / norm_sq, y / norm_sq, z / norm_sq


def is_semigroup(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if set and operation form a semigroup (associativity and closure)."""
    elem_set = set(elements)
    for a in elements:
        for b in elements:
            if op(a, b) not in elem_set:
                return False
    return is_associative_operation(elements, op)


def is_monoid(elements: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if set and operation form a monoid (semigroup with identity)."""
    return is_semigroup(elements, op) and find_identity_element(elements, op) is not None


def cayley_table(elements: List[Any], op: Callable[[Any, Any], Any]) -> List[List[Any]]:
    """Construct Cayley operation table for finite algebraic structure."""
    return [[op(a, b) for b in elements] for a in elements]


def element_order_in_group(x: Any, elements: List[Any], op: Callable[[Any, Any], Any]) -> int:
    """Find order of element x in finite group (smallest k > 0 such that x^k = e)."""
    e = find_identity_element(elements, op)
    curr = x
    order = 1
    while curr != e and order <= len(elements):
        curr = op(curr, x)
        order += 1
    if curr == e:
        return order
    raise ValueError("Element does not have finite order in given structure")


def subgroup_generated(generator: Any, elements: List[Any], op: Callable[[Any, Any], Any]) -> List[Any]:
    """Return cyclic subgroup generated by generator."""
    e = find_identity_element(elements, op)
    sub = [generator]
    curr = op(generator, generator)
    while curr != generator:
        sub.append(curr)
        curr = op(curr, generator)
    return sub


def coset_left(g: Any, subgroup: List[Any], op: Callable[[Any, Any], Any]) -> List[Any]:
    """Compute left coset g*H."""
    return [op(g, h) for h in subgroup]


def coset_right(g: Any, subgroup: List[Any], op: Callable[[Any, Any], Any]) -> List[Any]:
    """Compute right coset H*g."""
    return [op(h, g) for h in subgroup]


def is_normal_subgroup(subgroup: List[Any], group: List[Any], op: Callable[[Any, Any], Any]) -> bool:
    """Check if subgroup H is normal in group G (g*H == H*g for all g in G)."""
    sub_set = set(subgroup)
    for g in group:
        left = set(coset_left(g, subgroup, op))
        right = set(coset_right(g, subgroup, op))
        if left != right:
            return False
    return True


def solve_quadratic_inequality(a: float, b: float, c: float, op: str) -> str:
    """Solve quadratic inequality a*x^2 + b*x + c <op> 0 for op in '<', '<=', '>', '>='."""
    disc = b * b - 4.0 * a * c
    if disc < 0:
        # No real roots
        positive_everywhere = a > 0
        if op in ('>', '>='):
            return '(-inf, inf)' if positive_everywhere else 'empty'
        else:
            return 'empty' if positive_everywhere else '(-inf, inf)'
    r1 = (-b - math.sqrt(disc)) / (2.0 * a)
    r2 = (-b + math.sqrt(disc)) / (2.0 * a)
    if r1 > r2:
        r1, r2 = r2, r1
    if a > 0:
        if op == '<':
            return f'({r1}, {r2})'
        elif op == '<=':
            return f'[{r1}, {r2}]'
        elif op == '>':
            return f'(-inf, {r1}) U ({r2}, inf)'
        else:
            return f'(-inf, {r1}] U [{r2}, inf)'
    else:
        if op == '<':
            return f'(-inf, {r1}) U ({r2}, inf)'
        elif op == '<=':
            return f'(-inf, {r1}] U [{r2}, inf)'
        elif op == '>':
            return f'({r1}, {r2})'
        else:
            return f'[{r1}, {r2}]'
def quaternion_dot(q1: Tuple[float, float, float, float],
                   q2: Tuple[float, float, float, float]) -> float:
    """Euclidean dot product of two quaternions."""
    return sum(a * b for a, b in zip(q1, q2))


def quaternion_distance(q1: Tuple[float, float, float, float],
                        q2: Tuple[float, float, float, float]) -> float:
    """Euclidean distance between two quaternions."""
    diff = (q1[0]-q2[0], q1[1]-q2[1], q1[2]-q2[2], q1[3]-q2[3])
    return quaternion_norm(diff)


def solve_absolute_value_equation(a: float, b: float, c: float) -> List[float]:
    """Solve |a*x + b| = c for x."""
    if c < 0:
        return []
    if a == 0:
        return [0.0] if abs(b) == c else []
    if c == 0:
        return [-b / a]
    x1 = (c - b) / a
    x2 = (-c - b) / a
    return sorted([x1, x2])


def matrix_commutator_algebra(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Compute Lie bracket / commutator [A, B] = A*B - B*A."""
    n = len(A)
    # AB
    AB = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    # BA
    BA = [[sum(B[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[AB[i][j] - BA[i][j] for j in range(n)] for i in range(n)]


def matrix_anticommutator_algebra(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Compute Jordan product / anticommutator {A, B} = A*B + B*A."""
    n = len(A)
    AB = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    BA = [[sum(B[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[AB[i][j] + BA[i][j] for j in range(n)] for i in range(n)]
