"""Number theory algorithms for claudemath.

Pure-Python implementation of primality testing, sieve algorithms, integer factorization,
arithmetic functions, modular arithmetic, congruences, Diophantine equations,
continued fractions, and special number sequences.
"""

from typing import List, Tuple, Dict, Optional, Generator
import math


# ----------------------------------------------------
# 1. Primality Testing and Sieves
# ----------------------------------------------------

def is_prime(n: int) -> bool:
    """Deterministic 6k +/- 1 trial division primality test."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def is_prime_fermat(n: int, a: int = 2) -> bool:
    """Fermat primality test with base a."""
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if math.gcd(a, n) != 1:
        return False
    return pow(a, n - 1, n) == 1


def is_prime_miller_rabin(n: int, rounds: int = 25) -> bool:
    """Probabilistic Miller-Rabin primality test with deterministic coverage for 64-bit ints."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False

    # Decompose n - 1 = d * 2^s
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # Deterministic bases for n < 2^64
    deterministic_bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in deterministic_bases:
        if a >= n:
            break
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True


def is_prime_solovay_strassen(n: int, a: int = 2) -> bool:
    """Solovay-Strassen primality test with base a."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    g = math.gcd(a, n)
    if g > 1:
        return False
    jac = jacobi_symbol(a, n)
    if jac == 0:
        return False
    euler_crit = pow(a, (n - 1) // 2, n)
    return (jac % n) == euler_crit


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """Generate all prime numbers <= limit using classical sieve."""
    if limit < 2:
        return []
    is_pr = [True] * (limit + 1)
    is_pr[0] = is_pr[1] = False
    for p in range(2, int(math.isqrt(limit)) + 1):
        if is_pr[p]:
            for multiple in range(p * p, limit + 1, p):
                is_pr[multiple] = False
    return [p for p in range(2, limit + 1) if is_pr[p]]


def sieve_of_sundaram(n: int) -> List[int]:
    """Generate all primes <= 2*n + 2 using Sieve of Sundaram."""
    if n < 1:
        return []
    k = (n - 1) // 2
    marked = [False] * (k + 1)
    for i in range(1, k + 1):
        j = i
        while (i + j + 2 * i * j) <= k:
            marked[i + j + 2 * i * j] = True
            j += 1
    primes = [2] if n >= 2 else []
    for i in range(1, k + 1):
        if not marked[i]:
            val = 2 * i + 1
            if val <= 2 * n + 2:
                primes.append(val)
    return primes


def segmented_sieve(low: int, high: int) -> List[int]:
    """Generate primes in range [low, high] using segmented sieve."""
    if high < 2 or low > high:
        return []
    limit = int(math.isqrt(high)) + 1
    base_primes = sieve_of_eratosthenes(limit)
    length = high - low + 1
    is_prime_segment = [True] * length
    for p in base_primes:
        start = max(p * p, ((low + p - 1) // p) * p)
        for multiple in range(start, high + 1, p):
            is_prime_segment[multiple - low] = False
    primes = []
    for i in range(length):
        val = low + i
        if val >= 2 and is_prime_segment[i]:
            primes.append(val)
    return primes


def next_prime(n: int) -> int:
    """Find the smallest prime strictly greater than n."""
    if n < 2:
        return 2
    candidate = n + 1 if n % 2 == 0 else n + 2
    while not is_prime(candidate):
        candidate += 2
    return candidate


def prev_prime(n: int) -> int:
    """Find the largest prime strictly less than n."""
    if n <= 2:
        raise ValueError("No primes exist less than 2")
    if n == 3:
        return 2
    candidate = n - 1 if n % 2 == 0 else n - 2
    while candidate >= 2 and not is_prime(candidate):
        candidate -= 2
    if candidate < 2:
        raise ValueError("No prime found")
    return candidate


def nth_prime(n: int) -> int:
    """Find the n-th prime number (1-indexed: 1st is 2, 2nd is 3)."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if n == 1:
        return 2
    count = 1
    candidate = 3
    while True:
        if is_prime(candidate):
            count += 1
            if count == n:
                return candidate
        candidate += 2


def prime_count_approx(x: float) -> float:
    """Estimate pi(x) using logarithmic integral Li(x) approx x / ln(x)."""
    if x <= 1.0:
        return 0.0
    return x / math.log(x)


def is_mersenne_prime_lucas_lehmer(p: int) -> bool:
    """Test if M_p = 2^p - 1 is prime using Lucas-Lehmer test."""
    if not is_prime(p):
        return False
    if p == 2:
        return True
    m_p = (1 << p) - 1
    s = 4
    for _ in range(p - 2):
        s = (s * s - 2) % m_p
    return s == 0


def is_sophie_germain_prime(p: int) -> bool:
    """Check if prime p is a Sophie Germain prime (2*p + 1 is also prime)."""
    return is_prime(p) and is_prime(2 * p + 1)


def is_safe_prime(p: int) -> bool:
    """Check if p is a safe prime ((p - 1)/2 is also prime)."""
    return is_prime(p) and (p > 2) and is_prime((p - 1) // 2)


def twin_primes_range(start: int, stop: int) -> List[Tuple[int, int]]:
    """Return pairs of twin primes (p, p+2) in [start, stop]."""
    primes = sieve_of_eratosthenes(stop)
    prime_set = set(primes)
    twins = []
    for p in primes:
        if p >= start and (p + 2) in prime_set:
            twins.append((p, p + 2))
    return twins


def cousin_primes_range(start: int, stop: int) -> List[Tuple[int, int]]:
    """Return pairs of cousin primes (p, p+4) in [start, stop]."""
    primes = sieve_of_eratosthenes(stop)
    prime_set = set(primes)
    cousins = []
    for p in primes:
        if p >= start and (p + 4) in prime_set:
            cousins.append((p, p + 4))
    return cousins


def sexy_primes_range(start: int, stop: int) -> List[Tuple[int, int]]:
    """Return pairs of sexy primes (p, p+6) in [start, stop]."""
    primes = sieve_of_eratosthenes(stop)
    prime_set = set(primes)
    sexy = []
    for p in primes:
        if p >= start and (p + 6) in prime_set:
            sexy.append((p, p + 6))
    return sexy


def goldbach_partitions(n: int) -> List[Tuple[int, int]]:
    """Find all prime pairs (p, q) such that p + q = n (n even > 2)."""
    if n <= 2 or n % 2 != 0:
        raise ValueError("Goldbach conjecture applies to even numbers > 2")
    primes = sieve_of_eratosthenes(n)
    prime_set = set(primes)
    pairs = []
    for p in primes:
        if p > n // 2:
            break
        q = n - p
        if q in prime_set:
            pairs.append((p, q))
    return pairs


# ----------------------------------------------------
# 2. Divisibility, Divisors, and Factorization
# ----------------------------------------------------

def prime_factors(n: int) -> List[int]:
    """Return list of prime factors of n in non-decreasing order."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def prime_factor_multiplicities(n: int) -> Dict[int, int]:
    """Return dictionary mapping prime factor -> multiplicity."""
    factors = prime_factors(n)
    res: Dict[int, int] = {}
    for f in factors:
        res[f] = res.get(f, 0) + 1
    return res


def distinct_prime_factors(n: int) -> List[int]:
    """Return sorted list of unique prime factors."""
    return sorted(list(set(prime_factors(n))))


def divisors(n: int) -> List[int]:
    """Return sorted list of all positive divisors of n."""
    if n <= 0:
        raise ValueError("n must be positive")
    res = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            res.append(i)
            if i * i != n:
                res.append(n // i)
        i += 1
    return sorted(res)


def proper_divisors(n: int) -> List[int]:
    """Return list of all proper divisors of n (divisors < n)."""
    divs = divisors(n)
    return divs[:-1] if divs else []


def divisor_count_tau(n: int) -> int:
    """Number of divisors d(n) or tau(n) = prod(e_i + 1)."""
    if n <= 0:
        raise ValueError("n must be positive")
    mults = prime_factor_multiplicities(n)
    count = 1
    for e in mults.values():
        count *= (e + 1)
    return count


def divisor_sum_sigma(n: int) -> int:
    """Sum of divisors sigma_1(n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    mults = prime_factor_multiplicities(n)
    total = 1
    for p, e in mults.items():
        total *= (p ** (e + 1) - 1) // (p - 1)
    return total


def divisor_sum_power_sigma_k(n: int, k: int) -> int:
    """Sum of k-th powers of divisors sigma_k(n) = sum(d^k)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return sum(d ** k for d in divisors(n))


def aliquot_sum(n: int) -> int:
    """Sum of proper divisors of n: s(n) = sigma(n) - n."""
    return divisor_sum_sigma(n) - n


def is_perfect_number(n: int) -> bool:
    """Check if n is a perfect number (aliquot sum equals n)."""
    return n > 1 and aliquot_sum(n) == n


def is_abundant_number(n: int) -> bool:
    """Check if n is abundant (aliquot sum > n)."""
    return aliquot_sum(n) > n


def is_deficient_number(n: int) -> bool:
    """Check if n is deficient (aliquot sum < n)."""
    return aliquot_sum(n) < n


def is_amicable_pair(a: int, b: int) -> bool:
    """Check if (a, b) form an amicable pair."""
    return a != b and aliquot_sum(a) == b and aliquot_sum(b) == a


def greatest_common_divisor(a: int, b: int) -> int:
    """Euclidean algorithm for greatest common divisor gcd(a, b)."""
    while b:
        a, b = b, a % b
    return abs(a)


def least_common_multiple(a: int, b: int) -> int:
    """Least common multiple lcm(a, b) = |a * b| / gcd(a, b)."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // greatest_common_divisor(a, b)


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm returning (gcd, x, y) such that a*x + b*y = gcd."""
    if a == 0:
        return abs(b), 0, (1 if b >= 0 else -1)
    g, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def coprime(a: int, b: int) -> bool:
    """Check if a and b are coprime (gcd = 1)."""
    return greatest_common_divisor(a, b) == 1


def is_squarefree(n: int) -> bool:
    """Check if integer n is squarefree (no repeated prime factors)."""
    mults = prime_factor_multiplicities(n)
    return all(e == 1 for e in mults.values())


# ----------------------------------------------------
# 3. Arithmetic Functions
# ----------------------------------------------------

def euler_totient_phi(n: int) -> int:
    """Euler totient function phi(n) = n * prod(1 - 1/p)."""
    if n <= 0:
        raise ValueError("n must be positive")
    res = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            res -= res // p
        p += 1
    if temp > 1:
        res -= res // temp
    return res


def jordan_totient_j_k(n: int, k: int) -> int:
    """Jordan totient function J_k(n) = n^k * prod_{p|n} (1 - p^{-k})."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    primes = distinct_prime_factors(n)
    res = n ** k
    for p in primes:
        res = res * (p ** k - 1) // (p ** k)
    return res


def carmichael_lambda(n: int) -> int:
    """Carmichael function lambda(n) (smallest m such that a^m = 1 mod n for all coprime a)."""
    if n <= 0:
        raise ValueError("n must be positive")
    mults = prime_factor_multiplicities(n)
    lams = []
    for p, a in mults.items():
        if p == 2:
            if a == 1:
                lams.append(1)
            elif a == 2:
                lams.append(2)
            else:
                lams.append(2 ** (a - 2))
        else:
            lams.append((p - 1) * (p ** (a - 1)))
    res = 1
    for L in lams:
        res = least_common_multiple(res, L)
    return res


def mobius_mu(n: int) -> int:
    """Mobius function mu(n). Returns 1 if n=1, 0 if p^2|n, (-1)^k if squarefree with k factors."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 1
    mults = prime_factor_multiplicities(n)
    if any(e > 1 for e in mults.values()):
        return 0
    return -1 if len(mults) % 2 == 1 else 1


def liouville_lambda(n: int) -> int:
    """Liouville function lambda(n) = (-1)^Omega(n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    omega_big = sum(prime_factor_multiplicities(n).values())
    return -1 if omega_big % 2 == 1 else 1


def mangoldt_lambda(n: int) -> float:
    """Von Mangoldt function Lambda(n) = ln(p) if n = p^k, else 0."""
    if n <= 0:
        raise ValueError("n must be positive")
    mults = prime_factor_multiplicities(n)
    if len(mults) == 1:
        p = list(mults.keys())[0]
        return math.log(p)
    return 0.0


def prime_omega_little(n: int) -> int:
    """Number of distinct prime factors omega(n)."""
    return len(distinct_prime_factors(n))


def prime_omega_big(n: int) -> int:
    """Total number of prime factors with multiplicity Omega(n)."""
    return len(prime_factors(n))


def dedekind_psi(n: int) -> int:
    """Dedekind psi function psi(n) = n * prod_{p|n} (1 + 1/p)."""
    if n <= 0:
        raise ValueError("n must be positive")
    primes = distinct_prime_factors(n)
    res = n
    for p in primes:
        res = res * (p + 1) // p
    return res


# ----------------------------------------------------
# 4. Modular Arithmetic, Congruences, and Symbols
# ----------------------------------------------------

def modular_add(a: int, b: int, m: int) -> int:
    """Modular addition (a + b) mod m."""
    return (a + b) % m


def modular_sub(a: int, b: int, m: int) -> int:
    """Modular subtraction (a - b) mod m."""
    return (a - b) % m


def modular_mul(a: int, b: int, m: int) -> int:
    """Modular multiplication (a * b) mod m."""
    return (a * b) % m


def modular_pow(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation (base^exp) mod m using binary exponentiation."""
    return pow(base, exp, mod)


def modular_inverse(a: int, m: int) -> int:
    """Modular multiplicative inverse a^{-1} mod m via extended Euclidean."""
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse exists: gcd({a}, {m}) = {g} != 1")
    return (x % m + m) % m


def solve_linear_congruence(a: int, b: int, m: int) -> List[int]:
    """Solve linear congruence a*x = b (mod m). Returns all incongruent solutions mod m."""
    g, x0, _ = extended_gcd(a, m)
    if b % g != 0:
        return []
    base_sol = (x0 * (b // g)) % (m // g)
    step = m // g
    return [(base_sol + i * step) % m for i in range(g)]


def chinese_remainder_theorem(remainders: List[int], moduli: List[int]) -> int:
    """Solve system of congruences x = a_i (mod m_i) for pairwise coprime moduli."""
    if len(remainders) != len(moduli) or not moduli:
        raise ValueError("remainders and moduli must have matching non-empty lengths")
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for a_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        inv = modular_inverse(M_i, m_i)
        x = (x + a_i * M_i * inv) % M
    return x


def legendre_symbol(a: int, p: int) -> int:
    """Compute Legendre symbol (a / p) for odd prime p using Euler's criterion."""
    if p <= 2 or not is_prime(p):
        raise ValueError("p must be an odd prime")
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def jacobi_symbol(a: int, n: int) -> int:
    """Compute Jacobi symbol (a / n) for positive odd integer n."""
    if n <= 0 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    a %= n
    result = 1
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if n % 8 in (3, 5):
                result = -result
        a, n = n, a
        if a % 4 == 3 and n % 4 == 3:
            result = -result
        a %= n
    return result if n == 1 else 0


def tonelli_shanks_sqrt(n: int, p: int) -> Optional[int]:
    """Find square root of n mod p (p odd prime) using Tonelli-Shanks algorithm."""
    n %= p
    if n == 0:
        return 0
    if p == 2:
        return n
    if legendre_symbol(n, p) != 1:
        return None
    # Factor p - 1 = Q * 2^S with Q odd
    q = p - 1
    s = 0
    while q % 2 == 0:
        q //= 2
        s += 1
    if s == 1:
        return pow(n, (p + 1) // 4, p)
    # Find quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    c = pow(z, q, p)
    x = pow(n, (q + 1) // 2, p)
    t = pow(n, q, p)
    m = s
    while t % p != 1:
        i = 1
        t2i = (t * t) % p
        while i < m:
            if t2i == 1:
                break
            t2i = (t2i * t2i) % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        x = (x * b) % p
        c = (b * b) % p
        t = (t * c) % p
        m = i
    return x


def discrete_log_baby_step_giant_step(alpha: int, beta: int, p: int) -> Optional[int]:
    """Solve alpha^x = beta (mod p) using Shank's Baby-Step Giant-Step algorithm."""
    alpha %= p
    beta %= p
    m = int(math.isqrt(p)) + 1
    # Baby steps: store alpha^j for j in [0, m-1]
    table: Dict[int, int] = {}
    cur = 1
    for j in range(m):
        table[cur] = j
        cur = (cur * alpha) % p
    # Giant step multiplier: alpha^(-m) mod p
    alpha_inv_m = pow(modular_inverse(alpha, p), m, p)
    cur = beta
    for i in range(m):
        if cur in table:
            return i * m + table[cur]
        cur = (cur * alpha_inv_m) % p
    return None


def order_mod_m(a: int, m: int) -> int:
    """Multiplicative order of a modulo m: smallest k > 0 such that a^k = 1 mod m."""
    if math.gcd(a, m) != 1:
        raise ValueError(f"{a} and {m} must be coprime")
    phi = euler_totient_phi(m)
    for d in divisors(phi):
        if pow(a, d, m) == 1:
            return d
    return phi


def is_primitive_root(g: int, m: int) -> bool:
    """Check if g is a primitive root modulo m (order = phi(m))."""
    if math.gcd(g, m) != 1:
        return False
    return order_mod_m(g, m) == euler_totient_phi(m)


def primitive_root(m: int) -> Optional[int]:
    """Find smallest primitive root modulo m, or None if none exists."""
    if m in (1, 2, 4):
        return 1 if m == 2 else (3 if m == 4 else None)
    phi = euler_totient_phi(m)
    primes = distinct_prime_factors(phi)
    for g in range(2, m):
        if math.gcd(g, m) == 1:
            if all(pow(g, phi // p, m) != 1 for p in primes):
                return g
    return None


# ----------------------------------------------------
# 5. Continued Fractions and Diophantine Equations
# ----------------------------------------------------

def continued_fraction_expansion(numerator: int, denominator: int) -> List[int]:
    """Return partial quotients [a_0; a_1, a_2, ...] of rational numerator / denominator."""
    if denominator == 0:
        raise ZeroDivisionError("Denominator cannot be zero")
    a = []
    while denominator:
        q = numerator // denominator
        a.append(q)
        numerator, denominator = denominator, numerator - q * denominator
    return a


def continued_fraction_to_rational(quotients: List[int]) -> Tuple[int, int]:
    """Convert finite continued fraction quotients back to (numerator, denominator)."""
    if not quotients:
        raise ValueError("quotients cannot be empty")
    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0
    for a in quotients:
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next
    return h_curr, k_curr


def pell_equation_fundamental_solution(D: int) -> Tuple[int, int]:
    """Find minimal positive integer solution (x, y) to Pell's equation x^2 - D*y^2 = 1."""
    if D <= 0:
        raise ValueError("D must be positive")
    sqrt_D = math.isqrt(D)
    if sqrt_D * sqrt_D == D:
        raise ValueError("D cannot be a perfect square")
    m = 0
    d = 1
    a = sqrt_D
    h_prev, h_curr = 1, a
    k_prev, k_curr = 0, 1
    while h_curr * h_curr - D * k_curr * k_curr != 1:
        m = d * a - m
        d = (D - m * m) // d
        a = (sqrt_D + m) // d
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
    return h_curr, k_curr


def solve_linear_diophantine_2var(a: int, b: int, c: int) -> Tuple[int, int]:
    """Find particular solution (x_0, y_0) to a*x + b*y = c, or raise ValueError."""
    g, x, y = extended_gcd(a, b)
    if c % g != 0:
        raise ValueError(f"No integer solutions: gcd({a}, {b}) = {g} does not divide {c}")
    factor = c // g
    return x * factor, y * factor


def pythagorean_triples_primitive(limit: int) -> List[Tuple[int, int, int]]:
    """Generate all primitive Pythagorean triples (a, b, c) with c <= limit via Euclid's formula."""
    triples = []
    m_limit = int(math.isqrt(limit)) + 1
    for m in range(2, m_limit):
        for n in range(1, m):
            if (m - n) % 2 == 1 and coprime(m, n):
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                if c <= limit:
                    triples.append((min(a, b), max(a, b), c))
    return sorted(triples, key=lambda t: t[2])


# ----------------------------------------------------
# 6. Special Sequences and Number Properties
# ----------------------------------------------------

def collatz_sequence(n: int) -> List[int]:
    """Return Collatz trajectory starting at positive integer n."""
    if n <= 0:
        raise ValueError("Collatz sequence defined for positive integers")
    seq = [n]
    while n != 1:
        n = (n // 2) if (n % 2 == 0) else (3 * n + 1)
        seq.append(n)
    return seq


def collatz_length(n: int) -> int:
    """Return length of Collatz trajectory."""
    return len(collatz_sequence(n))


def farey_sequence(n: int) -> List[Tuple[int, int]]:
    """Generate Farey sequence of order n as sorted list of reduced fractions (p, q)."""
    if n <= 0:
        raise ValueError("n must be positive")
    a, b, c, d = 0, 1, 1, n
    seq = [(a, b)]
    while c <= n:
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        seq.append((a, b))
    return seq


def catalan_number(n: int) -> int:
    """Compute n-th Catalan number C_n = (2n)! / ((n+1)! * n!)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return math.comb(2 * n, n) // (n + 1)


def lucas_number(n: int) -> int:
    """Compute n-th Lucas number L_n (L_0 = 2, L_1 = 1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 2
    if n == 1:
        return 1
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def bell_number(n: int) -> int:
    """Compute n-th Bell number B_n using Bell triangle."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 1
    row = [1]
    for i in range(1, n + 1):
        next_row = [row[-1]]
        for j in range(len(row)):
            next_row.append(next_row[-1] + row[j])
        row = next_row
    return row[0]


def stirling_number_second_kind(n: int, k: int) -> int:
    """Stirling numbers of second kind S(n, k): ways to partition n elements into k non-empty sets."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    total = 0
    for j in range(k + 1):
        term = ((-1) ** (k - j)) * math.comb(k, j) * (j ** n)
        total += term
    return total // math.factorial(k)


def is_happy_number(n: int) -> bool:
    """Check if n is a happy number."""
    if n <= 0:
        return False
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1


def is_harshad_number(n: int) -> bool:
    """Check if n is a Harshad (Niven) number (divisible by sum of its digits)."""
    if n <= 0:
        return False
    digit_sum = sum(int(c) for c in str(n))
    return n % digit_sum == 0


def is_armstrong_number(n: int) -> bool:
    """Check if n is an Armstrong (narcissistic) number."""
    if n < 0:
        return False
    s = str(n)
    power = len(s)
    return sum(int(c) ** power for c in s) == n


def is_kaprekar_number(n: int) -> bool:
    """Check if n is a Kaprekar number."""
    if n <= 0:
        return False
    if n == 1:
        return True
    sq = n * n
    s = str(sq)
    d = len(str(n))
    right_str = s[-d:]
    left_str = s[:-d] if len(s) > d else "0"
    right = int(right_str)
    left = int(left_str) if left_str else 0
    return right > 0 and (left + right) == n
def bernoulli_number_exact(n: int) -> Tuple[int, int]:
    """Compute n-th Bernoulli number B_n as exact (numerator, denominator) pair using Akiyama-Tanigawa algorithm."""
    if n < 0:
        raise ValueError("n must be non-negative")
    from fractions import Fraction
    a = [Fraction(1, m + 1) for m in range(n + 1)]
    for j in range(1, n + 1):
        for m in range(n - j + 1):
            a[m] = (m + 1) * (a[m] - a[m + 1])
    return a[0].numerator, a[0].denominator


def euler_number_exact(n: int) -> int:
    """Compute n-th Euler secant number E_n."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n % 2 != 0:
        return 0
    # Zigzag permutation counts
    E = [0] * (n + 1)
    E[0] = 1
    for i in range(1, n + 1):
        total = 0
        for k in range(0, i):
            if (i - k) % 2 == 0:
                total += math.comb(i, k) * E[k]
        E[i] = -total
    return E[n]


def stirling_number_first_kind(n: int, k: int) -> int:
    """Unsigned Stirling numbers of first kind |c(n, k)|: permutations of n with k cycles."""
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0 or k > n:
        return 0
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(i, k) + 1):
            dp[i][j] = dp[i - 1][j - 1] + (i - 1) * dp[i - 1][j]
    return dp[n][k]


def partition_count_p(n: int) -> int:
    """Compute integer partition function p(n) using Euler pentagonal number theorem."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    partitions = [0] * (n + 1)
    partitions[0] = 1
    for i in range(1, n + 1):
        total = 0
        k = 1
        while True:
            pent1 = k * (3 * k - 1) // 2
            pent2 = k * (3 * k + 1) // 2
            sign = 1 if k % 2 == 1 else -1
            if pent1 <= i:
                total += sign * partitions[i - pent1]
            if pent2 <= i:
                total += sign * partitions[i - pent2]
            if pent1 > i and pent2 > i:
                break
            k += 1
        partitions[i] = total
    return partitions[n]


def polygonal_number(s: int, n: int) -> int:
    """Return n-th s-gonal number P(s, n) = (s-2)*n*(n-1)/2 + n."""
    if s < 3 or n < 1:
        raise ValueError("s must be >= 3 and n >= 1")
    return (s - 2) * n * (n - 1) // 2 + n


def is_polygonal_number(s: int, x: int) -> bool:
    """Check if integer x is an s-gonal number."""
    if s < 3 or x < 1:
        return False
    # Solve (s-2)*n^2 - (s-4)*n - 2*x = 0
    a = s - 2
    b = -(s - 4)
    c = -2 * x
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.isqrt(disc)
    if sq * sq != disc:
        return False
    num = -b + sq
    return num % (2 * a) == 0 and (num // (2 * a)) > 0


def sum_of_two_squares(n: int) -> Optional[Tuple[int, int]]:
    """Express positive integer n as a^2 + b^2 if possible."""
    if n < 0:
        return None
    for a in range(int(math.isqrt(n)) + 1):
        b2 = n - a * a
        b = math.isqrt(b2)
        if b * b == b2:
            return a, b
    return None


def sum_of_four_squares(n: int) -> Tuple[int, int, int, int]:
    """Lagrange four-square theorem: represent non-negative integer n as a^2 + b^2 + c^2 + d^2."""
    if n < 0:
        raise ValueError("n must be non-negative")
    max_a = int(math.isqrt(n))
    for a in range(max_a, -1, -1):
        rem1 = n - a * a
        max_b = int(math.isqrt(rem1))
        for b in range(max_b, -1, -1):
            rem2 = rem1 - b * b
            max_c = int(math.isqrt(rem2))
            for c in range(max_c, -1, -1):
                rem3 = rem2 - c * c
                d = math.isqrt(rem3)
                if d * d == rem3:
                    return a, b, c, d
    return 0, 0, 0, 0


def is_automorphic_number(n: int) -> bool:
    """Check if n is an automorphic number (square ends in n)."""
    if n < 0:
        return False
    sq = n * n
    return str(sq).endswith(str(n))


def stern_brocot_path(num: int, den: int) -> str:
    """Return binary path (L/R) to positive fraction num/den in Stern-Brocot tree."""
    if num <= 0 or den <= 0:
        raise ValueError("num and den must be positive")
    path = []
    L_num, L_den = 0, 1
    R_num, R_den = 1, 0
    while True:
        M_num = L_num + R_num
        M_den = L_den + R_den
        if num * M_den == den * M_num:
            break
        if num * M_den < den * M_num:
            path.append('L')
            R_num, R_den = M_num, M_den
        else:
            path.append('R')
            L_num, L_den = M_num, M_den
    return ''.join(path)


def pisano_period(m: int) -> int:
    """Find period of Fibonacci sequence modulo m."""
    if m <= 0:
        raise ValueError("m must be positive")
    a, b = 0, 1
    for i in range(m * m + 1):
        a, b = b, (a + b) % m
        if a == 0 and b == 1:
            return i + 1
    return 1


def aliquot_sequence(n: int, max_steps: int = 50) -> List[int]:
    """Generate aliquot sequence starting from n."""
    seq = [n]
    seen = {n}
    for _ in range(max_steps):
        next_val = aliquot_sum(seq[-1])
        seq.append(next_val)
        if next_val == 0 or next_val in seen:
            break
        seen.add(next_val)
    return seq


def pollard_rho_factor(n: int) -> int:
    """Pollard's rho integer factorization algorithm finding non-trivial factor."""
    if n % 2 == 0:
        return 2
    x = 2
    y = 2
    d = 1
    c = 1
    f = lambda val: (val * val + c) % n
    while d == 1:
        x = f(x)
        y = f(f(y))
        d = math.gcd(abs(x - y), n)
    return d if d != n else n


def pollard_p_minus_1(n: int, b: int = 1000) -> Optional[int]:
    """Pollard's p-1 factorization algorithm."""
    a = 2
    for j in range(2, b + 1):
        a = pow(a, j, n)
        d = math.gcd(a - 1, n)
        if 1 < d < n:
            return d
    return None


def fermat_factorization(n: int) -> Tuple[int, int]:
    """Fermat's factorization method for odd composite n."""
    if n % 2 == 0:
        return 2, n // 2
    a = math.isqrt(n)
    if a * a < n:
        a += 1
    b2 = a * a - n
    while math.isqrt(b2) ** 2 != b2:
        a += 1
        b2 = a * a - n
    b = math.isqrt(b2)
    return a - b, a + b


def kronecker_symbol(a: int, b: int) -> int:
    """Compute Kronecker symbol (a | b) for arbitrary integers."""
    if b == 0:
        return 1 if abs(a) == 1 else 0
    res = 1
    if b < 0:
        b = -b
        if a < 0:
            res = -res
    # Extract factor of 2 from b
    v = 0
    while b % 2 == 0:
        v += 1
        b //= 2
    if v % 2 == 1:
        if a % 8 in (3, 5) or a % 8 in (-3, -5):
            res = -res
        elif a % 2 == 0:
            return 0
    return res * jacobi_symbol(a, b)


def mertens_function(n: int) -> int:
    """Mertens function M(n) = sum_{k=1}^n mu(k)."""
    return sum(mobius_mu(k) for k in range(1, n + 1))


def is_carmichael_number(n: int) -> bool:
    """Check if n is a Carmichael number (Korselt's criterion)."""
    if n < 2 or is_prime(n) or n % 2 == 0:
        return False
    mults = prime_factor_multiplicities(n)
    if any(e > 1 for e in mults.values()):
        return False
    return all((n - 1) % (p - 1) == 0 for p in mults.keys())


def is_sphenic_number(n: int) -> bool:
    """Check if n is a sphenic number (product of exactly 3 distinct primes)."""
    factors = prime_factors(n)
    return len(factors) == 3 and len(set(factors)) == 3


def is_semiprime(n: int) -> bool:
    """Check if n is a semiprime (product of two prime numbers)."""
    return len(prime_factors(n)) == 2


def is_powerful_number(n: int) -> bool:
    """Check if n is powerful (if p|n, then p^2|n)."""
    if n <= 0:
        return False
    if n == 1:
        return True
    mults = prime_factor_multiplicities(n)
    return all(e >= 2 for e in mults.values())


def is_achilles_number(n: int) -> bool:
    """Check if n is an Achilles number (powerful but not a perfect power)."""
    if not is_powerful_number(n):
        return False
    mults = list(prime_factor_multiplicities(n).values())
    g = mults[0]
    for m in mults[1:]:
        g = math.gcd(g, m)
    return g == 1


def digital_root(n: int) -> int:
    """Compute single-digit digital root of positive integer n."""
    if n <= 0:
        raise ValueError("n must be positive")
    return 1 + (n - 1) % 9


def pell_numbers(count: int) -> List[int]:
    """Generate first count Pell numbers P_n (P_0 = 0, P_1 = 1, P_n = 2*P_{n-1} + P_{n-2})."""
    if count <= 0:
        return []
    if count == 1:
        return [0]
    res = [0, 1]
    for _ in range(2, count):
        res.append(2 * res[-1] + res[-2])
    return res


def perrin_number(n: int) -> int:
    """Compute n-th Perrin number (P_0=3, P_1=0, P_2=2, P_n = P_{n-2} + P_{n-3})."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 3
    if n == 1:
        return 0
    if n == 2:
        return 2
    p0, p1, p2 = 3, 0, 2
    for _ in range(3, n + 1):
        p_next = p1 + p0
        p0, p1, p2 = p1, p2, p_next
    return p2


def padovan_number(n: int) -> int:
    """Compute n-th Padovan number (P_0=1, P_1=1, P_2=1, P_n = P_{n-2} + P_{n-3})."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 2:
        return 1
    p0, p1, p2 = 1, 1, 1
    for _ in range(3, n + 1):
        p_next = p1 + p0
        p0, p1, p2 = p1, p2, p_next
    return p2


def tribonacci_number(n: int) -> int:
    """Compute n-th Tribonacci number (T_0=0, T_1=0, T_2=1, T_n = T_{n-1}+T_{n-2}+T_{n-3})."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 0
    if n == 2:
        return 1
    t0, t1, t2 = 0, 0, 1
    for _ in range(3, n + 1):
        t0, t1, t2 = t1, t2, t0 + t1 + t2
    return t2
