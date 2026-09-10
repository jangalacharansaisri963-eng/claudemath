"""Arithmetic algorithms and discrete number properties for claudemath.

Contains robust, pure-Python implementations of basic and advanced arithmetic,
modular operations, factorials, binomials, means, ratios, rounding, partitions,
and numerical sequence generation.
"""

from typing import List, Tuple, Union, Optional
import math


def add(*numbers: float) -> float:
    """Sum an arbitrary number of values: add(1, 2, 3, ...). No argument limit."""
    if not numbers:
        raise ValueError("add requires at least one argument")
    total = 0
    for n in numbers:
        total += n
    return total


def subtract(*numbers: float) -> float:
    """Subtract sequentially from the first value: subtract(10, 2, 3) = 10 - 2 - 3.
    With a single argument, returns its negation."""
    if not numbers:
        raise ValueError("subtract requires at least one argument")
    if len(numbers) == 1:
        return -numbers[0]
    result = numbers[0]
    for n in numbers[1:]:
        result -= n
    return result


def multiply(*numbers: float) -> float:
    """Multiply an arbitrary number of values: multiply(2, 3, 4, ...). No argument limit."""
    if not numbers:
        raise ValueError("multiply requires at least one argument")
    result = 1
    for n in numbers:
        result *= n
    return result


def divide(*numbers: float) -> float:
    """Divide sequentially from the first value: divide(100, 2, 5) = 100 / 2 / 5.
    With a single argument, returns its reciprocal."""
    if not numbers:
        raise ValueError("divide requires at least one argument")
    if len(numbers) == 1:
        if numbers[0] == 0:
            raise ZeroDivisionError("Cannot take reciprocal of zero")
        return 1 / numbers[0]
    result = numbers[0]
    for n in numbers[1:]:
        if n == 0:
            raise ZeroDivisionError("division by zero")
        result /= n
    return result


def power(base: float, exponent: float) -> float:
    """Raise base to the given exponent: power(2, 10) = 1024."""
    return base ** exponent


def sqrt(x: float) -> float:
    """Compute the principal (real) square root of a non-negative number.

    Raises ValueError for negative input; see complex_numbers for complex roots.
    """
    if x < 0:
        raise ValueError("sqrt of a negative number is not real; use complex_numbers module")
    return math.sqrt(x)


def cube_root(x: float) -> float:
    """Compute the real cube root of x, including negative values."""
    if x < 0:
        return -((-x) ** (1.0 / 3.0))
    return x ** (1.0 / 3.0)


def nth_root(x: float, n: float) -> float:
    """Compute the real n-th root of x. Negative x only supported for odd integer n."""
    if n == 0:
        raise ValueError("Root degree n cannot be zero")
    if x < 0:
        if isinstance(n, int) and n % 2 == 1:
            return -((-x) ** (1.0 / n))
        raise ValueError("Even/non-integer root of a negative number is not real")
    return x ** (1.0 / n)


def absolute_value(x: float) -> float:
    """Return the absolute value of x."""
    return abs(x)


def sign(x: float) -> int:
    """Return the sign of x: -1, 0, or 1."""
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def negate(x: float) -> float:
    """Return the arithmetic negation of x."""
    return -x


def square(x: float) -> float:
    """Return x squared."""
    return x * x


def cube(x: float) -> float:
    """Return x cubed."""
    return x * x * x


def reciprocal(x: float) -> float:
    """Return 1 / x."""
    if x == 0:
        raise ZeroDivisionError("Cannot take reciprocal of zero")
    return 1 / x


def gcd(a: int, b: int) -> int:
    """Compute the greatest common divisor of two integers using Euclidean algorithm."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("gcd requires integer arguments")
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def binary_gcd(a: int, b: int) -> int:
    """Compute GCD using Stein's binary GCD algorithm (shift and subtraction)."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("binary_gcd requires integer arguments")
    a, b = abs(a), abs(b)
    if a == 0:
        return b
    if b == 0:
        return a
    shift = 0
    while ((a | b) & 1) == 0:
        a >>= 1
        b >>= 1
        shift += 1
    while (a & 1) == 0:
        a >>= 1
    while b != 0:
        while (b & 1) == 0:
            b >>= 1
        if a > b:
            a, b = b, a
        b -= a
    return a << shift


def gcd_multiple(numbers: List[int]) -> int:
    """Compute GCD across a list of integers."""
    if not numbers:
        raise ValueError("Cannot compute GCD of an empty list")
    res = abs(numbers[0])
    for num in numbers[1:]:
        res = gcd(res, num)
    return res


def lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("lcm requires integer arguments")
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def lcm_multiple(numbers: List[int]) -> int:
    """Compute LCM across a list of integers."""
    if not numbers:
        raise ValueError("Cannot compute LCM of an empty list")
    res = abs(numbers[0])
    for num in numbers[1:]:
        res = lcm(res, num)
    return res


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm returning (gcd, x, y) such that a*x + b*y = gcd(a, b)."""
    if not isinstance(a, int) or not isinstance(b, int):
        raise TypeError("extended_gcd requires integer arguments")
    if a == 0:
        return (abs(b), 0, 1 if b >= 0 else -1)
    x0, x1, y0, y1 = 1, 0, 0, 1
    orig_a, orig_b = a, b
    a, b = abs(a), abs(b)
    while b != 0:
        q = a // b
        a, b = b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if orig_a < 0:
        x0 = -x0
    if orig_b < 0:
        y0 = -y0
    return a, x0, y0


def integer_sqrt(n: int) -> int:
    """Compute the integer floor square root of non-negative integer n using Heron's method."""
    if not isinstance(n, int):
        raise TypeError("integer_sqrt requires an integer")
    if n < 0:
        raise ValueError("Cannot compute square root of negative integer")
    if n == 0:
        return 0
    x = int(math.isqrt(n)) if hasattr(math, 'isqrt') else int(math.sqrt(n))
    # Refine if needed
    while (x + 1) * (x + 1) <= n:
        x += 1
    while x * x > n:
        x -= 1
    return x


def integer_nth_root(n: int, r: int) -> int:
    """Compute the integer floor r-th root of non-negative integer n."""
    if not isinstance(n, int) or not isinstance(r, int):
        raise TypeError("integer_nth_root requires integer arguments")
    if r <= 0:
        raise ValueError("Root degree r must be positive")
    if n < 0:
        if r % 2 == 1:
            return -integer_nth_root(-n, r)
        raise ValueError("Even root of negative integer not defined over reals")
    if n in (0, 1):
        return n
    # Initial estimate using float power
    x = int(math.pow(n, 1.0 / r))
    if (x + 1) ** r <= n:
        while (x + 1) ** r <= n:
            x += 1
    else:
        while x ** r > n:
            x -= 1
    return x


def is_perfect_square(n: int) -> bool:
    """Check if integer n is a perfect square."""
    if not isinstance(n, int) or n < 0:
        return False
    r = integer_sqrt(n)
    return r * r == n


def is_perfect_cube(n: int) -> bool:
    """Check if integer n is a perfect cube."""
    if not isinstance(n, int):
        return False
    r = integer_nth_root(n, 3)
    return r * r * r == n


def is_perfect_power(n: int) -> Optional[Tuple[int, int]]:
    """Return (base, exp) if n = base^exp for exp >= 2, else None."""
    if not isinstance(n, int) or n <= 1:
        return None
    max_exp = int(math.log2(n)) + 1
    for exp in range(2, max_exp + 1):
        b = integer_nth_root(n, exp)
        if b ** exp == n:
            return b, exp
    return None


def mod_add(a: int, b: int, m: int) -> int:
    """Modular addition: (a + b) mod m."""
    if m <= 0:
        raise ValueError("Modulus must be positive")
    return (a + b) % m


def mod_sub(a: int, b: int, m: int) -> int:
    """Modular subtraction: (a - b) mod m."""
    if m <= 0:
        raise ValueError("Modulus must be positive")
    return (a - b) % m


def mod_mul(a: int, b: int, m: int) -> int:
    """Modular multiplication: (a * b) mod m."""
    if m <= 0:
        raise ValueError("Modulus must be positive")
    return (a * b) % m


def mod_pow(base: int, exp: int, mod: int) -> int:
    """Modular exponentiation: (base^exp) mod mod using binary square-and-multiply."""
    if mod <= 0:
        raise ValueError("Modulus must be positive")
    if exp < 0:
        inv = mod_inv(base, mod)
        return mod_pow(inv, -exp, mod)
    res = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            res = (res * base) % mod
        base = (base * base) % mod
        exp //= 2
    return res


def mod_inv(a: int, m: int) -> int:
    """Modular inverse of a modulo m using extended Euclidean algorithm."""
    if m <= 0:
        raise ValueError("Modulus must be positive")
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"Modular inverse does not exist: gcd({a}, {m}) = {g} != 1")
    return (x % m + m) % m


def mod_div(a: int, b: int, m: int) -> int:
    """Modular division: (a / b) mod m."""
    return mod_mul(a, mod_inv(b, m), m)


def is_divisible(n: int, d: int) -> bool:
    """Return True if n is divisible by d."""
    if d == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return n % d == 0


def is_even(n: int) -> bool:
    """Return True if n is even."""
    return (n & 1) == 0


def is_odd(n: int) -> bool:
    """Return True if n is odd."""
    return (n & 1) == 1


def is_power_of_two(n: int) -> bool:
    """Return True if positive integer n is a power of 2."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def digit_sum(n: int, base: int = 10) -> int:
    """Sum of digits of n in specified base."""
    if base < 2:
        raise ValueError("Base must be at least 2")
    n = abs(n)
    s = 0
    while n > 0:
        s += n % base
        n //= base
    return s


def digital_root(n: int, base: int = 10) -> int:
    """Digital root (iterated digit sum) of n."""
    if base < 2:
        raise ValueError("Base must be at least 2")
    n = abs(n)
    if n == 0:
        return 0
    mod = n % (base - 1)
    return (base - 1) if mod == 0 else mod


def factorial(n: int) -> int:
    """Compute n! for non-negative integer n."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("factorial requires non-negative integer")
    if n <= 1:
        return 1
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def double_factorial(n: int) -> int:
    """Compute n!! (product of integers down to 1 or 2 with step 2)."""
    if not isinstance(n, int) or n < -1:
        raise ValueError("double_factorial requires integer >= -1")
    if n in (-1, 0):
        return 1
    res = 1
    for i in range(n, 0, -2):
        res *= i
    return res


def multifactorial(n: int, k: int) -> int:
    """Compute k-multi-factorial n!^(k)."""
    if n < 0 or k <= 0:
        raise ValueError("multifactorial requires n >= 0 and step k > 0")
    if n == 0:
        return 1
    res = 1
    while n > 0:
        res *= n
        n -= k
    return res


def rising_factorial(x: float, n: int) -> float:
    """Pochhammer symbol x^(n) = x * (x+1) * ... * (x+n-1)."""
    if n < 0:
        raise ValueError("Order n must be non-negative")
    res = 1.0
    for i in range(n):
        res *= (x + i)
    return res


def falling_factorial(x: float, n: int) -> float:
    """Falling factorial (x)_n = x * (x-1) * ... * (x-n+1)."""
    if n < 0:
        raise ValueError("Order n must be non-negative")
    res = 1.0
    for i in range(n):
        res *= (x - i)
    return res


def subfactorial(n: int) -> int:
    """Compute subfactorial !n (number of derangements of n elements)."""
    if n < 0:
        raise ValueError("subfactorial requires non-negative integer")
    if n == 0:
        return 1
    if n == 1:
        return 0
    d0, d1 = 1, 0
    for i in range(2, n + 1):
        d0, d1 = d1, (i - 1) * (d0 + d1)
    return d1


def binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k) = n! / (k! * (n - k)!)."""
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("binomial requires integer arguments")
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    num = 1
    for i in range(1, k + 1):
        num = num * (n - i + 1) // i
    return num


def multinomial(n: int, ks: List[int]) -> int:
    """Multinomial coefficient n! / (k1! * k2! * ... * km!)."""
    if sum(ks) != n:
        raise ValueError("Sum of partition counts must equal n")
    if any(k < 0 for k in ks):
        raise ValueError("Counts must be non-negative")
    res = 1
    rem = n
    for k in ks:
        res *= binomial(rem, k)
        rem -= k
    return res


def arithmetic_mean(values: List[float]) -> float:
    """Compute the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("Cannot compute mean of empty sequence")
    return sum(values) / len(values)


def geometric_mean(values: List[float]) -> float:
    """Compute the geometric mean of positive numbers."""
    if not values:
        raise ValueError("Empty list")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be positive for geometric mean")
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


def harmonic_mean(values: List[float]) -> float:
    """Compute harmonic mean of non-zero positive numbers."""
    if not values:
        raise ValueError("Empty list")
    if any(v <= 0 for v in values):
        raise ValueError("All values must be strictly positive")
    return len(values) / sum(1.0 / v for v in values)


def quadratic_mean(values: List[float]) -> float:
    """Compute quadratic mean (root mean square / RMS)."""
    if not values:
        raise ValueError("Empty list")
    return math.sqrt(sum(v * v for v in values) / len(values))


def generalized_mean(values: List[float], p: float) -> float:
    """Compute generalized/power mean M_p(x) = (1/n * sum(x_i^p))^(1/p)."""
    if not values:
        raise ValueError("Empty list")
    if any(v <= 0 for v in values):
        raise ValueError("Values must be positive")
    if p == 0:
        return geometric_mean(values)
    if p == 1:
        return arithmetic_mean(values)
    if p == -1:
        return harmonic_mean(values)
    if p == 2:
        return quadratic_mean(values)
    mean_pow = sum(v ** p for v in values) / len(values)
    return mean_pow ** (1.0 / p)


def lehmer_mean(values: List[float], p: float) -> float:
    """Lehmer mean: sum(x_i^p) / sum(x_i^(p-1))."""
    if not values:
        raise ValueError("Empty list")
    if any(v <= 0 for v in values):
        raise ValueError("Values must be positive")
    num = sum(v ** p for v in values)
    den = sum(v ** (p - 1) for v in values)
    if den == 0:
        raise ZeroDivisionError("Denominator in Lehmer mean is zero")
    return num / den


def contraharmonic_mean(values: List[float]) -> float:
    """Contraharmonic mean: sum(x_i^2) / sum(x_i)."""
    if not values:
        raise ValueError("Empty list")
    s = sum(values)
    if s == 0:
        raise ZeroDivisionError("Sum of values cannot be zero")
    return sum(v * v for v in values) / s


def logarithmic_mean(x: float, y: float) -> float:
    """Logarithmic mean between two positive numbers x and y."""
    if x <= 0 or y <= 0:
        raise ValueError("Arguments must be positive")
    if x == y:
        return float(x)
    return (y - x) / (math.log(y) - math.log(x))


def heronian_mean(a: float, b: float) -> float:
    """Heronian mean: (a + sqrt(a*b) + b) / 3."""
    if a < 0 or b < 0:
        raise ValueError("Arguments must be non-negative")
    return (a + math.sqrt(a * b) + b) / 3.0


def truncated_mean(values: List[float], discard_fraction: float) -> float:
    """Trimmed / truncated mean discarding symmetric fraction from both ends."""
    if not values:
        raise ValueError("Empty list")
    if not (0.0 <= discard_fraction < 0.5):
        raise ValueError("discard_fraction must be in [0, 0.5)")
    s = sorted(values)
    k = int(len(s) * discard_fraction)
    trimmed = s[k: len(s) - k] if k > 0 else s
    return sum(trimmed) / len(trimmed)


def simplify_ratio(a: int, b: int) -> Tuple[int, int]:
    """Simplify ratio a:b to lowest terms."""
    if b == 0:
        raise ZeroDivisionError("Second element of ratio cannot be zero")
    g = gcd(a, b)
    sign = -1 if b < 0 else 1
    return sign * (a // g), sign * (b // g)


def ratio_to_percentage(a: float, b: float) -> float:
    """Convert ratio a:b to a percentage (a / b * 100)."""
    if b == 0:
        raise ZeroDivisionError("Ratio denominator cannot be zero")
    return (a / b) * 100.0


def percentage_to_ratio(pct: float) -> Tuple[int, int]:
    """Convert percentage value to simplified integer ratio a:b."""
    # Scale by 10000 to keep 4 decimal places
    num = int(round(pct * 10000))
    den = 1000000
    return simplify_ratio(num, den)


def percentage_change(initial: float, final: float) -> float:
    """Compute percentage change from initial to final value."""
    if initial == 0:
        raise ZeroDivisionError("Initial value cannot be zero")
    return ((final - initial) / abs(initial)) * 100.0


def percentage_difference(a: float, b: float) -> float:
    """Compute percentage difference between two values."""
    denom = (abs(a) + abs(b)) / 2.0
    if denom == 0:
        return 0.0
    return (abs(a - b) / denom) * 100.0


def round_half_even(x: float, digits: int = 0) -> float:
    """Round to nearest integer/digit using round-half-to-even rule."""
    factor = 10 ** digits
    return round(x * factor) / factor


def round_half_up(x: float, digits: int = 0) -> float:
    """Round to nearest integer using standard round-half-up rule."""
    factor = 10 ** digits
    scaled = x * factor
    if scaled >= 0:
        rounded = math.floor(scaled + 0.5)
    else:
        rounded = math.ceil(scaled - 0.5)
    return rounded / factor


def round_to_multiple(x: float, multiple: float) -> float:
    """Round x to the nearest multiple of given base."""
    if multiple == 0:
        raise ValueError("Multiple cannot be zero")
    return round(x / multiple) * multiple


def round_sig_figs(x: float, sig_figs: int) -> float:
    """Round float x to a specified number of significant figures."""
    if sig_figs <= 0:
        raise ValueError("Significant figures must be positive")
    if x == 0:
        return 0.0
    magnitude = math.floor(math.log10(abs(x)))
    factor = 10 ** (sig_figs - 1 - magnitude)
    return round(x * factor) / factor


def integer_partitions_count(n: int) -> int:
    """Compute p(n), the number of unrestricted integer partitions of n (Euler pentagonal theorem)."""
    if n < 0:
        return 0
    if n == 0:
        return 1
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        k = 1
        val = 0
        while True:
            # generalized pentagonal numbers: k*(3k-1)//2 and k*(3k+1)//2
            p1 = k * (3 * k - 1) // 2
            p2 = k * (3 * k + 1) // 2
            sign = -1 if (k % 2 == 0) else 1
            if p1 <= i:
                val += sign * dp[i - p1]
            if p2 <= i:
                val += sign * dp[i - p2]
            if p1 > i and p2 > i:
                break
            k += 1
        dp[i] = val
    return dp[n]


def arithmetic_sequence_nth(a1: float, d: float, n: int) -> float:
    """Return the n-th term of an arithmetic sequence: a_n = a1 + (n - 1) * d."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return a1 + (n - 1) * d


def arithmetic_sequence_sum(a1: float, d: float, n: int) -> float:
    """Return sum of first n terms of arithmetic sequence."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return (n / 2.0) * (2 * a1 + (n - 1) * d)


def geometric_sequence_nth(a1: float, r: float, n: int) -> float:
    """Return the n-th term of a geometric sequence: a_n = a1 * r^(n - 1)."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    return a1 * (r ** (n - 1))


def geometric_sequence_sum(a1: float, r: float, n: int) -> float:
    """Return sum of first n terms of geometric sequence."""
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if r == 1.0:
        return a1 * n
    return a1 * (1.0 - r ** n) / (1.0 - r)


def harmonic_number(n: int) -> float:
    """Return n-th harmonic number H_n = sum_{k=1}^n (1 / k)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return sum(1.0 / k for k in range(1, n + 1))


def triangular_number(n: int) -> int:
    """Compute n-th triangular number T_n = n * (n + 1) / 2."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n + 1) // 2


def tetrahedral_number(n: int) -> int:
    """Compute n-th tetrahedral number Te_n = n * (n + 1) * (n + 2) / 6."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (n + 1) * (n + 2) // 6


def pentagonal_number(n: int) -> int:
    """Compute n-th pentagonal number P_n = n * (3n - 1) / 2."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (3 * n - 1) // 2


def hexagonal_number(n: int) -> int:
    """Compute n-th hexagonal number H_n = n * (2n - 1)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return n * (2 * n - 1)


def polygonal_number(s: int, n: int) -> int:
    """Compute n-th s-gonal number."""
    if s < 3:
        raise ValueError("Polygon must have at least 3 sides")
    if n < 0:
        raise ValueError("n must be non-negative")
    return ((s - 2) * n * n - (s - 4) * n) // 2


def catalan_number(n: int) -> int:
    """Compute n-th Catalan number C_n = (1 / (n + 1)) * C(2n, n)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return binomial(2 * n, n) // (n + 1)


def fibonacci(n: int) -> int:
    """Compute n-th Fibonacci number using matrix fast-doubling."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 0:
        # F(-n) = (-1)^(n+1) * F(n)
        return ((-1) ** ((-n) + 1)) * fibonacci(-n)
    if n in (0, 1):
        return n

    def _fib(k: int) -> Tuple[int, int]:
        if k == 0:
            return (0, 1)
        a, b = _fib(k >> 1)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k & 1:
            return (d, c + d)
        return (c, d)

    return _fib(n)[0]


def lucas(n: int) -> int:
    """Compute n-th Lucas number L_n = F(n-1) + F(n+1)."""
    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n == 0:
        return 2
    if n == 1:
        return 1
    if n < 0:
        return ((-1) ** (-n)) * lucas(-n)
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def tribonacci(n: int) -> int:
    """Compute n-th Tribonacci number T_0=0, T_1=1, T_2=1, T_n = T_{n-1}+T_{n-2}+T_{n-3}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0:
        return 0
    if n in (1, 2):
        return 1
    a, b, c = 0, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c


def pell(n: int) -> int:
    """Compute n-th Pell number P_0=0, P_1=1, P_n = 2*P_{n-1} + P_{n-2}."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n in (0, 1):
        return n
    p0, p1 = 0, 1
    for _ in range(2, n + 1):
        p0, p1 = p1, 2 * p1 + p0
    return p1


def collatz_sequence(n: int) -> List[int]:
    """Generate the 3n + 1 Collatz sequence starting at positive integer n."""
    if n <= 0:
        raise ValueError("Collatz sequence requires positive starting integer")
    seq = [n]
    curr = n
    while curr != 1:
        if curr % 2 == 0:
            curr //= 2
        else:
            curr = 3 * curr + 1
        seq.append(curr)
    return seq


def collatz_length(n: int) -> int:
    """Compute total stopping time / trajectory length of n under Collatz map."""
    return len(collatz_sequence(n))


def clamp(val: float, low: float, high: float) -> float:
    """Clamp value val to range [low, high]."""
    if low > high:
        raise ValueError("low bound cannot exceed high bound")
    return max(low, min(val, high))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by parameter t."""
    return a + t * (b - a)


def safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    """Safely divide numerator by denominator, returning fallback if denominator is 0."""
    return fallback if den == 0.0 else num / den


def fma(a: float, b: float, c: float) -> float:
    """Fused multiply-add: compute a * b + c."""
    return a * b + c
