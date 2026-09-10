"""Trigonometry module for claudemath.

Comprehensive pure-Python implementations of trigonometric, inverse trigonometric,
hyperbolic, versed, cardinal, triangle-solving, series-approximation, and spherical
trigonometry algorithms.
"""

from typing import Tuple, List, Optional
import math

# Fundamental Constants
PI: float = 3.14159265358979323846
TWO_PI: float = 6.28318530717958647692
HALF_PI: float = 1.57079632679489661923
DEG_TO_RAD_FACTOR: float = PI / 180.0
RAD_TO_DEG_FACTOR: float = 180.0 / PI


# ----------------------------------------------------
# 1. Angle Conversions
# ----------------------------------------------------

def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * DEG_TO_RAD_FACTOR


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * RAD_TO_DEG_FACTOR


def grad_to_rad(grad: float) -> float:
    """Convert gradians (gons) to radians."""
    return grad * (PI / 200.0)


def rad_to_grad(rad: float) -> float:
    """Convert radians to gradians (gons)."""
    return rad * (200.0 / PI)


def deg_to_grad(deg: float) -> float:
    """Convert degrees to gradians."""
    return deg * (200.0 / 180.0)


def grad_to_deg(grad: float) -> float:
    """Convert gradians to degrees."""
    return grad * (180.0 / 200.0)


def turns_to_rad(turns: float) -> float:
    """Convert full turns/revolutions to radians."""
    return turns * TWO_PI


def rad_to_turns(rad: float) -> float:
    """Convert radians to full turns/revolutions."""
    return rad / TWO_PI


def arcmin_to_rad(arcmin: float) -> float:
    """Convert minutes of arc to radians."""
    return (arcmin / 60.0) * DEG_TO_RAD_FACTOR


def arcsec_to_rad(arcsec: float) -> float:
    """Convert seconds of arc to radians."""
    return (arcsec / 3600.0) * DEG_TO_RAD_FACTOR


def rad_to_arcmin(rad: float) -> float:
    """Convert radians to minutes of arc."""
    return rad_to_deg(rad) * 60.0


def rad_to_arcsec(rad: float) -> float:
    """Convert radians to seconds of arc."""
    return rad_to_deg(rad) * 3600.0


def dms_to_deg(degrees: float, minutes: float, seconds: float) -> float:
    """Convert degrees, minutes, seconds into decimal degrees."""
    sign = -1.0 if degrees < 0 else 1.0
    return sign * (abs(degrees) + minutes / 60.0 + seconds / 3600.0)


def deg_to_dms(deg: float) -> Tuple[int, int, float]:
    """Convert decimal degrees to (degrees, minutes, seconds)."""
    sign = -1 if deg < 0 else 1
    total = abs(deg)
    d = int(math.floor(total))
    rem = (total - d) * 60.0
    m = int(math.floor(rem))
    s = (rem - m) * 60.0
    return sign * d, m, s


def dms_to_rad(degrees: float, minutes: float, seconds: float) -> float:
    """Convert degrees, minutes, seconds directly to radians."""
    return deg_to_rad(dms_to_deg(degrees, minutes, seconds))


def rad_to_dms(rad: float) -> Tuple[int, int, float]:
    """Convert radians directly to (degrees, minutes, seconds)."""
    return deg_to_dms(rad_to_deg(rad))


# ----------------------------------------------------
# 2. Angle Normalization
# ----------------------------------------------------

def normalize_angle_2pi(rad: float) -> float:
    """Normalize angle to [0, 2*pi)."""
    return rad % TWO_PI


def normalize_angle_pi(rad: float) -> float:
    """Normalize angle to [-pi, pi)."""
    val = (rad + PI) % TWO_PI
    return val - PI


def normalize_angle_360(deg: float) -> float:
    """Normalize angle to [0, 360) degrees."""
    return deg % 360.0


def normalize_angle_180(deg: float) -> float:
    """Normalize angle to [-180, 180) degrees."""
    val = (deg + 180.0) % 360.0
    return val - 180.0


# ----------------------------------------------------
# 3. Core Trigonometric Functions
# ----------------------------------------------------

def sin(x: float) -> float:
    """Compute sine of angle in radians using high-precision range reduction and Taylor series."""
    # Normalize to [-pi, pi]
    y = normalize_angle_pi(x)
    # 12-term Taylor expansion around 0
    term = y
    res = term
    y2 = y * y
    for i in range(1, 12):
        term *= -y2 / ((2 * i) * (2 * i + 1))
        res += term
        if abs(term) < 1e-16:
            break
    return res


def cos(x: float) -> float:
    """Compute cosine of angle in radians using Taylor expansion."""
    y = normalize_angle_pi(x)
    term = 1.0
    res = 1.0
    y2 = y * y
    for i in range(1, 12):
        term *= -y2 / ((2 * i - 1) * (2 * i))
        res += term
        if abs(term) < 1e-16:
            break
    return res


def tan(x: float) -> float:
    """Compute tangent: sin(x) / cos(x)."""
    c = cos(x)
    if abs(c) < 1e-15:
        raise ValueError("Tangent undefined for angle where cos(x) == 0")
    return sin(x) / c


def sin_deg(x: float) -> float:
    """Compute sine of angle given in degrees."""
    return sin(deg_to_rad(x))


def cos_deg(x: float) -> float:
    """Compute cosine of angle given in degrees."""
    return cos(deg_to_rad(x))


def tan_deg(x: float) -> float:
    """Compute tangent of angle given in degrees."""
    return tan(deg_to_rad(x))


def cot(x: float) -> float:
    """Compute cotangent: cos(x) / sin(x)."""
    s = sin(x)
    if abs(s) < 1e-15:
        raise ValueError("Cotangent undefined for angle where sin(x) == 0")
    return cos(x) / s


def sec(x: float) -> float:
    """Compute secant: 1 / cos(x)."""
    c = cos(x)
    if abs(c) < 1e-15:
        raise ValueError("Secant undefined where cos(x) == 0")
    return 1.0 / c


def csc(x: float) -> float:
    """Compute cosecant: 1 / sin(x)."""
    s = sin(x)
    if abs(s) < 1e-15:
        raise ValueError("Cosecant undefined where sin(x) == 0")
    return 1.0 / s


# ----------------------------------------------------
# 4. Inverse Trigonometric Functions
# ----------------------------------------------------

def asin(x: float) -> float:
    """Compute arcsine of x in [-1, 1] returning radians in [-pi/2, pi/2]."""
    if x < -1.0 or x > 1.0:
        raise ValueError(f"asin argument {x} must be within [-1, 1]")
    if x == 1.0:
        return HALF_PI
    if x == -1.0:
        return -HALF_PI
    # atan(x / sqrt(1 - x^2))
    return atan(x / math.sqrt(1.0 - x * x))


def acos(x: float) -> float:
    """Compute arccosine of x in [-1, 1] returning radians in [0, pi]."""
    if x < -1.0 or x > 1.0:
        raise ValueError(f"acos argument {x} must be within [-1, 1]")
    return HALF_PI - asin(x)


def atan(x: float) -> float:
    """Compute arctangent of x returning radians in (-pi/2, pi/2)."""
    sign = 1.0
    if x < 0:
        sign = -1.0
        x = -x

    if x > 1.0:
        return sign * (HALF_PI - atan(1.0 / x))

    # Argument reduction: atan(x) = 2 * atan(x / (1 + sqrt(1 + x^2))).
    # Repeat until x is small enough for the Maclaurin series below to
    # converge quickly and accurately.
    scale = 1.0
    while x > 0.1:
        x = x / (1.0 + math.sqrt(1.0 + x * x))
        scale *= 2.0

    # Maclaurin series: atan(x) = x - x^3/3 + x^5/5 - x^7/7 + ...
    term = x
    res = x
    x2 = x * x
    k = 1
    while True:
        term *= -x2
        delta = term / (2 * k + 1)
        res += delta
        if abs(delta) < 1e-17:
            break
        k += 1

    return sign * scale * res


def atan2(y: float, x: float) -> float:
    """Compute quadrant-aware atan2(y, x)."""
    if x > 0:
        return atan(y / x)
    elif x < 0 and y >= 0:
        return atan(y / x) + PI
    elif x < 0 and y < 0:
        return atan(y / x) - PI
    elif x == 0 and y > 0:
        return HALF_PI
    elif x == 0 and y < 0:
        return -HALF_PI
    return 0.0


def acot(x: float) -> float:
    """Compute arccotangent: atan(1/x)."""
    if x == 0:
        return HALF_PI
    return atan(1.0 / x)


def asec(x: float) -> float:
    """Compute arcsecant: acos(1/x)."""
    if abs(x) < 1.0:
        raise ValueError("asec domain is |x| >= 1")
    return acos(1.0 / x)


def acsc(x: float) -> float:
    """Compute arccosecant: asin(1/x)."""
    if abs(x) < 1.0:
        raise ValueError("acsc domain is |x| >= 1")
    return asin(1.0 / x)


# ----------------------------------------------------
# 5. Hyperbolic Functions
# ----------------------------------------------------

def sinh(x: float) -> float:
    """Compute hyperbolic sine: (e^x - e^(-x)) / 2."""
    if abs(x) < 1e-4:
        # Taylor expansion to avoid catastrophic cancellation
        return x + (x ** 3) / 6.0 + (x ** 5) / 120.0
    ep = math.exp(x)
    return (ep - 1.0 / ep) * 0.5


def cosh(x: float) -> float:
    """Compute hyperbolic cosine: (e^x + e^(-x)) / 2."""
    ep = math.exp(x)
    return (ep + 1.0 / ep) * 0.5


def tanh(x: float) -> float:
    """Compute hyperbolic tangent: sinh(x) / cosh(x)."""
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    ep2 = math.exp(2.0 * x)
    return (ep2 - 1.0) / (ep2 + 1.0)


def coth(x: float) -> float:
    """Compute hyperbolic cotangent: cosh(x) / sinh(x)."""
    if x == 0:
        raise ZeroDivisionError("coth undefined at 0")
    return 1.0 / tanh(x)


def sech(x: float) -> float:
    """Compute hyperbolic secant: 1 / cosh(x)."""
    return 1.0 / cosh(x)


def csch(x: float) -> float:
    """Compute hyperbolic cosecant: 1 / sinh(x)."""
    s = sinh(x)
    if s == 0:
        raise ZeroDivisionError("csch undefined at 0")
    return 1.0 / s


# ----------------------------------------------------
# 6. Inverse Hyperbolic Functions
# ----------------------------------------------------

def asinh(x: float) -> float:
    """Compute inverse hyperbolic sine: ln(x + sqrt(x^2 + 1))."""
    return math.log(x + math.sqrt(x * x + 1.0))


def acosh(x: float) -> float:
    """Compute inverse hyperbolic cosine: ln(x + sqrt(x^2 - 1)) for x >= 1."""
    if x < 1.0:
        raise ValueError("acosh requires x >= 1")
    return math.log(x + math.sqrt(x * x - 1.0))


def atanh(x: float) -> float:
    """Compute inverse hyperbolic tangent: 0.5 * ln((1 + x) / (1 - x)) for |x| < 1."""
    if abs(x) >= 1.0:
        raise ValueError("atanh requires -1 < x < 1")
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


def acoth(x: float) -> float:
    """Compute inverse hyperbolic cotangent: 0.5 * ln((x + 1) / (x - 1)) for |x| > 1."""
    if abs(x) <= 1.0:
        raise ValueError("acoth requires |x| > 1")
    return 0.5 * math.log((x + 1.0) / (x - 1.0))


def asech(x: float) -> float:
    """Compute inverse hyperbolic secant: acosh(1 / x) for 0 < x <= 1."""
    if x <= 0.0 or x > 1.0:
        raise ValueError("asech requires 0 < x <= 1")
    return acosh(1.0 / x)


def acsch(x: float) -> float:
    """Compute inverse hyperbolic cosecant: asinh(1 / x) for x != 0."""
    if x == 0.0:
        raise ZeroDivisionError("acsch undefined at 0")
    return asinh(1.0 / x)


# ----------------------------------------------------
# 7. Historical and Navigation (Versed) Functions
# ----------------------------------------------------

def versine(x: float) -> float:
    """Compute versine: 1 - cos(x)."""
    return 1.0 - cos(x)


def coversine(x: float) -> float:
    """Compute coversine: 1 - sin(x)."""
    return 1.0 - sin(x)


def vercosine(x: float) -> float:
    """Compute vercosine: 1 + cos(x)."""
    return 1.0 + cos(x)


def covercosine(x: float) -> float:
    """Compute covercosine: 1 + sin(x)."""
    return 1.0 + sin(x)


def haversine(x: float) -> float:
    """Compute haversine: sin^2(x / 2) = (1 - cos(x)) / 2."""
    s = sin(x * 0.5)
    return s * s


def archaversine(h: float) -> float:
    """Compute inverse haversine: 2 * asin(sqrt(h)) for 0 <= h <= 1."""
    if h < 0.0 or h > 1.0:
        raise ValueError("archaversine requires 0 <= h <= 1")
    return 2.0 * asin(math.sqrt(h))


def exsecant(x: float) -> float:
    """Compute exsecant: sec(x) - 1."""
    return sec(x) - 1.0


def excosecant(x: float) -> float:
    """Compute excosecant: csc(x) - 1."""
    return csc(x) - 1.0


def chord(theta: float, radius: float = 1.0) -> float:
    """Compute chord length for subtended angle theta and radius r: 2 * r * sin(theta / 2)."""
    return 2.0 * radius * sin(theta * 0.5)


def sagitta(theta: float, radius: float = 1.0) -> float:
    """Compute sagitta (height of circular arc): r * (1 - cos(theta / 2))."""
    return radius * versine(theta * 0.5)


def apothem(theta: float, radius: float = 1.0) -> float:
    """Compute apothem of circular sector: r * cos(theta / 2)."""
    return radius * cos(theta * 0.5)


# ----------------------------------------------------
# 8. Sinc and Cardinal Trigonometric Functions
# ----------------------------------------------------

def sinc(x: float) -> float:
    """Unnormalized cardinal sine: sin(x) / x (with sinc(0) = 1)."""
    if abs(x) < 1e-12:
        return 1.0 - (x * x) / 6.0
    return sin(x) / x


def normalized_sinc(x: float) -> float:
    """Normalized sinc function: sin(pi * x) / (pi * x)."""
    return sinc(PI * x)


def cosc(x: float) -> float:
    """Cardinal cosine derivative function: (x*cos(x) - sin(x)) / x^2."""
    if abs(x) < 1e-8:
        return -x / 3.0
    return (x * cos(x) - sin(x)) / (x * x)


def tanc(x: float) -> float:
    """Cardinal tangent: tan(x) / x."""
    if abs(x) < 1e-12:
        return 1.0 + (x * x) / 3.0
    return tan(x) / x


def sinhc(x: float) -> float:
    """Hyperbolic cardinal sine: sinh(x) / x."""
    if abs(x) < 1e-12:
        return 1.0 + (x * x) / 6.0
    return sinh(x) / x


def coshc(x: float) -> float:
    """Hyperbolic cardinal cosine: cosh(x) / x."""
    if x == 0:
        raise ZeroDivisionError("coshc undefined at 0")
    return cosh(x) / x


# ----------------------------------------------------
# 9. Angle Sum and Difference Formulas
# ----------------------------------------------------

def sin_sum(alpha: float, beta: float) -> float:
    """Compute sin(alpha + beta) = sin(alpha)*cos(beta) + cos(alpha)*sin(beta)."""
    return sin(alpha) * cos(beta) + cos(alpha) * sin(beta)


def cos_sum(alpha: float, beta: float) -> float:
    """Compute cos(alpha + beta) = cos(alpha)*cos(beta) - sin(alpha)*sin(beta)."""
    return cos(alpha) * cos(beta) - sin(alpha) * sin(beta)


def tan_sum(alpha: float, beta: float) -> float:
    """Compute tan(alpha + beta) = (tan(alpha) + tan(beta)) / (1 - tan(alpha)*tan(beta))."""
    ta, tb = tan(alpha), tan(beta)
    denom = 1.0 - ta * tb
    if abs(denom) < 1e-15:
        raise ValueError("tan_sum undefined (denominator is zero)")
    return (ta + tb) / denom


def cot_sum(alpha: float, beta: float) -> float:
    """Compute cot(alpha + beta) = (cot(alpha)*cot(beta) - 1) / (cot(alpha) + cot(beta))."""
    ca, cb = cot(alpha), cot(beta)
    denom = ca + cb
    if abs(denom) < 1e-15:
        raise ValueError("cot_sum undefined (denominator is zero)")
    return (ca * cb - 1.0) / denom


def sin_diff(alpha: float, beta: float) -> float:
    """Compute sin(alpha - beta) = sin(alpha)*cos(beta) - cos(alpha)*sin(beta)."""
    return sin(alpha) * cos(beta) - cos(alpha) * sin(beta)


def cos_diff(alpha: float, beta: float) -> float:
    """Compute cos(alpha - beta) = cos(alpha)*cos(beta) + sin(alpha)*sin(beta)."""
    return cos(alpha) * cos(beta) + sin(alpha) * sin(beta)


def tan_diff(alpha: float, beta: float) -> float:
    """Compute tan(alpha - beta) = (tan(alpha) - tan(beta)) / (1 + tan(alpha)*tan(beta))."""
    ta, tb = tan(alpha), tan(beta)
    denom = 1.0 + ta * tb
    if abs(denom) < 1e-15:
        raise ValueError("tan_diff undefined (denominator is zero)")
    return (ta - tb) / denom


def cot_diff(alpha: float, beta: float) -> float:
    """Compute cot(alpha - beta) = (cot(alpha)*cot(beta) + 1) / (cot(beta) - cot(alpha))."""
    ca, cb = cot(alpha), cot(beta)
    denom = cb - ca
    if abs(denom) < 1e-15:
        raise ValueError("cot_diff undefined (denominator is zero)")
    return (ca * cb + 1.0) / denom


# ----------------------------------------------------
# 10. Multiple-Angle and Half-Angle Formulas
# ----------------------------------------------------

def sin_double_angle(theta: float) -> float:
    """Compute sin(2*theta) = 2*sin(theta)*cos(theta)."""
    return 2.0 * sin(theta) * cos(theta)


def cos_double_angle(theta: float) -> float:
    """Compute cos(2*theta) = cos^2(theta) - sin^2(theta)."""
    c = cos(theta)
    s = sin(theta)
    return c * c - s * s


def tan_double_angle(theta: float) -> float:
    """Compute tan(2*theta) = 2*tan(theta) / (1 - tan^2(theta))."""
    t = tan(theta)
    denom = 1.0 - t * t
    if abs(denom) < 1e-15:
        raise ValueError("tan_double_angle undefined")
    return (2.0 * t) / denom


def cot_double_angle(theta: float) -> float:
    """Compute cot(2*theta) = (cot^2(theta) - 1) / (2*cot(theta))."""
    ct = cot(theta)
    return (ct * ct - 1.0) / (2.0 * ct)


def sin_triple_angle(theta: float) -> float:
    """Compute sin(3*theta) = 3*sin(theta) - 4*sin^3(theta)."""
    s = sin(theta)
    return 3.0 * s - 4.0 * (s ** 3)


def cos_triple_angle(theta: float) -> float:
    """Compute cos(3*theta) = 4*cos^3(theta) - 3*cos(theta)."""
    c = cos(theta)
    return 4.0 * (c ** 3) - 3.0 * c


def tan_triple_angle(theta: float) -> float:
    """Compute tan(3*theta) = (3*tan(theta) - tan^3(theta)) / (1 - 3*tan^2(theta))."""
    t = tan(theta)
    return (3.0 * t - t ** 3) / (1.0 - 3.0 * t * t)


def sin_half_angle(theta: float) -> float:
    """Compute sin(theta / 2). Sign matches quadrant of theta/2."""
    norm = normalize_angle_pi(theta * 0.5)
    sign = 1.0 if norm >= 0 else -1.0
    return sign * math.sqrt(max(0.0, (1.0 - cos(theta)) * 0.5))


def cos_half_angle(theta: float) -> float:
    """Compute cos(theta / 2). Sign matches quadrant of theta/2."""
    norm = normalize_angle_pi(theta * 0.5)
    sign = 1.0 if abs(norm) <= HALF_PI else -1.0
    return sign * math.sqrt(max(0.0, (1.0 + cos(theta)) * 0.5))


def tan_half_angle(theta: float) -> float:
    """Compute tan(theta / 2) = sin(theta) / (1 + cos(theta))."""
    c = cos(theta)
    if abs(1.0 + c) < 1e-15:
        raise ValueError("tan_half_angle undefined")
    return sin(theta) / (1.0 + c)


def cot_half_angle(theta: float) -> float:
    """Compute cot(theta / 2) = (1 + cos(theta)) / sin(theta)."""
    s = sin(theta)
    if abs(s) < 1e-15:
        raise ValueError("cot_half_angle undefined")
    return (1.0 + cos(theta)) / s


# ----------------------------------------------------
# 11. Product-to-Sum and Sum-to-Product
# ----------------------------------------------------

def product_to_sum_sin_sin(alpha: float, beta: float) -> float:
    """sin(a)*sin(b) = (cos(a - b) - cos(a + b)) / 2."""
    return (cos(alpha - beta) - cos(alpha + beta)) * 0.5


def product_to_sum_cos_cos(alpha: float, beta: float) -> float:
    """cos(a)*cos(b) = (cos(a - b) + cos(a + b)) / 2."""
    return (cos(alpha - beta) + cos(alpha + beta)) * 0.5


def product_to_sum_sin_cos(alpha: float, beta: float) -> float:
    """sin(a)*cos(b) = (sin(a + b) + sin(a - b)) / 2."""
    return (sin(alpha + beta) + sin(alpha - beta)) * 0.5


def sum_to_product_sin_plus_sin(alpha: float, beta: float) -> float:
    """sin(a) + sin(b) = 2*sin((a+b)/2)*cos((a-b)/2)."""
    return 2.0 * sin((alpha + beta) * 0.5) * cos((alpha - beta) * 0.5)


def sum_to_product_sin_minus_sin(alpha: float, beta: float) -> float:
    """sin(a) - sin(b) = 2*cos((a+b)/2)*sin((a-b)/2)."""
    return 2.0 * cos((alpha + beta) * 0.5) * sin((alpha - beta) * 0.5)


def sum_to_product_cos_plus_cos(alpha: float, beta: float) -> float:
    """cos(a) + cos(b) = 2*cos((a+b)/2)*cos((a-b)/2)."""
    return 2.0 * cos((alpha + beta) * 0.5) * cos((alpha - beta) * 0.5)


def sum_to_product_cos_minus_cos(alpha: float, beta: float) -> float:
    """cos(a) - cos(b) = -2*sin((a+b)/2)*sin((a-b)/2)."""
    return -2.0 * sin((alpha + beta) * 0.5) * sin((alpha - beta) * 0.5)


# ----------------------------------------------------
# 12. Hyperbolic Identities
# ----------------------------------------------------

def sinh_sum(u: float, v: float) -> float:
    """sinh(u + v) = sinh(u)*cosh(v) + cosh(u)*sinh(v)."""
    return sinh(u) * cosh(v) + cosh(u) * sinh(v)


def cosh_sum(u: float, v: float) -> float:
    """cosh(u + v) = cosh(u)*cosh(v) + sinh(u)*sinh(v)."""
    return cosh(u) * cosh(v) + sinh(u) * sinh(v)


def tanh_sum(u: float, v: float) -> float:
    """tanh(u + v) = (tanh(u) + tanh(v)) / (1 + tanh(u)*tanh(v))."""
    tu, tv = tanh(u), tanh(v)
    return (tu + tv) / (1.0 + tu * tv)


def sinh_diff(u: float, v: float) -> float:
    """sinh(u - v) = sinh(u)*cosh(v) - cosh(u)*sinh(v)."""
    return sinh(u) * cosh(v) - cosh(u) * sinh(v)


def cosh_diff(u: float, v: float) -> float:
    """cosh(u - v) = cosh(u)*cosh(v) - sinh(u)*sinh(v)."""
    return cosh(u) * cosh(v) - sinh(u) * sinh(v)


def tanh_diff(u: float, v: float) -> float:
    """tanh(u - v) = (tanh(u) - tanh(v)) / (1 - tanh(u)*tanh(v))."""
    tu, tv = tanh(u), tanh(v)
    return (tu - tv) / (1.0 - tu * tv)


def sinh_double(u: float) -> float:
    """sinh(2*u) = 2*sinh(u)*cosh(u)."""
    return 2.0 * sinh(u) * cosh(u)


def cosh_double(u: float) -> float:
    """cosh(2*u) = cosh^2(u) + sinh^2(u)."""
    c, s = cosh(u), sinh(u)
    return c * c + s * s


def tanh_double(u: float) -> float:
    """tanh(2*u) = 2*tanh(u) / (1 + tanh^2(u))."""
    t = tanh(u)
    return (2.0 * t) / (1.0 + t * t)


def sinh_half(u: float) -> float:
    """sinh(u / 2) = sgn(u) * sqrt((cosh(u) - 1) / 2)."""
    sign = 1.0 if u >= 0 else -1.0
    return sign * math.sqrt(max(0.0, (cosh(u) - 1.0) * 0.5))


def cosh_half(u: float) -> float:
    """cosh(u / 2) = sqrt((cosh(u) + 1) / 2)."""
    return math.sqrt((cosh(u) + 1.0) * 0.5)


def tanh_half(u: float) -> float:
    """tanh(u / 2) = sinh(u) / (cosh(u) + 1)."""
    return sinh(u) / (cosh(u) + 1.0)


# ----------------------------------------------------
# 13. Triangle Solving and Trigonometric Laws
# ----------------------------------------------------

def law_of_sines_side(side_a: float, angle_a: float, angle_b: float) -> float:
    """Compute side b given side a and angles A, B (in radians): b = a * sin(B) / sin(A)."""
    sa = sin(angle_a)
    if abs(sa) < 1e-15:
        raise ValueError("sin(angle_a) is zero")
    return side_a * sin(angle_b) / sa


def law_of_sines_angle(side_a: float, side_b: float, angle_a: float) -> float:
    """Compute angle B in radians given sides a, b and angle A: B = asin(b * sin(A) / a)."""
    if side_a <= 0:
        raise ValueError("side_a must be positive")
    ratio = (side_b * sin(angle_a)) / side_a
    if abs(ratio) > 1.0:
        raise ValueError("No triangle exists with given parameters")
    return asin(ratio)


def law_of_cosines_side(side_b: float, side_c: float, angle_a: float) -> float:
    """Compute side a given sides b, c and angle A: a = sqrt(b^2 + c^2 - 2*b*c*cos(A))."""
    val = side_b ** 2 + side_c ** 2 - 2.0 * side_b * side_c * cos(angle_a)
    if val < 0:
        raise ValueError("Negative radicand in law of cosines")
    return math.sqrt(val)


def law_of_cosines_angle(side_a: float, side_b: float, side_c: float) -> float:
    """Compute angle A in radians given sides a, b, c: cos(A) = (b^2 + c^2 - a^2) / (2*b*c)."""
    denom = 2.0 * side_b * side_c
    if denom == 0:
        raise ValueError("Sides b and c must be non-zero")
    cos_val = (side_b ** 2 + side_c ** 2 - side_a ** 2) / denom
    cos_val = max(-1.0, min(1.0, cos_val))
    return acos(cos_val)


def law_of_tangents(side_a: float, side_b: float, angle_a: float, angle_b: float) -> float:
    """Verify law of tangents ratio: (a - b)/(a + b) == tan((A-B)/2) / tan((A+B)/2). Returns diff."""
    lhs = (side_a - side_b) / (side_a + side_b)
    rhs = tan((angle_a - angle_b) * 0.5) / tan((angle_a + angle_b) * 0.5)
    return lhs - rhs


def mollweide_ratio_1(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    """Mollweide identity 1 difference: (a - b)/c - sin((alpha - beta)/2)/cos(gamma/2)."""
    lhs = (a - b) / c
    rhs = sin((alpha - beta) * 0.5) / cos(gamma * 0.5)
    return lhs - rhs


def mollweide_ratio_2(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> float:
    """Mollweide identity 2 difference: (a + b)/c - cos((alpha - beta)/2)/sin(gamma/2)."""
    lhs = (a + b) / c
    rhs = cos((alpha - beta) * 0.5) / sin(gamma * 0.5)
    return lhs - rhs


def solve_triangle_sss(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Solve triangle with side lengths a, b, c returning angles (A, B, C) in radians."""
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("Triangle inequality violated")
    ang_a = law_of_cosines_angle(a, b, c)
    ang_b = law_of_cosines_angle(b, a, c)
    ang_c = PI - (ang_a + ang_b)
    return ang_a, ang_b, ang_c


def solve_triangle_sas(side_b: float, angle_a: float, side_c: float) -> Tuple[float, float, float]:
    """Solve triangle with SAS returning side_a, angle_b, angle_c."""
    side_a = law_of_cosines_side(side_b, side_c, angle_a)
    angle_b = law_of_cosines_angle(side_b, side_a, side_c)
    angle_c = PI - (angle_a + angle_b)
    return side_a, angle_b, angle_c


def solve_triangle_asa(angle_a: float, side_c: float, angle_b: float) -> Tuple[float, float, float]:
    """Solve triangle with ASA returning side_a, side_b, angle_c."""
    angle_c = PI - (angle_a + angle_b)
    if angle_c <= 0:
        raise ValueError("Sum of angles exceeds 180 degrees")
    side_a = law_of_sines_side(side_c, angle_c, angle_a)
    side_b = law_of_sines_side(side_c, angle_c, angle_b)
    return side_a, side_b, angle_c


def solve_triangle_aas(angle_a: float, angle_b: float, side_a: float) -> Tuple[float, float, float]:
    """Solve triangle with AAS returning side_b, side_c, angle_c."""
    angle_c = PI - (angle_a + angle_b)
    if angle_c <= 0:
        raise ValueError("Sum of angles exceeds 180 degrees")
    side_b = law_of_sines_side(side_a, angle_a, angle_b)
    side_c = law_of_sines_side(side_a, angle_a, angle_c)
    return side_b, side_c, angle_c


def triangle_area_sas(side_a: float, side_b: float, angle_c: float) -> float:
    """Compute triangle area using SAS formula: 0.5 * a * b * sin(C)."""
    return 0.5 * side_a * side_b * sin(angle_c)


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """Compute triangle area using Heron's formula sqrt(s*(s-a)*(s-b)*(s-c))."""
    s = (a + b + c) * 0.5
    val = s * (s - a) * (s - b) * (s - c)
    if val < 0:
        raise ValueError("Invalid triangle side lengths")
    return math.sqrt(val)


def triangle_circumradius(a: float, b: float, c: float) -> float:
    """Compute triangle circumradius R = (a*b*c) / (4*Area)."""
    area = triangle_area_heron(a, b, c)
    if area == 0:
        raise ValueError("Degenerate triangle has zero area")
    return (a * b * c) / (4.0 * area)


def triangle_inradius(a: float, b: float, c: float) -> float:
    """Compute triangle inradius r = Area / semiperimeter."""
    s = (a + b + c) * 0.5
    if s == 0:
        raise ValueError("Zero perimeter")
    return triangle_area_heron(a, b, c) / s


# ----------------------------------------------------
# 14. Series and Approximations
# ----------------------------------------------------

def sin_taylor(x: float, terms: int = 7) -> float:
    """Taylor series approximation for sin(x) with specified term count."""
    y = normalize_angle_pi(x)
    res = 0.0
    for n in range(terms):
        coeff = ((-1) ** n) / math.factorial(2 * n + 1)
        res += coeff * (y ** (2 * n + 1))
    return res


def cos_taylor(x: float, terms: int = 7) -> float:
    """Taylor series approximation for cos(x) with specified term count."""
    y = normalize_angle_pi(x)
    res = 0.0
    for n in range(terms):
        coeff = ((-1) ** n) / math.factorial(2 * n)
        res += coeff * (y ** (2 * n))
    return res


def tan_series(x: float) -> float:
    """Padé / series approximation for tan(x) near 0."""
    y = normalize_angle_pi(x)
    # tan(x) ~ x + x^3/3 + 2*x^5/15 + 17*x^7/315
    y2 = y * y
    return y * (1.0 + y2 / 3.0 + (2.0 * y2 * y2) / 15.0 + (17.0 * (y2 ** 3)) / 315.0)


def atan_taylor(x: float, terms: int = 15) -> float:
    """Taylor series for atan(x) for |x| <= 1."""
    if abs(x) > 1.0:
        raise ValueError("atan_taylor requires |x| <= 1")
    res = 0.0
    for n in range(terms):
        res += (((-1) ** n) * (x ** (2 * n + 1))) / (2 * n + 1)
    return res


def sinh_taylor(x: float, terms: int = 8) -> float:
    """Taylor series approximation for sinh(x)."""
    res = 0.0
    for n in range(terms):
        res += (x ** (2 * n + 1)) / math.factorial(2 * n + 1)
    return res


def cosh_taylor(x: float, terms: int = 8) -> float:
    """Taylor series approximation for cosh(x)."""
    res = 0.0
    for n in range(terms):
        res += (x ** (2 * n)) / math.factorial(2 * n)
    return res


def bhaskara_sin_approx(deg: float) -> float:
    """Bhaskara I's 7th-century sine approximation formula for angles in [0, 180] deg: 16*x*(180-x)/(40500 - 4*x*(180-x))."""
    x = normalize_angle_360(deg)
    sign = 1.0
    if x > 180.0:
        x -= 180.0
        sign = -1.0
    num = 4.0 * x * (180.0 - x)
    den = 40500.0 - num
    return sign * (4.0 * num / den)


def bhaskara_cos_approx(deg: float) -> float:
    """Bhaskara I's cosine approximation: sin(90 - deg)."""
    return bhaskara_sin_approx(90.0 - deg)


# ----------------------------------------------------
# 15. Phase and Waveform Utilities
# ----------------------------------------------------

def cartesian_to_polar_phase(x: float, y: float) -> float:
    """Return phase angle theta = atan2(y, x) in radians [-pi, pi]."""
    return atan2(y, x)


def amplitude_from_components(in_phase: float, quadrature: float) -> float:
    """Return magnitude/amplitude sqrt(I^2 + Q^2)."""
    return math.sqrt(in_phase * in_phase + quadrature * quadrature)


def phase_difference(phase1: float, phase2: float) -> float:
    """Return wrapped shortest angular difference in [-pi, pi]."""
    return normalize_angle_pi(phase1 - phase2)


def wrap_phase(phase: float) -> float:
    """Wrap phase angle to [-pi, pi)."""
    return normalize_angle_pi(phase)


def unwrap_phase(phases: List[float]) -> List[float]:
    """Unwrap an array of phase angles by eliminating 2*pi discontinuities."""
    if not phases:
        return []
    unwrapped = [phases[0]]
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i - 1]
        diff_wrapped = normalize_angle_pi(diff)
        unwrapped.append(unwrapped[-1] + diff_wrapped)
    return unwrapped


def crest_factor(peak: float, rms: float) -> float:
    """Compute waveform crest factor: peak / rms."""
    if rms == 0:
        raise ZeroDivisionError("RMS cannot be zero")
    return abs(peak) / rms


def form_factor(rms: float, mean_abs: float) -> float:
    """Compute waveform form factor: rms / rectified mean."""
    if mean_abs == 0:
        raise ZeroDivisionError("Rectified mean cannot be zero")
    return rms / mean_abs


# ----------------------------------------------------
# 16. Gudermannian and Lambertian
# ----------------------------------------------------

def gudermannian(x: float) -> float:
    """Compute the Gudermannian function gd(x) = 2*atan(tanh(x/2)) = atan(sinh(x))."""
    return 2.0 * atan(tanh(x * 0.5))


def inverse_gudermannian(phi: float) -> float:
    """Compute inverse Gudermannian gd^(-1)(phi) = ln(tan(pi/4 + phi/2))."""
    if abs(phi) >= HALF_PI:
        raise ValueError("inverse_gudermannian requires |phi| < pi/2")
    return math.log(tan(PI * 0.25 + phi * 0.5))


# ----------------------------------------------------
# 17. Spherical Trigonometry
# ----------------------------------------------------

def spherical_law_of_cosines(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
    """Compute central angular distance in radians using spherical law of cosines."""
    delta_lon = lon2_rad - lon1_rad
    cos_d = sin(lat1_rad) * sin(lat2_rad) + cos(lat1_rad) * cos(lat2_rad) * cos(delta_lon)
    return acos(max(-1.0, min(1.0, cos_d)))


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371000.0) -> float:
    """Compute great-circle distance (meters) between two (lat, lon) coordinates in radians."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = haversine(dlat) + cos(lat1) * cos(lat2) * haversine(dlon)
    c = 2.0 * asin(math.sqrt(max(0.0, min(1.0, a))))
    return radius * c


def great_circle_distance(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float, radius: float = 6371.0) -> float:
    """Compute great-circle distance between two (lat, lon) coordinates in degrees."""
    phi1 = deg_to_rad(lat1_deg)
    phi2 = deg_to_rad(lat2_deg)
    lambda1 = deg_to_rad(lon1_deg)
    lambda2 = deg_to_rad(lon2_deg)
    return haversine_distance(phi1, lambda1, phi2, lambda2, radius)


def initial_bearing(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
    """Compute forward initial bearing in radians [0, 2*pi) from point 1 to point 2."""
    dlon = lon2_rad - lon1_rad
    y = sin(dlon) * cos(lat2_rad)
    x = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    return normalize_angle_2pi(atan2(y, x))


def final_bearing(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
    """Compute final bearing in radians arriving at point 2."""
    b = initial_bearing(lat2_rad, lon2_rad, lat1_rad, lon1_rad)
    return normalize_angle_2pi(b + PI)


def midpoint_spherical(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> Tuple[float, float]:
    """Compute spherical midpoint (lat, lon) in radians between two spherical points."""
    dlon = lon2_rad - lon1_rad
    bx = cos(lat2_rad) * cos(dlon)
    by = cos(lat2_rad) * sin(dlon)
    lat3 = atan2(sin(lat1_rad) + sin(lat2_rad), math.sqrt((cos(lat1_rad) + bx) ** 2 + by ** 2))
    lon3 = lon1_rad + atan2(by, cos(lat1_rad) + bx)
    return lat3, normalize_angle_pi(lon3)


def cross_track_distance(lat_p: float, lon_p: float, lat_start: float, lon_start: float,
                         lat_end: float, lon_end: float, radius: float = 6371000.0) -> float:
    """Compute cross-track distance in meters from point P to great circle path start->end."""
    d13 = haversine_distance(lat_start, lon_start, lat_p, lon_p, radius=1.0)
    theta13 = initial_bearing(lat_start, lon_start, lat_p, lon_p)
    theta12 = initial_bearing(lat_start, lon_start, lat_end, lon_end)
    return asin(sin(d13) * sin(theta13 - theta12)) * radius


def along_track_distance(lat_p: float, lon_p: float, lat_start: float, lon_start: float,
                         lat_end: float, lon_end: float, radius: float = 6371000.0) -> float:
    """Compute along-track distance in meters from path start along path to point closest to P."""
    d13 = haversine_distance(lat_start, lon_start, lat_p, lon_p, radius=1.0)
    xt = cross_track_distance(lat_p, lon_p, lat_start, lon_start, lat_end, lon_end, radius=1.0)
    return acos(cos(d13) / cos(xt)) * radius
