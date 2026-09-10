"""Coordinate systems module for claudemath.

Pure-Python implementation of transformations between Cartesian, Polar, Cylindrical,
Spherical, Elliptic, Parabolic, Toroidal, and WGS84 Geodetic coordinate systems,
along with metric scale factors, Jacobian determinants, and great-circle geodesy formulas.
"""

from typing import Tuple, List, Optional
import math


# ----------------------------------------------------
# 1. 2D Coordinate Transformations
# ----------------------------------------------------

def cartesian_to_polar_2d(x: float, y: float) -> Tuple[float, float]:
    """Convert Cartesian (x, y) to polar (r, theta) with theta in (-pi, pi]."""
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    return r, theta


def polar_to_cartesian_2d(r: float, theta: float) -> Tuple[float, float]:
    """Convert polar (r, theta) to Cartesian (x, y)."""
    if r < 0:
        raise ValueError("Radius r must be non-negative")
    return r * math.cos(theta), r * math.sin(theta)


def cartesian_to_log_polar_2d(x: float, y: float) -> Tuple[float, float]:
    """Convert Cartesian (x, y) to log-polar (rho = ln(r), theta)."""
    r, theta = cartesian_to_polar_2d(x, y)
    if r == 0:
        raise ValueError("Log-polar undefined at origin")
    return math.log(r), theta


def log_polar_to_cartesian_2d(rho: float, theta: float) -> Tuple[float, float]:
    """Convert log-polar (rho, theta) to Cartesian (x, y)."""
    r = math.exp(rho)
    return polar_to_cartesian_2d(r, theta)


def cartesian_to_bipolar_2d(x: float, y: float, a: float = 1.0) -> Tuple[float, float]:
    """Convert 2D Cartesian to bipolar coordinates (tau, sigma) with focal distance 2a."""
    d1_sq = (x + a) ** 2 + y * y
    d2_sq = (x - a) ** 2 + y * y
    if d2_sq == 0:
        raise ZeroDivisionError("Singular point at focus (a, 0)")
    tau = 0.5 * math.log(d1_sq / d2_sq)
    # sigma = angle between vectors from foci
    sigma = math.atan2(2.0 * a * y, x * x + y * y - a * a)
    return tau, sigma


def bipolar_to_cartesian_2d(tau: float, sigma: float, a: float = 1.0) -> Tuple[float, float]:
    """Convert bipolar coordinates (tau, sigma) to Cartesian (x, y)."""
    denom = math.cosh(tau) - math.cos(sigma)
    if abs(denom) < 1e-14:
        raise ZeroDivisionError("Denominator zero in bipolar mapping")
    x = a * math.sinh(tau) / denom
    y = a * math.sin(sigma) / denom
    return x, y


def cartesian_to_parabolic_2d(x: float, y: float) -> Tuple[float, float]:
    """Convert 2D Cartesian to parabolic coordinates (sigma, tau)."""
    r = math.hypot(x, y)
    sigma = math.sqrt(max(0.0, r + x))
    tau = math.sqrt(max(0.0, r - x)) * (1.0 if y >= 0 else -1.0)
    return sigma, tau


def parabolic_to_cartesian_2d(sigma: float, tau: float) -> Tuple[float, float]:
    """Convert parabolic coordinates (sigma, tau) to Cartesian (x, y)."""
    x = 0.5 * (sigma * sigma - tau * tau)
    y = sigma * tau
    return x, y


def cartesian_to_elliptic_2d(x: float, y: float, a: float = 1.0) -> Tuple[float, float]:
    """Convert 2D Cartesian to elliptic coordinates (mu, nu)."""
    d1 = math.hypot(x + a, y)
    d2 = math.hypot(x - a, y)
    mu = math.acosh(max(1.0, (d1 + d2) / (2.0 * a)))
    cos_nu = max(-1.0, min(1.0, (d1 - d2) / (2.0 * a)))
    nu = math.acos(cos_nu) * (1.0 if y >= 0 else -1.0)
    return mu, nu


def elliptic_to_cartesian_2d(mu: float, nu: float, a: float = 1.0) -> Tuple[float, float]:
    """Convert elliptic coordinates (mu, nu) to Cartesian (x, y)."""
    x = a * math.cosh(mu) * math.cos(nu)
    y = a * math.sinh(mu) * math.sin(nu)
    return x, y


# ----------------------------------------------------
# 2. 3D Coordinate Transformations
# ----------------------------------------------------

def cartesian_to_cylindrical_3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert 3D Cartesian (x, y, z) to cylindrical (rho, phi, z)."""
    rho = math.hypot(x, y)
    phi = math.atan2(y, x)
    return rho, phi, z


def cylindrical_to_cartesian_3d(rho: float, phi: float, z: float) -> Tuple[float, float, float]:
    """Convert cylindrical (rho, phi, z) to Cartesian (x, y, z)."""
    if rho < 0:
        raise ValueError("Radial distance rho must be non-negative")
    x = rho * math.cos(phi)
    y = rho * math.sin(phi)
    return x, y, z


def cartesian_to_spherical_physics_3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert 3D Cartesian to spherical (ISO/physics: r, theta=colatitude [0, pi], phi=azimuth [-pi, pi])."""
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, z / r)))
    phi = math.atan2(y, x)
    return r, theta, phi


def spherical_physics_to_cartesian_3d(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """Convert spherical (r, theta=colatitude, phi=azimuth) to Cartesian (x, y, z)."""
    if r < 0:
        raise ValueError("Radial distance r must be non-negative")
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def cylindrical_to_spherical_3d(rho: float, phi: float, z: float) -> Tuple[float, float, float]:
    """Convert cylindrical (rho, phi, z) to spherical (r, theta, phi)."""
    r = math.hypot(rho, z)
    theta = math.atan2(rho, z)
    return r, theta, phi


def spherical_to_cylindrical_3d(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """Convert spherical (r, theta, phi) to cylindrical (rho, phi, z)."""
    rho = r * math.sin(theta)
    z = r * math.cos(theta)
    return rho, phi, z


def cartesian_to_toroidal_3d(x: float, y: float, z: float, a: float = 1.0) -> Tuple[float, float, float]:
    """Convert 3D Cartesian to toroidal coordinates (tau, sigma, phi)."""
    r_cyl = math.hypot(x, y)
    phi = math.atan2(y, x)
    tau, sigma = cartesian_to_bipolar_2d(r_cyl, z, a)
    return tau, sigma, phi


def toroidal_to_cartesian_3d(tau: float, sigma: float, phi: float, a: float = 1.0) -> Tuple[float, float, float]:
    """Convert toroidal coordinates (tau, sigma, phi) to Cartesian (x, y, z)."""
    denom = math.cosh(tau) - math.cos(sigma)
    r_cyl = a * math.sinh(tau) / denom
    z = a * math.sin(sigma) / denom
    x = r_cyl * math.cos(phi)
    y = r_cyl * math.sin(phi)
    return x, y, z


# ----------------------------------------------------
# 3. Geodetic and Great-Circle Navigation (WGS84)
# ----------------------------------------------------

# WGS84 Ellipsoid constants
WGS84_A = 6378137.0          # Semi-major axis in meters
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis
WGS84_E_SQ = 2.0 * WGS84_F - WGS84_F * WGS84_F # Eccentricity squared


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> Tuple[float, float, float]:
    """Convert WGS84 geodetic coordinates (lat, lon in degrees, altitude in meters) to ECEF (X, Y, Z)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_lat * sin_lat)
    X = (N + alt_m) * cos_lat * math.cos(lon)
    Y = (N + alt_m) * cos_lat * math.sin(lon)
    Z = (N * (1.0 - WGS84_E_SQ) + alt_m) * sin_lat
    return X, Y, Z


def ecef_to_geodetic(X: float, Y: float, Z: float) -> Tuple[float, float, float]:
    """Convert ECEF (X, Y, Z in meters) to WGS84 geodetic (lat_deg, lon_deg, alt_m) via Bowring's method."""
    p = math.hypot(X, Y)
    if p < 1e-6:
        lat = math.pi * 0.5 if Z >= 0 else -math.pi * 0.5
        alt = abs(Z) - WGS84_B
        return math.degrees(lat), 0.0, alt
    lon = math.atan2(Y, X)
    # Bowring's closed formula
    e_prime_sq = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B)
    theta = math.atan2(Z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        Z + e_prime_sq * WGS84_B * (math.sin(theta) ** 3),
        p - WGS84_E_SQ * WGS84_A * (math.cos(theta) ** 3)
    )
    sin_lat = math.sin(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E_SQ * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N
    return math.degrees(lat), math.degrees(lon), alt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371000.0) -> float:
    """Great-circle distance in meters between two lat/lon points using Haversine formula."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi * 0.5) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlam * 0.5) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius * c


def initial_compass_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial forward azimuth / bearing in degrees [0, 360) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def destination_point_great_circle(lat: float, lon: float, distance_m: float,
                                   bearing_deg: float, radius: float = 6371000.0) -> Tuple[float, float]:
    """Calculate destination point (lat, lon in degrees) given distance and initial bearing."""
    delta = distance_m / radius
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lam1 = math.radians(lon)
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lam2 = lam1 + math.atan2(math.sin(theta) * math.sin(delta) * math.cos(phi1),
                             math.cos(delta) - math.sin(phi1) * math.sin(phi2))
    return math.degrees(phi2), (math.degrees(lam2) + 540.0) % 360.0 - 180.0


def great_circle_midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, float]:
    """Calculate intermediate midpoint between two coordinates on sphere."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    bx = math.cos(phi2) * math.cos(dlam)
    by = math.cos(phi2) * math.sin(dlam)
    phi3 = math.atan2(math.sin(phi1) + math.sin(phi2), math.sqrt((math.cos(phi1) + bx) ** 2 + by * by))
    lam3 = math.radians(lon1) + math.atan2(by, math.cos(phi1) + bx)
    return math.degrees(phi3), (math.degrees(lam3) + 540.0) % 360.0 - 180.0


def mercator_projection_forward(lat_deg: float, lon_deg: float, r: float = 6378137.0) -> Tuple[float, float]:
    """Forward standard Mercator projection mapping lat/lon to map (x, y) in meters."""
    if abs(lat_deg) > 85.051129:
        raise ValueError("Latitude beyond standard Mercator limits (-85 to +85 deg)")
    x = r * math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    y = r * math.log(math.tan(math.pi * 0.25 + lat_rad * 0.5))
    return x, y


def mercator_projection_inverse(x: float, y: float, r: float = 6378137.0) -> Tuple[float, float]:
    """Inverse Mercator projection mapping map (x, y) back to lat/lon in degrees."""
    lon_deg = math.degrees(x / r)
    lat_deg = math.degrees(2.0 * math.atan(math.exp(y / r)) - math.pi * 0.5)
    return lat_deg, lon_deg


# ----------------------------------------------------
# 4. Metric Scale Factors and Jacobian Determinants
# ----------------------------------------------------

def scale_factors_polar_2d(r: float) -> Tuple[float, float]:
    """Lame scale factors (h_r, h_theta) for 2D polar coordinates: (1, r)."""
    return 1.0, r


def scale_factors_cylindrical_3d(rho: float) -> Tuple[float, float, float]:
    """Lame scale factors (h_rho, h_phi, h_z) for cylindrical coordinates: (1, rho, 1)."""
    return 1.0, rho, 1.0


def scale_factors_spherical_3d(r: float, theta: float) -> Tuple[float, float, float]:
    """Lame scale factors (h_r, h_theta, h_phi) for spherical coordinates: (1, r, r*sin(theta))."""
    return 1.0, r, r * math.sin(theta)


def jacobian_determinant_polar_2d(r: float) -> float:
    """Jacobian determinant of Cartesian-to-Polar transform: |J| = r."""
    return r


def jacobian_determinant_cylindrical_3d(rho: float) -> float:
    """Jacobian determinant of Cartesian-to-Cylindrical transform: |J| = rho."""
    return rho


def jacobian_determinant_spherical_3d(r: float, theta: float) -> float:
    """Jacobian determinant of Cartesian-to-Spherical transform: |J| = r^2 * sin(theta)."""
    return r * r * math.sin(theta)
def cartesian_to_spherical_math_3d(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert Cartesian to spherical with elevation delta in [-pi/2, pi/2] and azimuth in [-pi, pi]."""
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0, 0.0
    elev = math.asin(max(-1.0, min(1.0, z / r)))
    az = math.atan2(y, x)
    return r, elev, az


def spherical_math_to_cartesian_3d(r: float, elev: float, az: float) -> Tuple[float, float, float]:
    """Convert spherical (r, elevation, azimuth) to Cartesian (x, y, z)."""
    x = r * math.cos(elev) * math.cos(az)
    y = r * math.cos(elev) * math.sin(az)
    z = r * math.sin(elev)
    return x, y, z


def spherical_distance(r: float, theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Great-circle distance on sphere of radius r between (theta1, phi1) and (theta2, phi2)."""
    cos_angle = math.cos(theta1)*math.cos(theta2) + math.sin(theta1)*math.sin(theta2)*math.cos(phi1 - phi2)
    return r * math.acos(max(-1.0, min(1.0, cos_angle)))


def spherical_law_of_cosines_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371000.0) -> float:
    """Spherical Law of Cosines distance between lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    cos_c = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(dlam)
    return radius * math.acos(max(-1.0, min(1.0, cos_c)))


def equirectangular_approximation_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371000.0) -> float:
    """Fast flat-surface approximation distance for nearby geographic coordinates."""
    x = math.radians(lon2 - lon1) * math.cos(0.5 * math.radians(lat1 + lat2))
    y = math.radians(lat2 - lat1)
    return radius * math.hypot(x, y)


def rhumb_line_distance(lat1: float, lon1: float, lat2: float, lon2: float, radius: float = 6371000.0) -> float:
    """Distance along loxodrome (rhumb line / constant bearing)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    if abs(dlam) > math.pi:
        dlam = -(2.0 * math.pi - dlam) if dlam > 0 else (2.0 * math.pi + dlam)
    # Projected latitude difference
    dpsi = math.log(math.tan(math.pi * 0.25 + phi2 * 0.5) / math.tan(math.pi * 0.25 + phi1 * 0.5))
    q = (dphi / dpsi) if abs(dpsi) > 1e-12 else math.cos(phi1)
    return radius * math.sqrt(dphi * dphi + q * q * dlam * dlam)


def rhumb_line_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Constant bearing in degrees along rhumb line from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    if abs(dlam) > math.pi:
        dlam = -(2.0 * math.pi - dlam) if dlam > 0 else (2.0 * math.pi + dlam)
    dpsi = math.log(math.tan(math.pi * 0.25 + phi2 * 0.5) / math.tan(math.pi * 0.25 + phi1 * 0.5))
    bearing = math.degrees(math.atan2(dlam, dpsi))
    return (bearing + 360.0) % 360.0


def cross_track_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                         lat3: float, lon3: float, radius: float = 6371000.0) -> float:
    """Cross-track distance from point 3 to great circle path between points 1 and 2."""
    d13 = haversine_distance(lat1, lon1, lat3, lon3, radius) / radius
    theta13 = math.radians(initial_compass_bearing(lat1, lon1, lat3, lon3))
    theta12 = math.radians(initial_compass_bearing(lat1, lon1, lat2, lon2))
    return radius * math.asin(math.sin(d13) * math.sin(theta13 - theta12))


def prolate_spheroidal_to_cartesian(xi: float, eta: float, phi: float, a: float = 1.0) -> Tuple[float, float, float]:
    """Convert prolate spheroidal (xi, eta, phi) to Cartesian (x, y, z)."""
    x = a * math.sinh(xi) * math.sin(eta) * math.cos(phi)
    y = a * math.sinh(xi) * math.sin(eta) * math.sin(phi)
    z = a * math.cosh(xi) * math.cos(eta)
    return x, y, z


def oblate_spheroidal_to_cartesian(xi: float, eta: float, phi: float, a: float = 1.0) -> Tuple[float, float, float]:
    """Convert oblate spheroidal (xi, eta, phi) to Cartesian (x, y, z)."""
    x = a * math.cosh(xi) * math.sin(eta) * math.cos(phi)
    y = a * math.cosh(xi) * math.sin(eta) * math.sin(phi)
    z = a * math.sinh(xi) * math.cos(eta)
    return x, y, z


def scale_factors_parabolic_2d(sigma: float, tau: float) -> Tuple[float, float]:
    """Lame scale factors (h_sigma, h_tau) for parabolic 2D coordinates: (sqrt(sigma^2 + tau^2), sqrt(sigma^2 + tau^2))."""
    h = math.hypot(sigma, tau)
    return h, h


def scale_factors_elliptic_2d(mu: float, nu: float, a: float = 1.0) -> Tuple[float, float]:
    """Lame scale factors for elliptic 2D coordinates."""
    h = a * math.sqrt(math.sinh(mu) ** 2 + math.sin(nu) ** 2)
    return h, h


def scale_factors_bipolar_2d(tau: float, sigma: float, a: float = 1.0) -> Tuple[float, float]:
    """Lame scale factors for bipolar 2D coordinates."""
    denom = math.cosh(tau) - math.cos(sigma)
    h = a / denom
    return h, h


def gradient_in_polar_coordinates(df_dr: float, df_dtheta: float, r: float) -> Tuple[float, float]:
    """Components of gradient in polar basis (e_r, e_theta): (df/dr, (1/r)*df/dtheta)."""
    if r == 0:
        raise ZeroDivisionError("Gradient singular at origin in polar coordinates")
    return df_dr, df_dtheta / r


def divergence_in_polar_coordinates(fr: float, ftheta: float,
                                    dfr_dr: float, dftheta_dtheta: float, r: float) -> float:
    """Divergence of vector field in polar coordinates: (1/r)*d(r*fr)/dr + (1/r)*dftheta/dtheta."""
    if r == 0:
        raise ZeroDivisionError("Divergence singular at origin in polar coordinates")
    return dfr_dr + (fr / r) + (dftheta_dtheta / r)


def gradient_in_cylindrical_coordinates(df_drho: float, df_dphi: float, df_dz: float, rho: float) -> Tuple[float, float, float]:
    """Components of gradient in cylindrical basis (e_rho, e_phi, e_z)."""
    if rho == 0:
        raise ZeroDivisionError("Singular on z-axis")
    return df_drho, df_dphi / rho, df_dz


def divergence_in_cylindrical_coordinates(frho: float, fphi: float, fz: float,
                                         dfrho_drho: float, dfphi_dphi: float, dfz_dz: float,
                                         rho: float) -> float:
    """Divergence in cylindrical coordinates."""
    if rho == 0:
        raise ZeroDivisionError("Singular on z-axis")
    return dfrho_drho + (frho / rho) + (dfphi_dphi / rho) + dfz_dz


def gradient_in_spherical_coordinates(df_dr: float, df_dtheta: float, df_dphi: float,
                                      r: float, theta: float) -> Tuple[float, float, float]:
    """Components of gradient in spherical basis (e_r, e_theta, e_phi)."""
    if r == 0 or math.sin(theta) == 0:
        raise ZeroDivisionError("Singular point in spherical gradient")
    return df_dr, df_dtheta / r, df_dphi / (r * math.sin(theta))


def gnomonic_projection_forward(lat_deg: float, lon_deg: float,
                                lat0_deg: float, lon0_deg: float, r: float = 6371000.0) -> Tuple[float, float]:
    """Gnomonic map projection from sphere center onto tangent plane (great circles become straight lines)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    phi0 = math.radians(lat0_deg)
    lam0 = math.radians(lon0_deg)
    cos_c = math.sin(phi0)*math.sin(phi) + math.cos(phi0)*math.cos(phi)*math.cos(lam - lam0)
    if cos_c <= 0:
        raise ValueError("Point not in visible hemisphere for Gnomonic projection")
    x = (r * math.cos(phi) * math.sin(lam - lam0)) / cos_c
    y = (r * (math.cos(phi0)*math.sin(phi) - math.sin(phi0)*math.cos(phi)*math.cos(lam - lam0))) / cos_c
    return x, y
def stereographic_projection_forward(lat_deg: float, lon_deg: float,
                                     lat0_deg: float, lon0_deg: float, r: float = 6371000.0) -> Tuple[float, float]:
    """Conformal Stereographic projection forward map."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    phi0 = math.radians(lat0_deg)
    lam0 = math.radians(lon0_deg)
    k = 2.0 * r / (1.0 + math.sin(phi0) * math.sin(phi) + math.cos(phi0) * math.cos(phi) * math.cos(lam - lam0))
    x = k * math.cos(phi) * math.sin(lam - lam0)
    y = k * (math.cos(phi0) * math.sin(phi) - math.sin(phi0) * math.cos(phi) * math.cos(lam - lam0))
    return x, y


def orthographic_projection_forward(lat_deg: float, lon_deg: float,
                                    lat0_deg: float, lon0_deg: float, r: float = 6371000.0) -> Tuple[float, float]:
    """Orthographic perspective projection (as seen from deep space)."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    phi0 = math.radians(lat0_deg)
    lam0 = math.radians(lon0_deg)
    cos_c = math.sin(phi0)*math.sin(phi) + math.cos(phi0)*math.cos(phi)*math.cos(lam - lam0)
    if cos_c < 0:
        raise ValueError("Point on back side of globe")
    x = r * math.cos(phi) * math.sin(lam - lam0)
    y = r * (math.cos(phi0)*math.sin(phi) - math.sin(phi0)*math.cos(phi)*math.cos(lam - lam0))
    return x, y


def lambert_cylindrical_forward(lat_deg: float, lon_deg: float,
                                lon0_deg: float = 0.0, r: float = 6371000.0) -> Tuple[float, float]:
    """Lambert cylindrical equal-area projection."""
    x = r * math.radians(lon_deg - lon0_deg)
    y = r * math.sin(math.radians(lat_deg))
    return x, y


def azimuthal_equidistant_forward(lat_deg: float, lon_deg: float,
                                  lat0_deg: float, lon0_deg: float, r: float = 6371000.0) -> Tuple[float, float]:
    """Azimuthal equidistant projection preserving true distances from projection center."""
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    phi0 = math.radians(lat0_deg)
    lam0 = math.radians(lon0_deg)
    cos_c = math.sin(phi0)*math.sin(phi) + math.cos(phi0)*math.cos(phi)*math.cos(lam - lam0)
    c = math.acos(max(-1.0, min(1.0, cos_c)))
    if c == 0:
        return 0.0, 0.0
    k = c / math.sin(c)
    x = r * k * math.cos(phi) * math.sin(lam - lam0)
    y = r * k * (math.cos(phi0)*math.sin(phi) - math.sin(phi0)*math.cos(phi)*math.cos(lam - lam0))
    return x, y


def vincenty_inverse_distance_approx(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Iterative Vincenty geodesic distance on WGS84 ellipsoid."""
    a = WGS84_A
    b = WGS84_B
    f = WGS84_F
    L = math.radians(lon2 - lon1)
    U1 = math.atan((1.0 - f) * math.tan(math.radians(lat1)))
    U2 = math.atan((1.0 - f) * math.tan(math.radians(lat2)))
    sinU1, cosU1 = math.sin(U1), math.cos(U1)
    sinU2, cosU2 = math.sin(U2), math.cos(U2)
    lam = L
    for _ in range(100):
        sinLam, cosLam = math.sin(lam), math.cos(lam)
        sinSigma = math.hypot(cosU2 * sinLam, cosU1 * sinU2 - sinU1 * cosU2 * cosLam)
        if sinSigma == 0:
            return 0.0
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLam
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = (cosU1 * cosU2 * sinLam) / sinSigma
        cosSqAlpha = 1.0 - sinAlpha * sinAlpha
        cos2SigmaM = cosSigma - 2.0 * sinU1 * sinU2 / cosSqAlpha if cosSqAlpha != 0 else 0.0
        C = f / 16.0 * cosSqAlpha * (4.0 + f * (4.0 - 3.0 * cosSqAlpha))
        lamPrev = lam
        lam = L + (1.0 - C) * f * sinAlpha * (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)))
        if abs(lam - lamPrev) < 1e-12:
            break
    uSq = cosSqAlpha * (a * a - b * b) / (b * b)
    A_val = 1.0 + uSq / 16384.0 * (4096.0 + uSq * (-768.0 + uSq * (320.0 - 175.0 * uSq)))
    B_val = uSq / 1024.0 * (256.0 + uSq * (-128.0 + uSq * (74.0 - 47.0 * uSq)))
    deltaSigma = B_val * sinSigma * (cos2SigmaM + B_val / 4.0 * (cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM) -
                 B_val / 6.0 * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) * (-3.0 + 4.0 * cos2SigmaM * cos2SigmaM)))
    s = b * A_val * (sigma - deltaSigma)
    return s
