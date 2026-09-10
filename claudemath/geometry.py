"""Geometry algorithms for claudemath.

Pure-Python implementations of 2D and 3D Euclidean analytic geometry, polygons,
circles, conic sections, polyhedra, convex hull, trigonometry in geometry, and geometric transformations.
"""

from typing import List, Tuple, Optional
import math


# ----------------------------------------------------
# 1. 2D Points, Lines, and Distance Metrics
# ----------------------------------------------------

def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def manhattan_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Manhattan (L1) distance between two 2D points."""
    return abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])


def chebyshev_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Chebyshev (L-infinity) distance between two 2D points."""
    return max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1]))


def minkowski_distance_2d(p1: Tuple[float, float], p2: Tuple[float, float], p: float) -> float:
    """Minkowski (L_p) distance between two 2D points."""
    if p <= 0:
        raise ValueError("p must be positive")
    return (abs(p2[0] - p1[0]) ** p + abs(p2[1] - p1[1]) ** p) ** (1.0 / p)


def midpoint_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Midpoint between two 2D points."""
    return (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5


def point_on_segment_ratio(p1: Tuple[float, float], p2: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Point dividing segment p1-p2 with ratio t (p1 + t*(p2 - p1))."""
    return p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])


def slope_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Slope of line passing through p1 and p2."""
    dx = p2[0] - p1[0]
    if abs(dx) < 1e-15:
        raise ValueError("Line is vertical; slope is undefined")
    return (p2[1] - p1[1]) / dx


def are_collinear_2d(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], tol: float = 1e-9) -> bool:
    """Check if three 2D points are collinear using cross-product determinant."""
    area2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return abs(area2) < tol


def line_standard_form(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float, float]:
    """Return coefficients (A, B, C) of standard line equation A*x + B*y + C = 0."""
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    norm = math.hypot(A, B)
    if norm == 0:
        raise ValueError("Points must be distinct to define a line")
    return A / norm, B / norm, C / norm


def distance_point_to_line_2d(p: Tuple[float, float], line_p1: Tuple[float, float], line_p2: Tuple[float, float]) -> float:
    """Perpendicular distance from point p to infinite line line_p1 - line_p2."""
    A, B, C = line_standard_form(line_p1, line_p2)
    return abs(A * p[0] + B * p[1] + C)


def distance_point_to_segment_2d(p: Tuple[float, float], s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
    """Shortest distance from point p to finite line segment s1-s2."""
    dx = s2[0] - s1[0]
    dy = s2[1] - s1[1]
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return distance_2d(p, s1)
    t = ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy) / l2
    t = max(0.0, min(1.0, t))
    proj = (s1[0] + t * dx, s1[1] + t * dy)
    return distance_2d(p, proj)


def line_intersection_2d(p1: Tuple[float, float], p2: Tuple[float, float],
                         p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Intersection of line (p1, p2) with line (p3, p4), or None if parallel."""
    denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if abs(denom) < 1e-12:
        return None
    det1 = p1[0] * p2[1] - p1[1] * p2[0]
    det2 = p3[0] * p4[1] - p3[1] * p4[0]
    x = (det1 * (p3[0] - p4[0]) - (p1[0] - p2[0]) * det2) / denom
    y = (det1 * (p3[1] - p4[1]) - (p1[1] - p2[1]) * det2) / denom
    return x, y


def reflect_point_across_line_2d(p: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Reflect point p across line (p1, p2)."""
    A, B, C = line_standard_form(p1, p2)
    d = A * p[0] + B * p[1] + C
    return p[0] - 2.0 * A * d, p[1] - 2.0 * B * d


def is_point_on_segment_2d(p: Tuple[float, float], s1: Tuple[float, float], s2: Tuple[float, float], tol: float = 1e-7) -> bool:
    """Check if point p lies on segment s1-s2."""
    return distance_point_to_segment_2d(p, s1, s2) < tol


# ----------------------------------------------------
# 2. Triangles in 2D
# ----------------------------------------------------

def triangle_perimeter(a: float, b: float, c: float) -> float:
    """Triangle perimeter."""
    return a + b + c


def triangle_semiperimeter(a: float, b: float, c: float) -> float:
    """Triangle semiperimeter s = (a + b + c) / 2."""
    return (a + b + c) * 0.5


def triangle_area_heron(a: float, b: float, c: float) -> float:
    """Compute triangle area using Heron's formula sqrt(s*(s-a)*(s-b)*(s-c))."""
    if a + b <= c or a + c <= b or b + c <= a:
        raise ValueError("Side lengths violate triangle inequality")
    s = triangle_semiperimeter(a, b, c)
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def triangle_area_vertices(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """Area of triangle from 3 coordinates using shoelace formula."""
    return 0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))


def triangle_centroid(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
    """Centroid of triangle (average of vertices)."""
    return (p1[0] + p2[0] + p3[0]) / 3.0, (p1[1] + p2[1] + p3[1]) / 3.0


def triangle_inradius(a: float, b: float, c: float) -> float:
    """Inradius r = Area / s."""
    area = triangle_area_heron(a, b, c)
    s = triangle_semiperimeter(a, b, c)
    return area / s


def triangle_circumradius(a: float, b: float, c: float) -> float:
    """Circumradius R = (a * b * c) / (4 * Area)."""
    area = triangle_area_heron(a, b, c)
    return (a * b * c) / (4.0 * area)


def triangle_incenter(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
    """Coordinates of the triangle incenter."""
    a = distance_2d(p2, p3)
    b = distance_2d(p1, p3)
    c = distance_2d(p1, p2)
    p = a + b + c
    if p == 0:
        raise ValueError("Degenerate triangle")
    ix = (a * p1[0] + b * p2[0] + c * p3[0]) / p
    iy = (a * p1[1] + b * p2[1] + c * p3[1]) / p
    return ix, iy


def triangle_circumcenter(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
    """Coordinates of triangle circumcenter (intersection of perpendicular bisectors)."""
    d = 2.0 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(d) < 1e-12:
        raise ValueError("Collinear points do not form a triangle")
    p1_sq = p1[0] ** 2 + p1[1] ** 2
    p2_sq = p2[0] ** 2 + p2[1] ** 2
    p3_sq = p3[0] ** 2 + p3[1] ** 2
    ux = (p1_sq * (p2[1] - p3[1]) + p2_sq * (p3[1] - p1[1]) + p3_sq * (p1[1] - p2[1])) / d
    uy = (p1_sq * (p3[0] - p2[0]) + p2_sq * (p1[0] - p3[0]) + p3_sq * (p2[0] - p1[0])) / d
    return ux, uy


def triangle_orthocenter(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[float, float]:
    """Coordinates of triangle orthocenter using Euler line relation: H = 3*G - 2*O."""
    gx, gy = triangle_centroid(p1, p2, p3)
    ox, oy = triangle_circumcenter(p1, p2, p3)
    return 3.0 * gx - 2.0 * ox, 3.0 * gy - 2.0 * oy


def triangle_angles_from_sides(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """Compute triangle interior angles (A, B, C) in radians via Law of Cosines."""
    cos_A = (b * b + c * c - a * a) / (2.0 * b * c)
    cos_B = (a * a + c * c - b * b) / (2.0 * a * c)
    cos_C = (a * a + b * b - c * c) / (2.0 * a * b)
    A = math.acos(max(-1.0, min(1.0, cos_A)))
    B = math.acos(max(-1.0, min(1.0, cos_B)))
    C = math.acos(max(-1.0, min(1.0, cos_C)))
    return A, B, C


def is_right_triangle(a: float, b: float, c: float, tol: float = 1e-7) -> bool:
    """Check if sides satisfy Pythagorean theorem."""
    sides = sorted([a, b, c])
    return abs(sides[0] ** 2 + sides[1] ** 2 - sides[2] ** 2) < tol


def is_equilateral_triangle(a: float, b: float, c: float, tol: float = 1e-7) -> bool:
    """Check if all sides are equal."""
    return abs(a - b) < tol and abs(b - c) < tol


def is_isosceles_triangle(a: float, b: float, c: float, tol: float = 1e-7) -> bool:
    """Check if at least two sides are equal."""
    return abs(a - b) < tol or abs(b - c) < tol or abs(a - c) < tol


# ----------------------------------------------------
# 3. Polygons and Convex Hulls
# ----------------------------------------------------

def polygon_perimeter(vertices: List[Tuple[float, float]]) -> float:
    """Perimeter of closed polygon."""
    n = len(vertices)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    return sum(distance_2d(vertices[i], vertices[(i + 1) % n]) for i in range(n))


def polygon_area_shoelace(vertices: List[Tuple[float, float]]) -> float:
    """Area of polygon using Shoelace formula (surveyor's formula)."""
    n = len(vertices)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    total = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        total += (x1 * y2 - x2 * y1)
    return 0.5 * abs(total)


def polygon_centroid_shoelace(vertices: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Centroid (center of mass) of non-self-intersecting closed polygon."""
    n = len(vertices)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    area_factor = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area_factor += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = 0.5 * area_factor
    if abs(area) < 1e-14:
        raise ValueError("Degenerate polygon (zero area)")
    return cx / (6.0 * area), cy / (6.0 * area)


def is_convex_polygon(vertices: List[Tuple[float, float]]) -> bool:
    """Check if 2D polygon is convex."""
    n = len(vertices)
    if n < 3:
        return False
    sign = None
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        p3 = vertices[(i + 2) % n]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if abs(cross) > 1e-10:
            cur_sign = cross > 0
            if sign is None:
                sign = cur_sign
            elif cur_sign != sign:
                return False
    return True


def point_in_polygon_ray_casting(point: Tuple[float, float], vertices: List[Tuple[float, float]]) -> bool:
    """Check if point is inside polygon using Jordan curve ray casting algorithm."""
    x, y = point
    n = len(vertices)
    inside = False
    p1x, p1y = vertices[0]
    for i in range(n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def convex_hull_graham_scan(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Compute 2D convex hull using Graham scan algorithm."""
    if len(points) <= 2:
        return list(points)
    # Pivot is lowest y (and leftmost x)
    pivot = min(points, key=lambda p: (p[1], p[0]))
    def polar_angle(p):
        return math.atan2(p[1] - pivot[1], p[0] - pivot[0])
    def dist_sq(p):
        return (p[0] - pivot[0]) ** 2 + (p[1] - pivot[1]) ** 2
    sorted_pts = sorted([p for p in points if p != pivot], key=lambda p: (polar_angle(p), dist_sq(p)))
    hull = [pivot]
    for p in sorted_pts:
        while len(hull) > 1:
            p1 = hull[-2]
            p2 = hull[-1]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def regular_polygon_area(n: int, side: float) -> float:
    """Area of regular n-gon: (n * s^2) / (4 * tan(pi / n))."""
    if n < 3:
        raise ValueError("n must be at least 3")
    return (n * (side ** 2)) / (4.0 * math.tan(math.pi / n))


def regular_polygon_perimeter(n: int, side: float) -> float:
    """Perimeter of regular n-gon: n * side."""
    return n * side


def regular_polygon_apothem(n: int, side: float) -> float:
    """Apothem of regular n-gon: side / (2 * tan(pi / n))."""
    return side / (2.0 * math.tan(math.pi / n))


def regular_polygon_interior_angle(n: int) -> float:
    """Interior angle in radians: (n - 2) * pi / n."""
    return (n - 2) * math.pi / n


def regular_polygon_exterior_angle(n: int) -> float:
    """Exterior angle in radians: 2 * pi / n."""
    return 2.0 * math.pi / n


# ----------------------------------------------------
# 4. Circles and Arcs
# ----------------------------------------------------

def circle_area(radius: float) -> float:
    """Area of circle pi * r^2."""
    return math.pi * (radius ** 2)


def circle_circumference(radius: float) -> float:
    """Circumference of circle 2 * pi * r."""
    return 2.0 * math.pi * radius


def circle_arc_length(radius: float, theta_radians: float) -> float:
    """Arc length s = r * theta."""
    return radius * theta_radians


def circle_sector_area(radius: float, theta_radians: float) -> float:
    """Area of circular sector: 0.5 * r^2 * theta."""
    return 0.5 * (radius ** 2) * theta_radians


def circle_segment_area(radius: float, theta_radians: float) -> float:
    """Area of circular segment: 0.5 * r^2 * (theta - sin(theta))."""
    return 0.5 * (radius ** 2) * (theta_radians - math.sin(theta_radians))


def circle_chord_length(radius: float, theta_radians: float) -> float:
    """Chord length: 2 * r * sin(theta / 2)."""
    return 2.0 * radius * math.sin(theta_radians * 0.5)


def circle_sagitta(radius: float, theta_radians: float) -> float:
    """Sagitta (height of circular arc): r * (1 - cos(theta / 2))."""
    return radius * (1.0 - math.cos(theta_radians * 0.5))


def circle_from_three_points(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    """Find center (cx, cy) and radius r of unique circle passing through 3 non-collinear points."""
    center = triangle_circumcenter(p1, p2, p3)
    radius = distance_2d(center, p1)
    return center, radius


def power_of_point(point: Tuple[float, float], circle_center: Tuple[float, float], radius: float) -> float:
    """Power of point with respect to circle: d^2 - r^2."""
    d = distance_2d(point, circle_center)
    return (d ** 2) - (radius ** 2)


# ----------------------------------------------------
# 5. 3D Points, Lines, and Planes
# ----------------------------------------------------

def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Euclidean distance in 3D space."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)


def midpoint_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Midpoint of two 3D points."""
    return (p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5, (p1[2] + p2[2]) * 0.5


def plane_from_three_points(p1: Tuple[float, float, float],
                            p2: Tuple[float, float, float],
                            p3: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """Return normalized (A, B, C, D) for plane A*x + B*y + C*z + D = 0."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
    v2 = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])
    # Normal via cross product
    nx = v1[1] * v2[2] - v1[2] * v2[1]
    ny = v1[2] * v2[0] - v1[0] * v2[2]
    nz = v1[0] * v2[1] - v1[1] * v2[0]
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm == 0:
        raise ValueError("Collinear points cannot define a unique plane")
    A, B, C = nx / norm, ny / norm, nz / norm
    D = -(A * p1[0] + B * p1[1] + C * p1[2])
    return A, B, C, D


def distance_point_to_plane_3d(point: Tuple[float, float, float], plane: Tuple[float, float, float, float]) -> float:
    """Perpendicular distance from point to plane A*x + B*y + C*z + D = 0."""
    A, B, C, D = plane
    return abs(A * point[0] + B * point[1] + C * point[2] + D) / math.sqrt(A * A + B * B + C * C)


def sphere_volume(radius: float) -> float:
    """Volume of sphere: (4/3) * pi * r^3."""
    return (4.0 / 3.0) * math.pi * (radius ** 3)


def sphere_surface_area(radius: float) -> float:
    """Surface area of sphere: 4 * pi * r^2."""
    return 4.0 * math.pi * (radius ** 2)


def sphere_cap_volume(radius: float, height: float) -> float:
    """Volume of spherical cap of height h: (1/3) * pi * h^2 * (3r - h)."""
    return (1.0 / 3.0) * math.pi * (height ** 2) * (3.0 * radius - height)


def sphere_cap_surface_area(radius: float, height: float) -> float:
    """Curved surface area of spherical cap: 2 * pi * r * h."""
    return 2.0 * math.pi * radius * height


def ellipsoid_volume(a: float, b: float, c: float) -> float:
    """Volume of triaxial ellipsoid: (4/3) * pi * a * b * c."""
    return (4.0 / 3.0) * math.pi * a * b * c


def cylinder_volume(radius: float, height: float) -> float:
    """Volume of right circular cylinder: pi * r^2 * h."""
    return math.pi * (radius ** 2) * height


def cylinder_surface_area(radius: float, height: float) -> float:
    """Total surface area of closed right circular cylinder: 2*pi*r*h + 2*pi*r^2."""
    return 2.0 * math.pi * radius * height + 2.0 * math.pi * (radius ** 2)


def cylinder_lateral_area(radius: float, height: float) -> float:
    """Lateral surface area of cylinder: 2 * pi * r * h."""
    return 2.0 * math.pi * radius * height


def cone_slant_height(radius: float, height: float) -> float:
    """Slant height of right circular cone: sqrt(r^2 + h^2)."""
    return math.hypot(radius, height)


def cone_volume(radius: float, height: float) -> float:
    """Volume of right circular cone: (1/3) * pi * r^2 * h."""
    return (1.0 / 3.0) * math.pi * (radius ** 2) * height


def cone_surface_area(radius: float, height: float) -> float:
    """Total surface area of closed cone: pi * r * (r + slant_height)."""
    return math.pi * radius * (radius + cone_slant_height(radius, height))


def cone_frustum_volume(r1: float, r2: float, height: float) -> float:
    """Volume of conical frustum: (1/3) * pi * h * (r1^2 + r1*r2 + r2^2)."""
    return (1.0 / 3.0) * math.pi * height * (r1 * r1 + r1 * r2 + r2 * r2)


def torus_volume(major_radius: float, minor_radius: float) -> float:
    """Volume of torus: 2 * pi^2 * R * r^2."""
    return 2.0 * (math.pi ** 2) * major_radius * (minor_radius ** 2)


def torus_surface_area(major_radius: float, minor_radius: float) -> float:
    """Surface area of torus: 4 * pi^2 * R * r."""
    return 4.0 * (math.pi ** 2) * major_radius * minor_radius


def rectangular_prism_volume(l: float, w: float, h: float) -> float:
    """Volume of rectangular box l * w * h."""
    return l * w * h


def rectangular_prism_surface_area(l: float, w: float, h: float) -> float:
    """Total surface area of rectangular prism: 2*(lw + lh + wh)."""
    return 2.0 * (l * w + l * h + w * h)


def rectangular_prism_diagonal(l: float, w: float, h: float) -> float:
    """Space diagonal of rectangular box: sqrt(l^2 + w^2 + h^2)."""
    return math.sqrt(l * l + w * w + h * h)


# ----------------------------------------------------
# 6. Platonic Solids and Polyhedra
# ----------------------------------------------------

def tetrahedron_volume(edge: float) -> float:
    """Volume of regular tetrahedron: edge^3 / (6 * sqrt(2))."""
    return (edge ** 3) / (6.0 * math.sqrt(2.0))


def tetrahedron_surface_area(edge: float) -> float:
    """Surface area of regular tetrahedron: sqrt(3) * edge^2."""
    return math.sqrt(3.0) * (edge ** 2)


def octahedron_volume(edge: float) -> float:
    """Volume of regular octahedron: (sqrt(2) / 3) * edge^3."""
    return (math.sqrt(2.0) / 3.0) * (edge ** 3)


def octahedron_surface_area(edge: float) -> float:
    """Surface area of regular octahedron: 2 * sqrt(3) * edge^2."""
    return 2.0 * math.sqrt(3.0) * (edge ** 2)


def dodecahedron_volume(edge: float) -> float:
    """Volume of regular dodecahedron: (15 + 7*sqrt(5)) / 4 * edge^3."""
    return ((15.0 + 7.0 * math.sqrt(5.0)) / 4.0) * (edge ** 3)


def dodecahedron_surface_area(edge: float) -> float:
    """Surface area of regular dodecahedron: 3 * sqrt(25 + 10*sqrt(5)) * edge^2."""
    return 3.0 * math.sqrt(25.0 + 10.0 * math.sqrt(5.0)) * (edge ** 2)


def icosahedron_volume(edge: float) -> float:
    """Volume of regular icosahedron: (5 / 12) * (3 + sqrt(5)) * edge^3."""
    return (5.0 / 12.0) * (3.0 + math.sqrt(5.0)) * (edge ** 3)


def icosahedron_surface_area(edge: float) -> float:
    """Surface area of regular icosahedron: 5 * sqrt(3) * edge^2."""
    return 5.0 * math.sqrt(3.0) * (edge ** 2)


def euler_characteristic_polyhedron(v: int, e: int, f: int) -> int:
    """Compute Euler characteristic chi = V - E + F (equals 2 for convex polyhedron)."""
    return v - e + f
def cube_volume(edge: float) -> float:
    """Volume of a cube: edge^3."""
    return edge ** 3


def cube_surface_area(edge: float) -> float:
    """Surface area of a cube: 6 * edge^2."""
    return 6.0 * (edge ** 2)


def cube_space_diagonal(edge: float) -> float:
    """Space diagonal of cube: sqrt(3) * edge."""
    return math.sqrt(3.0) * edge


def ellipse_area(semi_major: float, semi_minor: float) -> float:
    """Area of ellipse: pi * a * b."""
    return math.pi * semi_major * semi_minor


def ellipse_circumference_ramanujan(semi_major: float, semi_minor: float) -> float:
    """Perimeter of ellipse using Ramanujan second approximation."""
    a, b = semi_major, semi_minor
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(4.0 - 3.0 * h)))


def ellipse_eccentricity(semi_major: float, semi_minor: float) -> float:
    """Eccentricity e = sqrt(1 - (b/a)^2)."""
    if semi_major < semi_minor:
        semi_major, semi_minor = semi_minor, semi_major
    return math.sqrt(1.0 - (semi_minor / semi_major) ** 2)


def hyperbola_eccentricity(semi_transverse: float, semi_conjugate: float) -> float:
    """Eccentricity e = sqrt(1 + (b/a)^2)."""
    return math.sqrt(1.0 + (semi_conjugate / semi_transverse) ** 2)


def parabola_focal_length(a: float) -> float:
    """Focal length f = 1 / (4*a) for parabola y = a*x^2."""
    if a == 0:
        raise ValueError("a cannot be zero")
    return 1.0 / (4.0 * abs(a))


def angle_between_planes_3d(plane1: Tuple[float, float, float, float],
                           plane2: Tuple[float, float, float, float]) -> float:
    """Dihedral angle in radians between two planes."""
    n1 = (plane1[0], plane1[1], plane1[2])
    n2 = (plane2[0], plane2[1], plane2[2])
    dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
    norm1 = math.sqrt(n1[0] ** 2 + n1[1] ** 2 + n1[2] ** 2)
    norm2 = math.sqrt(n2[0] ** 2 + n2[1] ** 2 + n2[2] ** 2)
    cos_theta = abs(dot) / (norm1 * norm2)
    return math.acos(max(-1.0, min(1.0, cos_theta)))
