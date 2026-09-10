"""Linear algebra algorithms for claudemath.

Pure-Python implementation of matrix-vector systems, decompositions,
eigenvalue algorithms, vector spaces, projections, and geometric transformations.
"""

from typing import List, Tuple, Optional
import math


def vector_dot(u: List[float], v: List[float]) -> float:
    """Compute dot product of two vectors."""
    if len(u) != len(v):
        raise ValueError("Vectors must have equal length")
    return sum(x * y for x, y in zip(u, v))


def vector_norm(v: List[float]) -> float:
    """Compute Euclidean (L2) norm of vector."""
    return math.sqrt(sum(x * x for x in v))


def vector_p_norm(v: List[float], p: float) -> float:
    """Compute L_p norm of vector."""
    if p <= 0:
        raise ValueError("p must be positive")
    return sum(abs(x) ** p for x in v) ** (1.0 / p)


def vector_manhattan_norm(v: List[float]) -> float:
    """Compute Manhattan (L1) norm of vector."""
    return sum(abs(x) for x in v)


def vector_chebyshev_norm(v: List[float]) -> float:
    """Compute Chebyshev (L-infinity) norm of vector."""
    return max(abs(x) for x in v) if v else 0.0


def vector_normalize(v: List[float]) -> List[float]:
    """Return unit vector in direction of v."""
    n = vector_norm(v)
    if n == 0:
        raise ValueError("Cannot normalize zero vector")
    return [x / n for x in v]


def vector_add(u: List[float], v: List[float]) -> List[float]:
    """Vector addition u + v."""
    if len(u) != len(v):
        raise ValueError("Dimension mismatch")
    return [x + y for x, y in zip(u, v)]


def vector_sub(u: List[float], v: List[float]) -> List[float]:
    """Vector subtraction u - v."""
    if len(u) != len(v):
        raise ValueError("Dimension mismatch")
    return [x - y for x, y in zip(u, v)]


def vector_scale(v: List[float], s: float) -> List[float]:
    """Scalar multiplication s * v."""
    return [s * x for x in v]


def vector_cross_3d(u: List[float], v: List[float]) -> List[float]:
    """Compute 3D cross product u x v."""
    if len(u) != 3 or len(v) != 3:
        raise ValueError("Cross product requires 3D vectors")
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0]
    ]


def vector_outer_product(u: List[float], v: List[float]) -> List[List[float]]:
    """Compute outer product u v^T producing m x n matrix."""
    return [[x * y for y in v] for x in u]


def vector_projection(u: List[float], v: List[float]) -> List[float]:
    """Compute vector projection of u onto v: proj_v(u) = ((u.v) / (v.v)) * v."""
    denom = vector_dot(v, v)
    if denom == 0:
        raise ValueError("Cannot project onto zero vector")
    scalar = vector_dot(u, v) / denom
    return vector_scale(v, scalar)


def vector_rejection(u: List[float], v: List[float]) -> List[float]:
    """Compute vector rejection of u from v: u - proj_v(u)."""
    return vector_sub(u, vector_projection(u, v))


def angle_between_vectors(u: List[float], v: List[float]) -> float:
    """Compute angle in radians between two vectors."""
    nu = vector_norm(u)
    nv = vector_norm(v)
    if nu == 0 or nv == 0:
        raise ValueError("Zero vector has undefined angle")
    cos_theta = vector_dot(u, v) / (nu * nv)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def is_orthogonal(u: List[float], v: List[float], tol: float = 1e-9) -> bool:
    """Check if vectors u and v are orthogonal."""
    return abs(vector_dot(u, v)) < tol


def is_orthonormal(vectors: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if set of vectors is orthonormal."""
    k = len(vectors)
    for i in range(k):
        if abs(vector_norm(vectors[i]) - 1.0) > tol:
            return False
        for j in range(i + 1, k):
            if abs(vector_dot(vectors[i], vectors[j])) > tol:
                return False
    return True


def basis_orthogonalization(vectors: List[List[float]]) -> List[List[float]]:
    """Gram-Schmidt orthogonalization producing orthogonal basis."""
    basis: List[List[float]] = []
    for v in vectors:
        w = list(v)
        for u in basis:
            proj = vector_projection(v, u)
            w = vector_sub(w, proj)
        if vector_norm(w) > 1e-9:
            basis.append(w)
    return basis


def modified_gram_schmidt(vectors: List[List[float]]) -> List[List[float]]:
    """Numerically stable Modified Gram-Schmidt returning orthonormal basis."""
    basis: List[List[float]] = []
    for v in vectors:
        w = list(v)
        for q in basis:
            dot = vector_dot(w, q)
            w = vector_sub(w, vector_scale(q, dot))
        nw = vector_norm(w)
        if nw > 1e-9:
            basis.append(vector_scale(w, 1.0 / nw))
    return basis


def is_linearly_independent(vectors: List[List[float]], tol: float = 1e-8) -> bool:
    """Check if a set of vectors is linearly independent."""
    if not vectors:
        return True
    m = len(vectors)
    ortho = basis_orthogonalization(vectors)
    return len(ortho) == m


def span_contains(vectors: List[List[float]], target: List[float], tol: float = 1e-7) -> bool:
    """Check if target vector lies within the span of vectors."""
    basis = modified_gram_schmidt(vectors)
    residual = list(target)
    for q in basis:
        dot = vector_dot(target, q)
        residual = vector_sub(residual, vector_scale(q, dot))
    return vector_norm(residual) < tol


# ----------------------------------------------------
# Matrix Fundamentals
# ----------------------------------------------------

def identity_matrix(n: int) -> List[List[float]]:
    """Return n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def zero_matrix(rows: int, cols: int) -> List[List[float]]:
    """Return rows x cols zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def diag_matrix(elements: List[float]) -> List[List[float]]:
    """Create diagonal matrix from list of elements."""
    n = len(elements)
    return [[elements[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_trace(A: List[List[float]]) -> float:
    """Compute trace (sum of diagonal entries) of square matrix."""
    return sum(A[i][i] for i in range(min(len(A), len(A[0]))))


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """Compute transpose A^T."""
    m, n = len(A), len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix addition A + B."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matrix_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix subtraction A - B."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def matrix_scale(A: List[List[float]], s: float) -> List[List[float]]:
    """Scalar multiplication s * A."""
    return [[s * x for x in row] for row in A]


def matrix_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix multiplication A * B."""
    m = len(A)
    k_len = len(A[0])
    n = len(B[0])
    if k_len != len(B):
        raise ValueError("Incompatible dimensions for matrix multiplication")
    res = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for k in range(k_len):
            aik = A[i][k]
            for j in range(n):
                res[i][j] += aik * B[k][j]
    return res


def matrix_vector_mul(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix-vector multiplication A * v."""
    return [sum(row[j] * v[j] for j in range(len(v))) for row in A]


def is_symmetric(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if square matrix A is symmetric (A = A^T)."""
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def is_skew_symmetric(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if square matrix A is skew-symmetric (A = -A^T)."""
    n = len(A)
    for i in range(n):
        if abs(A[i][i]) > tol:
            return False
        for j in range(i + 1, n):
            if abs(A[i][j] + A[j][i]) > tol:
                return False
    return True


def is_orthogonal_matrix(A: List[List[float]], tol: float = 1e-7) -> bool:
    """Check if square matrix A is orthogonal (A^T * A = I)."""
    n = len(A)
    At = matrix_transpose(A)
    prod = matrix_mul(At, A)
    for i in range(n):
        for j in range(n):
            target = 1.0 if i == j else 0.0
            if abs(prod[i][j] - target) > tol:
                return False
    return True


# ----------------------------------------------------
# Systems and Decompositions
# ----------------------------------------------------

def solve_triangular_lower(L: List[List[float]], b: List[float]) -> List[float]:
    """Forward substitution solving L x = b for lower triangular L."""
    n = len(L)
    x = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * x[j] for j in range(i))
        if abs(L[i][i]) < 1e-15:
            raise ZeroDivisionError("Singular triangular matrix")
        x[i] = (b[i] - s) / L[i][i]
    return x


def solve_triangular_upper(U: List[List[float]], b: List[float]) -> List[float]:
    """Back substitution solving U x = b for upper triangular U."""
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        if abs(U[i][i]) < 1e-15:
            raise ZeroDivisionError("Singular triangular matrix")
        x[i] = (b[i] - s) / U[i][i]
    return x


def solve_gaussian(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system A x = b using Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [A[i] + [b[i]] for i in range(n)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-12:
            raise ValueError("Matrix is singular or near-singular")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (M[i][n] - s) / M[i][i]
    return x


def solve_gauss_jordan(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve linear system A x = b using full Gauss-Jordan elimination to RREF."""
    n = len(A)
    M = [A[i] + [b[i]] for i in range(n)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular system in Gauss-Jordan")
        for k in range(i, n + 1):
            M[i][k] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for k in range(i, n + 1):
                    M[r][k] -= factor * M[i][k]
    return [M[i][n] for i in range(n)]


def solve_least_squares_normal_eq(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve overdetermined system A x ~ b using normal equations: (A^T A) x = A^T b."""
    At = matrix_transpose(A)
    AtA = matrix_mul(At, A)
    Atb = matrix_vector_mul(At, b)
    return solve_gaussian(AtA, Atb)


def solve_tridiagonal_thomas(diag_low: List[float], diag_main: List[float],
                             diag_up: List[float], d: List[float]) -> List[float]:
    """Thomas algorithm for tridiagonal system."""
    n = len(diag_main)
    c_star = [0.0] * n
    d_star = [0.0] * n
    c_star[0] = diag_up[0] / diag_main[0]
    d_star[0] = d[0] / diag_main[0]
    for i in range(1, n - 1):
        denom = diag_main[i] - diag_low[i - 1] * c_star[i - 1]
        c_star[i] = diag_up[i] / denom
        d_star[i] = (d[i] - diag_low[i - 1] * d_star[i - 1]) / denom
    denom = diag_main[n - 1] - diag_low[n - 2] * c_star[n - 2]
    d_star[n - 1] = (d[n - 1] - diag_low[n - 2] * d_star[n - 2]) / denom
    x = [0.0] * n
    x[n - 1] = d_star[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_star[i] - c_star[i] * x[i + 1]
    return x


def solve_cramer_2x2(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve 2x2 linear system using Cramer's rule."""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-14:
        raise ValueError("System is singular (det = 0)")
    det_x = b[0] * A[1][1] - A[0][1] * b[1]
    det_y = A[0][0] * b[1] - b[0] * A[1][0]
    return [det_x / det, det_y / det]


def solve_cramer_3x3(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve 3x3 linear system using Cramer's rule."""
    def det3(m: List[List[float]]) -> float:
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    det = det3(A)
    if abs(det) < 1e-14:
        raise ValueError("Singular system (det = 0)")
    Ax = [[b[i], A[i][1], A[i][2]] for i in range(3)]
    Ay = [[A[i][0], b[i], A[i][2]] for i in range(3)]
    Az = [[A[i][0], A[i][1], b[i]] for i in range(3)]
    return [det3(Ax) / det, det3(Ay) / det, det3(Az) / det]


def matrix_determinant_gauss(A: List[List[float]]) -> float:
    """Compute determinant of square matrix via Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [list(row) for row in A]
    sign = 1.0
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-14:
            return 0.0
        if i != max_row:
            M[i], M[max_row] = M[max_row], M[i]
            sign = -sign
        pivot = M[i][i]
        for j in range(i + 1, n):
            factor = M[j][i] / pivot
            for k in range(i, n):
                M[j][k] -= factor * M[i][k]
    det = sign
    for i in range(n):
        det *= M[i][i]
    return det


def matrix_inverse_gauss(A: List[List[float]]) -> List[List[float]]:
    """Compute inverse of square matrix A using Gauss-Jordan elimination."""
    n = len(A)
    M = [A[i] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[max_row][i]) < 1e-14:
            raise ValueError("Matrix is singular and cannot be inverted")
        M[i], M[max_row] = M[max_row], M[i]
        pivot = M[i][i]
        for k in range(2 * n):
            M[i][k] /= pivot
        for r in range(n):
            if r != i:
                factor = M[r][i]
                for k in range(2 * n):
                    M[r][k] -= factor * M[i][k]
    return [row[n:] for row in M]


def lu_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute Doolittle LU decomposition A = L * U."""
    n = len(A)
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - s
        for k in range(i + 1, n):
            s = sum(L[k][j] * U[j][i] for j in range(i))
            if abs(U[i][i]) < 1e-15:
                raise ValueError("Zero pivot in LU decomposition")
            L[k][i] = (A[k][i] - s) / U[i][i]
    return L, U


def plu_decomposition(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Compute PLU decomposition P * A = L * U with partial pivoting."""
    n = len(A)
    P = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [list(A[i]) for i in range(n)]
    for i in range(n):
        max_row = max(range(i, n), key=lambda r: abs(U[r][i]))
        if i != max_row:
            U[i], U[max_row] = U[max_row], U[i]
            P[i], P[max_row] = P[max_row], P[i]
            for k in range(i):
                L[i][k], L[max_row][k] = L[max_row][k], L[i][k]
        pivot = U[i][i]
        if abs(pivot) < 1e-14:
            continue
        for j in range(i + 1, n):
            factor = U[j][i] / pivot
            L[j][i] = factor
            for k in range(i, n):
                U[j][k] -= factor * U[i][k]
    return P, L, U


def cholesky_decomposition(A: List[List[float]]) -> List[List[float]]:
    """Compute Cholesky decomposition A = L * L^T for symmetric positive-definite A."""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive-definite")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def qr_decomposition_gram_schmidt(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute QR decomposition A = Q * R using Modified Gram-Schmidt."""
    m = len(A)
    n = len(A[0])
    cols = [[A[i][j] for i in range(m)] for j in range(n)]
    Q_cols: List[List[float]] = []
    R = [[0.0] * n for _ in range(n)]
    for j in range(n):
        v = list(cols[j])
        for i in range(j):
            R[i][j] = vector_dot(Q_cols[i], cols[j])
            v = vector_sub(v, vector_scale(Q_cols[i], R[i][j]))
        norm_v = vector_norm(v)
        R[j][j] = norm_v
        if norm_v > 1e-12:
            Q_cols.append(vector_scale(v, 1.0 / norm_v))
        else:
            Q_cols.append([0.0] * m)
    Q = [[Q_cols[j][i] for j in range(n)] for i in range(m)]
    return Q, R


def householder_matrix(v: List[float]) -> List[List[float]]:
    """Compute Householder reflection matrix H = I - 2 * (v v^T) / (v^T v)."""
    n = len(v)
    norm_sq = sum(x * x for x in v)
    if norm_sq < 1e-15:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    H = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            H[i][j] = delta - 2.0 * v[i] * v[j] / norm_sq
    return H


def givens_rotation_matrix(n: int, i: int, j: int, theta: float) -> List[List[float]]:
    """Compute n x n Givens rotation matrix rotating planes (i, j) by angle theta."""
    G = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
    c = math.cos(theta)
    s = math.sin(theta)
    G[i][i] = c
    G[i][j] = -s
    G[j][i] = s
    G[j][j] = c
    return G


# ----------------------------------------------------
# Subspaces, Rank, and Nullity
# ----------------------------------------------------

def matrix_rank(A: List[List[float]], tol: float = 1e-9) -> int:
    """Compute rank of matrix A via row echelon reduction."""
    M = [list(row) for row in A]
    m, n = len(M), len(M[0])
    rank = 0
    row = 0
    for col in range(n):
        max_r = max(range(row, m), key=lambda r: abs(M[r][col]))
        if abs(M[max_r][col]) < tol:
            continue
        M[row], M[max_r] = M[max_r], M[row]
        pivot = M[row][col]
        for r in range(row + 1, m):
            factor = M[r][col] / pivot
            for c in range(col, n):
                M[r][c] -= factor * M[row][c]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def nullity(A: List[List[float]], tol: float = 1e-9) -> int:
    """Compute nullity (dimension of null space) = cols - rank."""
    return len(A[0]) - matrix_rank(A, tol)


def null_space_basis(A: List[List[float]], tol: float = 1e-7) -> List[List[float]]:
    """Find a basis for the null space ker(A) using RREF."""
    m, n = len(A), len(A[0])
    M = [list(row) for row in A]
    lead = 0
    pivot_cols = []
    for r in range(m):
        if lead >= n:
            break
        i = r
        while abs(M[i][lead]) < tol:
            i += 1
            if i == m:
                i = r
                lead += 1
                if lead == n:
                    break
        if lead == n:
            break
        M[i], M[r] = M[r], M[i]
        pivot = M[r][lead]
        for c in range(n):
            M[r][c] /= pivot
        for i_row in range(m):
            if i_row != r:
                factor = M[i_row][lead]
                for c in range(n):
                    M[i_row][c] -= factor * M[r][c]
        pivot_cols.append(lead)
        lead += 1
    free_cols = [c for c in range(n) if c not in pivot_cols]
    basis = []
    for free in free_cols:
        vec = [0.0] * n
        vec[free] = 1.0
        for r, piv in enumerate(pivot_cols):
            vec[piv] = -M[r][free]
        basis.append(vec)
    return basis


def column_space_basis(A: List[List[float]], tol: float = 1e-8) -> List[List[float]]:
    """Return basis vectors for column space of matrix A."""
    m, n = len(A), len(A[0])
    cols = [[A[i][j] for i in range(m)] for j in range(n)]
    return basis_orthogonalization(cols)


def row_space_basis(A: List[List[float]], tol: float = 1e-8) -> List[List[float]]:
    """Return basis vectors for row space of matrix A."""
    return basis_orthogonalization(A)


# ----------------------------------------------------
# Eigenvalues and Power Iteration
# ----------------------------------------------------

def power_iteration(A: List[List[float]], max_iter: int = 200, tol: float = 1e-9) -> Tuple[float, List[float]]:
    """Power iteration to find dominant eigenvalue and eigenvector."""
    n = len(A)
    x = [1.0] * n
    lam = 0.0
    for _ in range(max_iter):
        Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        norm = vector_norm(Ax)
        if norm < 1e-15:
            return 0.0, x
        x_new = [val / norm for val in Ax]
        lam = sum(x_new[i] * sum(A[i][j] * x_new[j] for j in range(n)) for i in range(n))
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_new, x)))
        x = x_new
        if diff < tol:
            break
    return lam, x


def inverse_power_iteration(A: List[List[float]], sigma: float = 0.0, max_iter: int = 100, tol: float = 1e-9) -> Tuple[float, List[float]]:
    """Inverse power iteration with spectral shift sigma to find closest eigenvalue."""
    n = len(A)
    A_shifted = [[A[i][j] - (sigma if i == j else 0.0) for j in range(n)] for i in range(n)]
    x = [1.0] * n
    for _ in range(max_iter):
        y = solve_gaussian(A_shifted, x)
        norm = vector_norm(y)
        x_new = [val / norm for val in y]
        diff = math.sqrt(sum((a - b) ** 2 for a, b in zip(x_new, x)))
        x = x_new
        if diff < tol:
            break
    lam = sum(x[i] * sum(A[i][j] * x[j] for j in range(n)) for i in range(n))
    return lam, x


def rayleigh_quotient(A: List[List[float]], x: List[float]) -> float:
    """Compute Rayleigh quotient R(A, x) = (x^T A x) / (x^T x)."""
    n = len(x)
    Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    return vector_dot(x, Ax) / vector_dot(x, x)


def characteristic_polynomial_2x2(A: List[List[float]]) -> Tuple[float, float, float]:
    """Coefficients (a, b, c) of lambda^2 + b*lambda + c = 0 for 2x2 matrix."""
    trace = A[0][0] + A[1][1]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return 1.0, -trace, det


def eigenvalues_2x2(A: List[List[float]]) -> Tuple[complex, complex]:
    """Compute exact eigenvalues of 2x2 matrix."""
    a, b, c = characteristic_polynomial_2x2(A)
    disc = complex(b * b - 4.0 * a * c, 0.0)
    root_disc = disc ** 0.5
    return (-b + root_disc) / (2.0 * a), (-b - root_disc) / (2.0 * a)


def spectral_radius(A: List[List[float]], max_iter: int = 150) -> float:
    """Compute spectral radius rho(A) = max |lambda|."""
    lam, _ = power_iteration(A, max_iter=max_iter)
    return abs(lam)


def gershgorin_disks(A: List[List[float]]) -> List[Tuple[float, float]]:
    """Compute Gershgorin disks [(center_i, radius_i)] for square matrix A."""
    n = len(A)
    disks = []
    for i in range(n):
        center = A[i][i]
        radius = sum(abs(A[i][j]) for j in range(n) if j != i)
        disks.append((center, radius))
    return disks


# ----------------------------------------------------
# Geometric Transformations and Special Matrices
# ----------------------------------------------------

def rotation_matrix_2d(theta: float) -> List[List[float]]:
    """2D counterclockwise rotation matrix."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def rotation_matrix_3d_x(theta: float) -> List[List[float]]:
    """3D rotation matrix about X axis."""
    c, s = math.cos(theta), math.sin(theta)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def rotation_matrix_3d_y(theta: float) -> List[List[float]]:
    """3D rotation matrix about Y axis."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rotation_matrix_3d_z(theta: float) -> List[List[float]]:
    """3D rotation matrix about Z axis."""
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def rotation_matrix_axis_angle(axis: List[float], theta: float) -> List[List[float]]:
    """Rodrigues formula for 3D rotation by theta around arbitrary unit axis."""
    u = vector_normalize(axis)
    ux, uy, uz = u[0], u[1], u[2]
    c, s = math.cos(theta), math.sin(theta)
    c1 = 1.0 - c
    return [
        [c + ux * ux * c1, ux * uy * c1 - uz * s, ux * uz * c1 + uy * s],
        [uy * ux * c1 + uz * s, c + uy * uy * c1, uy * uz * c1 - ux * s],
        [uz * ux * c1 - uy * s, uz * uy * c1 + ux * s, c + uz * uz * c1]
    ]


def scaling_matrix_2d(sx: float, sy: float) -> List[List[float]]:
    """2D non-uniform scaling matrix."""
    return [[sx, 0.0], [0.0, sy]]


def scaling_matrix_3d(sx: float, sy: float, sz: float) -> List[List[float]]:
    """3D non-uniform scaling matrix."""
    return [[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, sz]]


def shear_matrix_2d(kx: float, ky: float) -> List[List[float]]:
    """2D shear matrix: x' = x + kx*y, y' = y + ky*x."""
    return [[1.0, kx], [ky, 1.0]]


def reflection_matrix_2d(theta: float) -> List[List[float]]:
    """2D reflection across line at angle theta with positive x-axis."""
    c2 = math.cos(2.0 * theta)
    s2 = math.sin(2.0 * theta)
    return [[c2, s2], [s2, -c2]]


def reflection_matrix_hyperplane(normal: List[float]) -> List[List[float]]:
    """Householder reflection across hyperplane perpendicular to normal."""
    return householder_matrix(normal)


def projection_matrix_onto_subspace(basis: List[List[float]]) -> List[List[float]]:
    """Compute projection matrix P onto subspace with orthonormal basis: P = sum(q_k q_k^T)."""
    ortho = modified_gram_schmidt(basis)
    if not ortho:
        return []
    n = len(ortho[0])
    P = [[0.0] * n for _ in range(n)]
    for q in ortho:
        for i in range(n):
            for j in range(n):
                P[i][j] += q[i] * q[j]
    return P


def matrix_frobenius_norm(A: List[List[float]]) -> float:
    """Frobenius norm of matrix A: sqrt(sum_{i,j} |A_{i,j}|^2)."""
    return math.sqrt(sum(x * x for row in A for x in row))


def matrix_1_norm(A: List[List[float]]) -> float:
    """Maximum absolute column sum norm."""
    m, n = len(A), len(A[0])
    return max(sum(abs(A[i][j]) for i in range(m)) for j in range(n))


def matrix_inf_norm(A: List[List[float]]) -> float:
    """Maximum absolute row sum norm."""
    return max(sum(abs(x) for x in row) for row in A)


def matrix_condition_number_est(A: List[List[float]]) -> float:
    """Condition number estimate: ||A||_1 * ||A^{-1}||_1."""
    A_inv = matrix_inverse_gauss(A)
    return matrix_1_norm(A) * matrix_1_norm(A_inv)


def bilinear_form_eval(A: List[List[float]], x: List[float], y: List[float]) -> float:
    """Evaluate bilinear form x^T A y."""
    n = len(x)
    Ay = [sum(A[i][j] * y[j] for j in range(len(y))) for i in range(n)]
    return vector_dot(x, Ay)


def quadratic_form_eval(A: List[List[float]], x: List[float]) -> float:
    """Evaluate quadratic form x^T A x."""
    return bilinear_form_eval(A, x, x)


def is_positive_definite(A: List[List[float]]) -> bool:
    """Check if matrix is positive definite via Cholesky decomposition."""
    try:
        cholesky_decomposition(A)
        return True
    except Exception:
        return False


def commutator(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix commutator [A, B] = A B - B A."""
    n = len(A)
    AB = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    BA = [[sum(B[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[AB[i][j] - BA[i][j] for j in range(n)] for i in range(n)]


def anticommutator(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix anticommutator {A, B} = A B + B A."""
    n = len(A)
    AB = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    BA = [[sum(B[i][k] * A[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return [[AB[i][j] + BA[i][j] for j in range(n)] for i in range(n)]


def companion_matrix(poly_coeffs: List[float]) -> List[List[float]]:
    """Construct Frobenius companion matrix for monic polynomial p(x) = x^n + c_{n-1} x^{n-1} + ... + c_0."""
    # poly_coeffs = [c_0, c_1, ..., c_{n-1}]
    n = len(poly_coeffs)
    C = [[0.0] * n for _ in range(n)]
    for i in range(1, n):
        C[i][i - 1] = 1.0
    for j in range(n):
        C[0][j] = -poly_coeffs[n - 1 - j]
    return C


def vandermonde_matrix(x: List[float], degree: int) -> List[List[float]]:
    """Construct Vandermonde matrix V_{i,j} = x_i^j."""
    return [[(val ** p) for p in range(degree + 1)] for val in x]


def hilbert_matrix(n: int) -> List[List[float]]:
    """Construct n x n Hilbert matrix H_{i,j} = 1 / (i + j + 1)."""
    return [[1.0 / (i + j + 1.0) for j in range(n)] for i in range(n)]
