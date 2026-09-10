"""Matrix operations module for claudemath.

Pure-Python implementation of matrix algebra, special matrices, decompositions,
matrix functions (exponential, powers), norms, condition numbers, and sparse representations.
"""

from typing import List, Tuple, Optional, Dict, Any
import math


# ----------------------------------------------------
# 1. Fundamental Matrix Arithmetic and Products
# ----------------------------------------------------

def matrix_zeros(rows: int, cols: int) -> List[List[float]]:
    """Create rows x cols matrix initialized with zeros."""
    return [[0.0] * cols for _ in range(rows)]


def matrix_ones(rows: int, cols: int) -> List[List[float]]:
    """Create rows x cols matrix initialized with ones."""
    return [[1.0] * cols for _ in range(rows)]


def matrix_identity(n: int) -> List[List[float]]:
    """Create n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_diagonal(diag_elements: List[float]) -> List[List[float]]:
    """Create diagonal matrix from list of elements."""
    n = len(diag_elements)
    return [[diag_elements[i] if i == j else 0.0 for j in range(n)] for i in range(n)]


def matrix_shape(A: List[List[float]]) -> Tuple[int, int]:
    """Return dimensions (rows, cols) of matrix."""
    if not A:
        return 0, 0
    return len(A), len(A[0])


def matrix_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix addition A + B."""
    r, c = matrix_shape(A)
    return [[A[i][j] + B[i][j] for j in range(c)] for i in range(r)]


def matrix_sub(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix subtraction A - B."""
    r, c = matrix_shape(A)
    return [[A[i][j] - B[i][j] for j in range(c)] for i in range(r)]


def matrix_scalar_mul(A: List[List[float]], s: float) -> List[List[float]]:
    """Scalar multiplication s * A."""
    return [[s * val for val in row] for row in A]


def matrix_transpose(A: List[List[float]]) -> List[List[float]]:
    """Transpose matrix A^T."""
    r, c = matrix_shape(A)
    return [[A[i][j] for i in range(r)] for j in range(c)]


def matrix_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Standard matrix multiplication A * B."""
    rA, cA = matrix_shape(A)
    rB, cB = matrix_shape(B)
    if cA != rB:
        raise ValueError(f"Matrix dimension mismatch: ({rA}x{cA}) * ({rB}x{cB})")
    res = matrix_zeros(rA, cB)
    for i in range(rA):
        for k in range(cA):
            aik = A[i][k]
            if aik != 0.0:
                for j in range(cB):
                    res[i][j] += aik * B[k][j]
    return res


def matrix_vector_mul(A: List[List[float]], x: List[float]) -> List[float]:
    """Matrix-vector product A * x."""
    r, c = matrix_shape(A)
    if c != len(x):
        raise ValueError("Matrix columns must match vector length")
    return [sum(A[i][j] * x[j] for j in range(c)) for i in range(r)]


def vector_outer_product(u: List[float], v: List[float]) -> List[List[float]]:
    """Outer product u (x) v = u * v^T."""
    return [[ui * vj for vj in v] for ui in u]


def hadamard_product(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Element-wise (Schur / Hadamard) product A .* B."""
    r, c = matrix_shape(A)
    return [[A[i][j] * B[i][j] for j in range(c)] for i in range(r)]


def kronecker_product(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Kronecker tensor product A (x) B."""
    rA, cA = matrix_shape(A)
    rB, cB = matrix_shape(B)
    res = matrix_zeros(rA * rB, cA * cB)
    for i in range(rA):
        for j in range(cA):
            for k in range(rB):
                for l in range(cB):
                    res[i * rB + k][j * cB + l] = A[i][j] * B[k][l]
    return res


def frobenius_inner_product(A: List[List[float]], B: List[List[float]]) -> float:
    """Frobenius inner product <A, B> = Tr(A^T B) = sum(A_ij * B_ij)."""
    r, c = matrix_shape(A)
    return sum(A[i][j] * B[i][j] for i in range(r) for j in range(c))


def matrix_trace(A: List[List[float]]) -> float:
    """Trace of square matrix Tr(A) = sum(A_ii)."""
    n, m = matrix_shape(A)
    if n != m:
        raise ValueError("Trace is only defined for square matrices")
    return sum(A[i][i] for i in range(n))


def matrix_power(A: List[List[float]], p: int) -> List[List[float]]:
    """Compute integer matrix power A^p using binary exponentiation."""
    n, m = matrix_shape(A)
    if n != m:
        raise ValueError("Matrix power requires square matrix")
    if p < 0:
        return matrix_power(matrix_inverse(A), -p)
    res = matrix_identity(n)
    base = [row[:] for row in A]
    while p > 0:
        if p % 2 == 1:
            res = matrix_mul(res, base)
        base = matrix_mul(base, base)
        p //= 2
    return res


# ----------------------------------------------------
# 2. Special Matrices Construction
# ----------------------------------------------------

def toeplitz_matrix(col: List[float], row: List[float]) -> List[List[float]]:
    """Construct Toeplitz matrix where entry (i, j) depends on i - j."""
    m = len(col)
    n = len(row)
    T = matrix_zeros(m, n)
    for i in range(m):
        for j in range(n):
            if i >= j:
                T[i][j] = col[i - j]
            else:
                T[i][j] = row[j - i]
    return T


def hankel_matrix(col: List[float], row: List[float]) -> List[List[float]]:
    """Construct Hankel matrix (constant along anti-diagonals)."""
    m = len(col)
    n = len(row)
    H = matrix_zeros(m, n)
    vals = col + row[1:]
    for i in range(m):
        for j in range(n):
            H[i][j] = vals[i + j]
    return H


def circulant_matrix(c: List[float]) -> List[List[float]]:
    """Construct circulant matrix generated by vector c."""
    n = len(c)
    return [[c[(j - i) % n] for j in range(n)] for i in range(n)]


def vandermonde_matrix(x: List[float], n: Optional[int] = None) -> List[List[float]]:
    """Construct Vandermonde matrix V_ij = x_i^j."""
    m = len(x)
    cols = n if n is not None else m
    return [[(x[i] ** j) for j in range(cols)] for i in range(m)]


def hilbert_matrix(n: int) -> List[List[float]]:
    """Construct n x n Hilbert matrix H_ij = 1 / (i + j + 1)."""
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def companion_matrix(poly_monic: List[float]) -> List[List[float]]:
    """Construct companion matrix for monic polynomial x^n + c_{n-1}*x^{n-1} + ... + c_0."""
    # poly_monic = [c_0, c_1, ..., c_{n-1}]
    n = len(poly_monic)
    C = matrix_zeros(n, n)
    for i in range(1, n):
        C[i][i - 1] = 1.0
    for i in range(n):
        C[i][n - 1] = -poly_monic[i]
    return C


def tridiagonal_matrix(diag_sub: List[float], diag_main: List[float], diag_super: List[float]) -> List[List[float]]:
    """Construct tridiagonal matrix."""
    n = len(diag_main)
    T = matrix_zeros(n, n)
    for i in range(n):
        T[i][i] = diag_main[i]
        if i > 0:
            T[i][i - 1] = diag_sub[i - 1]
        if i < n - 1:
            T[i][i + 1] = diag_super[i]
    return T


def block_matrix_2x2(A11: List[List[float]], A12: List[List[float]],
                     A21: List[List[float]], A22: List[List[float]]) -> List[List[float]]:
    """Assemble 2x2 block matrix from four submatrices."""
    r1, _ = matrix_shape(A11)
    r2, _ = matrix_shape(A21)
    res = []
    for i in range(r1):
        res.append(A11[i] + A12[i])
    for i in range(r2):
        res.append(A21[i] + A22[i])
    return res


# ----------------------------------------------------
# 3. Matrix Norms and Properties
# ----------------------------------------------------

def matrix_norm_1(A: List[List[float]]) -> float:
    """Matrix 1-norm (maximum absolute column sum)."""
    r, c = matrix_shape(A)
    return max(sum(abs(A[i][j]) for i in range(r)) for j in range(c))


def matrix_norm_inf(A: List[List[float]]) -> float:
    """Matrix infinity norm (maximum absolute row sum)."""
    return max(sum(abs(val) for val in row) for row in A)


def matrix_norm_frobenius(A: List[List[float]]) -> float:
    """Frobenius norm ||A||_F = sqrt(sum |A_ij|^2)."""
    return math.sqrt(sum(val * val for row in A for val in row))


def matrix_norm_max(A: List[List[float]]) -> float:
    """Max-norm max |A_ij|."""
    return max(abs(val) for row in A for val in row)


def is_matrix_symmetric(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if square matrix is symmetric A == A^T."""
    n, m = matrix_shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j] - A[j][i]) > tol:
                return False
    return True


def is_matrix_skew_symmetric(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if square matrix is skew-symmetric A == -A^T."""
    n, m = matrix_shape(A)
    if n != m:
        return False
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] + A[j][i]) > tol:
                return False
    return True


def is_matrix_orthogonal(A: List[List[float]], tol: float = 1e-8) -> bool:
    """Check if square matrix is orthogonal A^T * A == I."""
    n, m = matrix_shape(A)
    if n != m:
        return False
    AtA = matrix_mul(matrix_transpose(A), A)
    I = matrix_identity(n)
    for i in range(n):
        for j in range(n):
            if abs(AtA[i][j] - I[i][j]) > tol:
                return False
    return True


def is_matrix_diagonally_dominant(A: List[List[float]]) -> bool:
    """Check if square matrix is strictly diagonally dominant."""
    n, m = matrix_shape(A)
    if n != m:
        return False
    for i in range(n):
        diag = abs(A[i][i])
        off_diag = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag <= off_diag:
            return False
    return True


# ----------------------------------------------------
# 4. Solvers, Inverse, and Condition Number
# ----------------------------------------------------

def forward_substitution(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve lower-triangular system L * y = b."""
    n = len(b)
    y = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        if L[i][i] == 0:
            raise ZeroDivisionError("Singular lower-triangular matrix")
        y[i] = (b[i] - s) / L[i][i]
    return y


def backward_substitution(U: List[List[float]], y: List[float]) -> List[float]:
    """Solve upper-triangular system U * x = y."""
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        if U[i][i] == 0:
            raise ZeroDivisionError("Singular upper-triangular matrix")
        x[i] = (y[i] - s) / U[i][i]
    return x


def lu_decomposition_pivot(A: List[List[float]]) -> Tuple[List[List[float]], List[List[float]], List[int]]:
    """LU decomposition with partial pivoting: P * A = L * U."""
    n, _ = matrix_shape(A)
    U = [row[:] for row in A]
    L = matrix_identity(n)
    pivots = list(range(n))

    for k in range(n - 1):
        # Find pivot
        max_idx = max(range(k, n), key=lambda i: abs(U[i][k]))
        if abs(U[max_idx][k]) < 1e-14:
            continue
        # Swap rows
        U[k], U[max_idx] = U[max_idx], U[k]
        pivots[k], pivots[max_idx] = pivots[max_idx], pivots[k]
        for j in range(k):
            L[k][j], L[max_idx][j] = L[max_idx][j], L[k][j]
        # Elimination
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]
    return L, U, pivots


def matrix_inverse(A: List[List[float]]) -> List[List[float]]:
    """Compute matrix inverse A^{-1} via Gauss-Jordan elimination with partial pivoting."""
    n, m = matrix_shape(A)
    if n != m:
        raise ValueError("Matrix must be square to invert")
    # Augmented matrix [A | I]
    aug = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        # Pivot
        max_r = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[max_r][i]) < 1e-14:
            raise ValueError("Matrix is singular (non-invertible)")
        aug[i], aug[max_r] = aug[max_r], aug[i]
        pivot = aug[i][i]
        for c in range(2 * n):
            aug[i][c] /= pivot
        for r in range(n):
            if r != i:
                factor = aug[r][i]
                for c in range(2 * n):
                    aug[r][c] -= factor * aug[i][c]
    return [row[n:] for row in aug]


def matrix_condition_number_1(A: List[List[float]]) -> float:
    """Compute 1-norm condition number cond_1(A) = ||A||_1 * ||A^{-1}||_1."""
    norm_A = matrix_norm_1(A)
    inv_A = matrix_inverse(A)
    norm_inv = matrix_norm_1(inv_A)
    return norm_A * norm_inv


def matrix_condition_number_inf(A: List[List[float]]) -> float:
    """Compute infinity-norm condition number cond_inf(A) = ||A||_inf * ||A^{-1}||_inf."""
    norm_A = matrix_norm_inf(A)
    inv_A = matrix_inverse(A)
    norm_inv = matrix_norm_inf(inv_A)
    return norm_A * norm_inv


def moore_penrose_pseudoinverse(A: List[List[float]]) -> List[List[float]]:
    """Compute Moore-Penrose pseudo-inverse A^+ using normal equations."""
    r, c = matrix_shape(A)
    At = matrix_transpose(A)
    if r >= c:
        # Full column rank: A^+ = (A^T A)^{-1} A^T
        AtA = matrix_mul(At, A)
        return matrix_mul(matrix_inverse(AtA), At)
    else:
        # Full row rank: A^+ = A^T (A A^T)^{-1}
        AAt = matrix_mul(A, At)
        return matrix_mul(At, matrix_inverse(AAt))


# ----------------------------------------------------
# 5. Matrix Exponential and Functions
# ----------------------------------------------------

def matrix_exponential(A: List[List[float]], terms: int = 25) -> List[List[float]]:
    """Compute matrix exponential exp(A) via Taylor series with scaling and squaring."""
    n, _ = matrix_shape(A)
    norm = matrix_norm_inf(A)
    # Scaling factor 2^s
    s = max(0, int(math.ceil(math.log2(norm)))) if norm > 1.0 else 0
    scale = 2.0 ** s
    scaled_A = matrix_scalar_mul(A, 1.0 / scale)

    # Taylor series for exp(scaled_A)
    res = matrix_identity(n)
    curr = matrix_identity(n)
    for k in range(1, terms + 1):
        curr = matrix_scalar_mul(matrix_mul(curr, scaled_A), 1.0 / k)
        res = matrix_add(res, curr)

    # Squaring step s times
    for _ in range(s):
        res = matrix_mul(res, res)
    return res


def matrix_logarithm_approx(A: List[List[float]], terms: int = 30) -> List[List[float]]:
    """Approximate matrix logarithm log(A) for matrices close to identity using Mercator series."""
    n, _ = matrix_shape(A)
    I = matrix_identity(n)
    X = matrix_sub(A, I)
    res = matrix_zeros(n, n)
    power = list(X)
    for k in range(1, terms + 1):
        term = matrix_scalar_mul(power, ((-1.0) ** (k - 1)) / k)
        res = matrix_add(res, term)
        power = matrix_mul(power, X)
    return res


def matrix_cosine(A: List[List[float]], terms: int = 15) -> List[List[float]]:
    """Compute matrix cosine cos(A) = sum_{k=0}^inf (-1)^k A^{2k} / (2k)!."""
    n, _ = matrix_shape(A)
    res = matrix_identity(n)
    power = matrix_identity(n)
    A2 = matrix_mul(A, A)
    for k in range(1, terms + 1):
        power = matrix_mul(power, A2)
        fact = math.factorial(2 * k)
        sign = (-1.0) ** k
        res = matrix_add(res, matrix_scalar_mul(power, sign / fact))
    return res


def matrix_sine(A: List[List[float]], terms: int = 15) -> List[List[float]]:
    """Compute matrix sine sin(A) = sum_{k=0}^inf (-1)^k A^{2k+1} / (2k+1)!."""
    n, _ = matrix_shape(A)
    res = [row[:] for row in A]
    power = [row[:] for row in A]
    A2 = matrix_mul(A, A)
    for k in range(1, terms + 1):
        power = matrix_mul(power, A2)
        fact = math.factorial(2 * k + 1)
        sign = (-1.0) ** k
        res = matrix_add(res, matrix_scalar_mul(power, sign / fact))
    return res


# ----------------------------------------------------
# 6. Sparse Matrix Representations (COO / Dictionary)
# ----------------------------------------------------

def sparse_from_dense(A: List[List[float]], tol: float = 1e-12) -> Dict[Tuple[int, int], float]:
    """Convert dense matrix to dictionary-of-keys (DOK) sparse representation."""
    r, c = matrix_shape(A)
    dok = {}
    for i in range(r):
        for j in range(c):
            if abs(A[i][j]) > tol:
                dok[(i, j)] = A[i][j]
    return dok


def sparse_to_dense(dok: Dict[Tuple[int, int], float], rows: int, cols: int) -> List[List[float]]:
    """Convert sparse dictionary matrix back to dense rows x cols format."""
    dense = matrix_zeros(rows, cols)
    for (i, j), val in dok.items():
        if 0 <= i < rows and 0 <= j < cols:
            dense[i][j] = val
    return dense


def sparse_vector_mul(dok: Dict[Tuple[int, int], float], x: List[float], rows: int) -> List[float]:
    """Fast sparse matrix-vector product dok * x."""
    res = [0.0] * rows
    for (i, j), val in dok.items():
        if j < len(x):
            res[i] += val * x[j]
    return res


def sparse_add(dokA: Dict[Tuple[int, int], float], dokB: Dict[Tuple[int, int], float]) -> Dict[Tuple[int, int], float]:
    """Add two sparse matrices in DOK format."""
    res = dict(dokA)
    for k, v in dokB.items():
        res[k] = res.get(k, 0.0) + v
        if abs(res[k]) < 1e-14:
            del res[k]
    return res
def matrix_row_echelon_form(A: List[List[float]], tol: float = 1e-12) -> List[List[float]]:
    """Compute Gaussian row echelon form (REF) of matrix A."""
    r, c = matrix_shape(A)
    M = [row[:] for row in A]
    lead = 0
    for row in range(r):
        if lead >= c:
            break
        i = row
        while abs(M[i][lead]) < tol:
            i += 1
            if i == r:
                i = row
                lead += 1
                if lead == c:
                    return M
        M[i], M[row] = M[row], M[i]
        pivot = M[row][lead]
        for col in range(c):
            M[row][col] /= pivot
        for i in range(row + 1, r):
            factor = M[i][lead]
            for col in range(c):
                M[i][col] -= factor * M[row][col]
        lead += 1
    return M


def matrix_reduced_row_echelon_form(A: List[List[float]], tol: float = 1e-12) -> List[List[float]]:
    """Compute Gauss-Jordan reduced row echelon form (RREF) of matrix A."""
    r, c = matrix_shape(A)
    M = [row[:] for row in A]
    lead = 0
    for row in range(r):
        if lead >= c:
            break
        i = row
        while abs(M[i][lead]) < tol:
            i += 1
            if i == r:
                i = row
                lead += 1
                if lead == c:
                    return M
        M[i], M[row] = M[row], M[i]
        pivot = M[row][lead]
        for col in range(c):
            M[row][col] /= pivot
        for i in range(r):
            if i != row:
                factor = M[i][lead]
                for col in range(c):
                    M[i][col] -= factor * M[row][col]
        lead += 1
    return M


def givens_rotation_matrix(n: int, i: int, j: int, theta: float) -> List[List[float]]:
    """Construct n x n Givens rotation matrix in (i, j) plane."""
    G = matrix_identity(n)
    c = math.cos(theta)
    s = math.sin(theta)
    G[i][i] = c
    G[j][j] = c
    G[i][j] = -s
    G[j][i] = s
    return G


def householder_reflection_matrix(v: List[float]) -> List[List[float]]:
    """Construct Householder reflection matrix H = I - 2 * (v v^T) / (v^T v)."""
    n = len(v)
    v_norm_sq = sum(x * x for x in v)
    if v_norm_sq == 0:
        return matrix_identity(n)
    H = matrix_identity(n)
    factor = 2.0 / v_norm_sq
    for i in range(n):
        for j in range(n):
            H[i][j] -= factor * v[i] * v[j]
    return H


def schur_complement(A11: List[List[float]], A12: List[List[float]],
                     A21: List[List[float]], A22: List[List[float]]) -> List[List[float]]:
    """Schur complement S = A22 - A21 * A11^{-1} * A12."""
    inv_A11 = matrix_inverse(A11)
    temp = matrix_mul(A21, inv_A11)
    return matrix_sub(A22, matrix_mul(temp, A12))


def woodbury_matrix_identity(A_inv: List[List[float]],
                             U: List[List[float]],
                             C_inv: List[List[float]],
                             V: List[List[float]]) -> List[List[float]]:
    """Woodbury matrix formula (A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}."""
    VAinv = matrix_mul(V, A_inv)
    VAinvU = matrix_mul(VAinv, U)
    middle = matrix_inverse(matrix_add(C_inv, VAinvU))
    AinvU = matrix_mul(A_inv, U)
    term = matrix_mul(matrix_mul(AinvU, middle), VAinv)
    return matrix_sub(A_inv, term)


def matrix_sign_function(A: List[List[float]], max_iter: int = 50, tol: float = 1e-10) -> List[List[float]]:
    """Compute matrix sign function sign(A) via Roberts iteration X_{k+1} = 0.5 * (X_k + X_k^{-1})."""
    X = [row[:] for row in A]
    for _ in range(max_iter):
        X_inv = matrix_inverse(X)
        X_next = matrix_scalar_mul(matrix_add(X, X_inv), 0.5)
        diff = matrix_norm_frobenius(matrix_sub(X_next, X))
        X = X_next
        if diff < tol:
            break
    return X


def matrix_square_root_denman_beavers(A: List[List[float]], max_iter: int = 50, tol: float = 1e-10) -> List[List[float]]:
    """Compute matrix principal square root sqrt(A) using Denman-Beavers iteration."""
    n, _ = matrix_shape(A)
    Y = [row[:] for row in A]
    Z = matrix_identity(n)
    for _ in range(max_iter):
        Y_inv = matrix_inverse(Y)
        Z_inv = matrix_inverse(Z)
        Y_next = matrix_scalar_mul(matrix_add(Y, Z_inv), 0.5)
        Z_next = matrix_scalar_mul(matrix_add(Z, Y_inv), 0.5)
        diff = matrix_norm_frobenius(matrix_sub(Y_next, Y))
        Y, Z = Y_next, Z_next
        if diff < tol:
            break
    return Y


def sparse_density(dok: Dict[Tuple[int, int], float], rows: int, cols: int) -> float:
    """Sparsity density: ratio of non-zero entries to total matrix elements."""
    if rows <= 0 or cols <= 0:
        return 0.0
    return len(dok) / (rows * cols)


def matrix_rank_row_echelon(A: List[List[float]], tol: float = 1e-10) -> int:
    """Compute matrix rank from RREF."""
    rref = matrix_reduced_row_echelon_form(A, tol)
    rank = 0
    for row in rref:
        if any(abs(val) > tol for val in row):
            rank += 1
    return rank
def matrix_frobenius_distance(A: List[List[float]], B: List[List[float]]) -> float:
    """Frobenius distance ||A - B||_F between two matrices."""
    return matrix_norm_frobenius(matrix_sub(A, B))


def matrix_is_idempotent(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if matrix is idempotent: A^2 == A."""
    A2 = matrix_mul(A, A)
    return matrix_frobenius_distance(A2, A) < tol


def matrix_is_nilpotent(A: List[List[float]], max_k: Optional[int] = None, tol: float = 1e-9) -> bool:
    """Check if square matrix is nilpotent: A^k == 0 for some k <= n."""
    n, _ = matrix_shape(A)
    limit = max_k if max_k is not None else n
    curr = [row[:] for row in A]
    for _ in range(1, limit + 1):
        if matrix_norm_frobenius(curr) < tol:
            return True
        curr = matrix_mul(curr, A)
    return False


def matrix_is_involutory(A: List[List[float]], tol: float = 1e-9) -> bool:
    """Check if matrix is an involution: A^2 == I."""
    n, _ = matrix_shape(A)
    A2 = matrix_mul(A, A)
    I = matrix_identity(n)
    return matrix_frobenius_distance(A2, I) < tol


def matrix_kronecker_sum(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Kronecker sum A (+) B = A (x) I_m + I_n (x) B."""
    n, _ = matrix_shape(A)
    m, _ = matrix_shape(B)
    term1 = kronecker_product(A, matrix_identity(m))
    term2 = kronecker_product(matrix_identity(n), B)
    return matrix_add(term1, term2)
