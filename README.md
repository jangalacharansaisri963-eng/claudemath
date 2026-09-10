# claudemath

**claudemath** is an advanced, comprehensive, zero-dependency pure-Python mathematics library designed for PyPI. It features **1,490+ mathematical algorithms** across **20 specialized mathematical domains**.

---

## Key Highlights

- **Pure Python Standard Library**: Absolutely zero external dependencies (no NumPy or SciPy required). Runs anywhere Python 3.8+ runs.
- **20 Comprehensive Domains**: 1,490+ verified algorithms, numerical routines, symbolic helpers, and mathematical transforms.
- **Deterministic & High-Precision**: Built with precision, type annotations (`typing`), robust numerical stability, and comprehensive edge-case handling.
- **Educational & Production Ready**: Clear mathematical docstrings citing formulas and theorems.

---

## Domain Overview

| Module | Description | Function Count |
|---|---|---|
| `claudemath.arithmetic` | GCD/LCM, modular arithmetic, factorials, binomials, means, ratios, rounding | 72 |
| `claudemath.algebra` | Linear/quadratic/cubic/quartic solvers, inequalities, systems, sequences | 80 |
| `claudemath.trigonometry` | Standard & hyperbolic functions, inverse hyperbolic, angle conversions, identities | 140 |
| `claudemath.calculus` | Numerical derivatives, single & multivariable integrals, Taylor series, limits | 114 |
| `claudemath.linear_algebra` | Vectors, dot/cross products, norms, angles, projections, Gram-Schmidt | 84 |
| `claudemath.matrix_operations` | Matrix addition, multiplication, determinants, inverses, rank, LU, QR, Cholesky | 63 |
| `claudemath.statistics` | Central tendency, dispersion, skewness, kurtosis, covariance, correlation, regression | 86 |
| `claudemath.probability` | PMF/PDF/CDF for binomial, Poisson, normal, exponential, beta, gamma distributions | 72 |
| `claudemath.combinatorics` | Permutations, combinations, partitions, Bell numbers, Stirling numbers, derangements | 73 |
| `claudemath.geometry` | 2D/3D shapes, Euclidean geometry, polygons, polyhedra, intersection tests | 87 |
| `claudemath.number_theory` | Primality tests (Miller-Rabin), sieve of Eratosthenes, totient, Mobius, Chinese remainder | 102 |
| `claudemath.complex_numbers` | Arithmetic, polar/rectangular conversions, complex roots, exponential/log/trig | 65 |
| `claudemath.polynomial_operations` | Polynomial arithmetic, roots (Durand-Kerner), Horner's method, Chebyshev, Legendre | 62 |
| `claudemath.discrete_mathematics` | Logic, truth tables, boolean algebra, recurrence relations, posets, lattices | 56 |
| `claudemath.numerical_analysis` | Root finding (Newton, Halley, Brent), interpolation (Lagrange, spline), Romberg | 66 |
| `claudemath.vector_operations` | 2D/3D/nD vector operations, coordinate transforms, angle between vectors, distance metrics | 60 |
| `claudemath.coordinate_systems` | Cartesian, Polar, Cylindrical, Spherical, Toroidal, Parabolic conversions | 56 |
| `claudemath.fourier_analysis` | FFT (Cooley-Tukey), IFFT, DFT, DCT-I..IV, DST-I..IV, FWHT, STFT, spectral analysis | 51 |
| `claudemath.differential_equations`| Euler, Heun, RK4, RKF45, Adams-Bashforth/Moulton, Lorenz, Lotka-Volterra, PDEs | 51 |
| `claudemath.graph_theory` | BFS, DFS, Dijkstra, Bellman-Ford, Floyd-Warshall, A*, Kruskal, Prim, PageRank, Tarjan | 56 |

**Total: 1,496 functions across 20 modules.**

---

## Installation

```bash
pip install claudemath
```

Or clone and install in development mode:

```bash
git clone https://github.com/claudemath/claudemath.git
cd claudemath
pip install -e .
```

---

## Quickstart Examples

### 1. Arithmetic & Number Theory
```python
from claudemath import arithmetic, number_theory

# Extended Euclidean Algorithm
gcd, x, y = number_theory.extended_gcd(240, 46)
print(f"gcd(240, 46) = {gcd}, x = {x}, y = {y}")

# Fast Miller-Rabin Primality Test
print("Is 104729 prime?", number_theory.is_prime_miller_rabin(104729))

# Modular Inverse
print("Mod inverse of 3 mod 11:", number_theory.modular_inverse(3, 11))
```

### 2. Calculus & Numerical Analysis
```python
from claudemath import calculus, numerical_analysis
import math

# Adaptive Simpson's rule integration
f = lambda x: math.sin(x) / (x + 1.0)
integral = calculus.integral_simpson(f, 0.0, math.pi, n_intervals=100)
print("Integral:", integral)

# Brent's method root finding
root = numerical_analysis.brent_root(lambda x: x**3 - 2*x - 5, 2.0, 3.0)
print("Root of x^3 - 2x - 5 = 0:", root)
```

### 3. Differential Equations & Chaos
```python
from claudemath import differential_equations

# Solve Lorenz chaotic attractor with classical RK4
ts, states = differential_equations.ode_system_solve_rk4(
    differential_equations.lorenz_system_derivative,
    t0=0.0,
    y0=[1.0, 1.0, 1.0],
    t_end=10.0,
    n_steps=1000
)
print(f"Computed {len(states)} time steps. Final state: {states[-1]}")
```

### 4. Graph Theory
```python
from claudemath import graph_theory

# Shortest paths via Dijkstra
adj = {
    0: [(1, 4.0), (2, 2.0)],
    1: [(2, 1.0), (3, 5.0)],
    2: [(3, 8.0), (4, 10.0)],
    3: [(4, 2.0)],
    4: []
}
dist, prev = graph_theory.dijkstra_shortest_paths(adj, start=0)
print("Distances from node 0:", dist)
```

### 5. Fourier Analysis
```python
from claudemath import fourier_analysis
import math

# Compute Cooley-Tukey Radix-2 FFT
signal = [math.sin(2 * math.pi * 5 * i / 64) for i in range(64)]
fft_result = fourier_analysis.fft_cooley_tukey([complex(x, 0.0) for x in signal])
print(f"FFT bin 5 magnitude: {abs(fft_result[5]):.4f}")
```

---

## License

MIT License. Copyright (c) 2026 claudemath contributors.
