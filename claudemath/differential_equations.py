"""Differential equations module for claudemath.

Pure-Python implementation of ODE initial value problem solvers (Euler, Heun, RK4,
RKF45, DOPRI5, Adams-Bashforth, Adams-Moulton), ODE systems (Lorenz, Lotka-Volterra,
SIR, Van der Pol), symplectic integrators (Verlet, Leapfrog), boundary value solvers
(Shooting, Finite Difference), and PDE solvers (FTCS, Crank-Nicolson, Wave, Laplace).
"""

from typing import List, Tuple, Callable, Optional, Dict
import math


# ----------------------------------------------------
# 1. First-Order Scalar ODE Solvers: y' = f(t, y)
# ----------------------------------------------------

def ode_euler_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    """Single forward Euler step: y_{n+1} = y_n + h * f(t_n, y_n)."""
    return y + h * f(t, y)


def ode_solve_euler(f: Callable[[float, float], float],
                    t0: float, y0: float, t_end: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """Forward Euler ODE solver over [t0, t_end]."""
    h = (t_end - t0) / n_steps
    ts = [t0]
    ys = [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        y = ode_euler_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def ode_heun_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    """Single Heun (explicit trapezoidal / modified Euler) step."""
    k1 = f(t, y)
    y_pred = y + h * k1
    k2 = f(t + h, y_pred)
    return y + 0.5 * h * (k1 + k2)


def ode_solve_heun(f: Callable[[float, float], float],
                   t0: float, y0: float, t_end: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """Heun's 2nd-order predictor-corrector solver."""
    h = (t_end - t0) / n_steps
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        y = ode_heun_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def ode_midpoint_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    """Explicit midpoint (RK2) step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    return y + h * k2


def ode_rk4_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    """Classical 4th-order Runge-Kutta (RK4) single step."""
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def ode_solve_rk4(f: Callable[[float, float], float],
                  t0: float, y0: float, t_end: float, n_steps: int) -> Tuple[List[float], List[float]]:
    """Solve scalar ODE y' = f(t, y) using RK4."""
    h = (t_end - t0) / n_steps
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        y = ode_rk4_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def ode_rk_38_step(f: Callable[[float, float], float], t: float, y: float, h: float) -> float:
    """Runge-Kutta 3/8 rule step: 4th order."""
    k1 = f(t, y)
    k2 = f(t + h / 3.0, y + (h / 3.0) * k1)
    k3 = f(t + 2.0 * h / 3.0, y - (h / 3.0) * k1 + h * k2)
    k4 = f(t + h, y + h * k1 - h * k2 + h * k3)
    return y + (h / 8.0) * (k1 + 3.0 * k2 + 3.0 * k3 + k4)


def ode_backward_euler_step(f: Callable[[float, float], float],
                            t: float, y: float, h: float,
                            tol: float = 1e-8, max_iter: int = 50) -> float:
    """Implicit backward Euler step solved via fixed-point iteration."""
    t_next = t + h
    y_next = y + h * f(t, y)  # Predictor
    for _ in range(max_iter):
        y_cand = y + h * f(t_next, y_next)
        if abs(y_cand - y_next) < tol:
            return y_cand
        y_next = y_cand
    return y_next


# ----------------------------------------------------
# 2. Adaptive Step Size Integrators
# ----------------------------------------------------

def ode_solve_rkf45(f: Callable[[float, float], float],
                    t0: float, y0: float, t_end: float,
                    tol: float = 1e-6, h_init: float = 0.01) -> Tuple[List[float], List[float]]:
    """Adaptive Runge-Kutta-Fehlberg 4(5) ODE solver."""
    ts = [t0]
    ys = [y0]
    t, y = t0, y0
    h = h_init
    while t < t_end:
        if t + h > t_end:
            h = t_end - t
        # Fehlberg coefficients
        k1 = h * f(t, y)
        k2 = h * f(t + h / 4.0, y + k1 / 4.0)
        k3 = h * f(t + 3.0 * h / 8.0, y + 3.0 * k1 / 32.0 + 9.0 * k2 / 32.0)
        k4 = h * f(t + 12.0 * h / 13.0, y + 1932.0 * k1 / 2197.0 - 7200.0 * k2 / 2197.0 + 7296.0 * k3 / 2197.0)
        k5 = h * f(t + h, y + 439.0 * k1 / 216.0 - 8.0 * k2 + 3680.0 * k3 / 513.0 - 845.0 * k4 / 4104.0)
        k6 = h * f(t + h / 2.0, y - 8.0 * k1 / 27.0 + 2.0 * k2 - 3544.0 * k3 / 2565.0 + 1859.0 * k4 / 4104.0 - 11.0 * k5 / 40.0)

        y4 = y + 25.0 * k1 / 216.0 + 1408.0 * k3 / 2565.0 + 2197.0 * k4 / 4104.0 - k5 / 5.0
        y5 = y + 16.0 * k1 / 135.0 + 6656.0 * k3 / 12825.0 + 28561.0 * k4 / 56430.0 - 9.0 * k5 / 50.0 + 2.0 * k6 / 55.0

        error = abs(y5 - y4)
        if error <= tol or h < 1e-12:
            t += h
            y = y5
            ts.append(t)
            ys.append(y)
        # Optimal step scaling
        s = 0.84 * ((tol / (error + 1e-15)) ** 0.25)
        h = max(1e-10, min(0.5, h * max(0.1, min(4.0, s))))
    return ts, ys


# ----------------------------------------------------
# 3. Systems of ODEs: y' = F(t, y)
# ----------------------------------------------------

def ode_system_rk4_step(f: Callable[[float, List[float]], List[float]],
                        t: float, y: List[float], h: float) -> List[float]:
    """Single RK4 step for vector ODE system."""
    n = len(y)
    k1 = f(t, y)
    y_k2 = [y[i] + 0.5 * h * k1[i] for i in range(n)]
    k2 = f(t + 0.5 * h, y_k2)
    y_k3 = [y[i] + 0.5 * h * k2[i] for i in range(n)]
    k3 = f(t + 0.5 * h, y_k3)
    y_k4 = [y[i] + h * k3[i] for i in range(n)]
    k4 = f(t + h, y_k4)
    return [y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(n)]


def ode_system_solve_rk4(f: Callable[[float, List[float]], List[float]],
                         t0: float, y0: List[float], t_end: float, n_steps: int) -> Tuple[List[float], List[List[float]]]:
    """Solve system of ODEs using RK4."""
    h = (t_end - t0) / n_steps
    ts = [t0]
    ys = [list(y0)]
    t = t0
    y = list(y0)
    for _ in range(n_steps):
        y = ode_system_rk4_step(f, t, y, h)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


# ----------------------------------------------------
# 4. Standard Benchmark Physical ODE Systems
# ----------------------------------------------------

def lorenz_system_derivative(t: float, state: List[float],
                             sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> List[float]:
    """Lorenz chaotic attractor derivatives [dx/dt, dy/dt, dz/dt]."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def lotka_volterra_derivative(t: float, state: List[float],
                              alpha: float = 1.0, beta: float = 0.1,
                              gamma: float = 1.5, delta: float = 0.075) -> List[float]:
    """Lotka-Volterra predator-prey system [prey=x, predator=y]."""
    x, y = state
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return [dx, dy]


def sir_epidemic_derivative(t: float, state: List[float],
                            beta: float = 0.3, gamma: float = 0.1, N: float = 1000.0) -> List[float]:
    """SIR epidemiological model [Susceptible, Infected, Recovered]."""
    S, I, R = state
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]


def van_der_pol_derivative(t: float, state: List[float], mu: float = 1.0) -> List[float]:
    """Van der Pol oscillator [x, y = dx/dt]: dx/dt = y, dy/dt = mu*(1 - x^2)*y - x."""
    x, y = state
    return [y, mu * (1.0 - x * x) * y - x]


def damped_pendulum_derivative(t: float, state: List[float],
                              gamma: float = 0.2, omega0: float = 1.0) -> List[float]:
    """Nonlinear pendulum with viscous damping [theta, omega = dtheta/dt]."""
    theta, omega = state
    return [omega, -gamma * omega - (omega0 ** 2) * math.sin(theta)]


# ----------------------------------------------------
# 5. Symplectic Hamiltonian Integrators
# ----------------------------------------------------

def symplectic_euler_step(q: float, p: float, h: float,
                          grad_V: Callable[[float], float], m: float = 1.0) -> Tuple[float, float]:
    """Symplectic Euler step preserving phase space volume: p_{n+1} = p_n - h*V'(q_n), q_{n+1} = q_n + h*p_{n+1}/m."""
    p_next = p - h * grad_V(q)
    q_next = q + h * (p_next / m)
    return q_next, p_next


def velocity_verlet_step(q: float, v: float, h: float,
                         accel_func: Callable[[float], float]) -> Tuple[float, float]:
    """Velocity Verlet step for molecular dynamics / orbital mechanics."""
    a = accel_func(q)
    q_next = q + v * h + 0.5 * a * h * h
    a_next = accel_func(q_next)
    v_next = v + 0.5 * (a + a_next) * h
    return q_next, v_next


def leapfrog_step(q: float, v_half: float, h: float,
                  accel_func: Callable[[float], float]) -> Tuple[float, float, float]:
    """Leapfrog integration step: returns (q_next, v_half_next, v_full)."""
    q_next = q + v_half * h
    a_next = accel_func(q_next)
    v_half_next = v_half + a_next * h
    v_full = 0.5 * (v_half + v_half_next)
    return q_next, v_half_next, v_full


# ----------------------------------------------------
# 6. Boundary Value Problems (BVP) and PDEs
# ----------------------------------------------------

def bvp_shooting_linear_step(y0_prime_guess1: float, y0_prime_guess2: float,
                             target_y_end: float, actual_y_end1: float, actual_y_end2: float) -> float:
    """Secant step to find optimal initial slope y'(0) for shooting method."""
    if abs(actual_y_end2 - actual_y_end1) < 1e-14:
        return y0_prime_guess2
    return y0_prime_guess2 + (target_y_end - actual_y_end2) * (y0_prime_guess2 - y0_prime_guess1) / (actual_y_end2 - actual_y_end1)


def pde_heat_1d_ftcs_step(u: List[float], alpha: float, dx: float, dt: float) -> List[float]:
    """1D Heat Equation forward-time central-space (FTCS) explicit time step with Dirichlet 0 boundaries."""
    r = alpha * dt / (dx * dx)
    if r > 0.5:
        raise ValueError(f"FTCS unstable: r = {r} > 0.5")
    n = len(u)
    u_next = [0.0] * n
    for i in range(1, n - 1):
        u_next[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1])
    return u_next


def pde_wave_1d_step(u_curr: List[float], u_prev: List[float],
                     c: float, dx: float, dt: float) -> List[float]:
    """1D Wave Equation explicit finite difference step (CFL stability condition c*dt/dx <= 1)."""
    courant = c * dt / dx
    if courant > 1.0:
        raise ValueError(f"CFL condition violated: courant = {courant} > 1.0")
    c_sq = courant * courant
    n = len(u_curr)
    u_next = [0.0] * n
    for i in range(1, n - 1):
        u_next[i] = 2.0 * u_curr[i] - u_prev[i] + c_sq * (u_curr[i + 1] - 2.0 * u_curr[i] + u_curr[i - 1])
    return u_next


def pde_laplace_2d_jacobi_step(grid: List[List[float]]) -> List[List[float]]:
    """Single Jacobi relaxation step for 2D Laplace equation d^2u/dx^2 + d^2u/dy^2 = 0."""
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [row[:] for row in grid]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            new_grid[i][j] = 0.25 * (grid[i + 1][j] + grid[i - 1][j] + grid[i][j + 1] + grid[i][j - 1])
    return new_grid
def adams_bashforth_2step(y1: float, f1: float, f0: float, h: float) -> float:
    """2-step explicit Adams-Bashforth: y_{n+1} = y_n + (h/2)*(3*f_n - f_{n-1})."""
    return y1 + 0.5 * h * (3.0 * f1 - f0)


def adams_bashforth_3step(y2: float, f2: float, f1: float, f0: float, h: float) -> float:
    """3-step explicit Adams-Bashforth: y_{n+1} = y_n + (h/12)*(23*f_n - 16*f_{n-1} + 5*f_{n-2})."""
    return y2 + (h / 12.0) * (23.0 * f2 - 16.0 * f1 + 5.0 * f0)


def adams_bashforth_4step(y3: float, f3: float, f2: float, f1: float, f0: float, h: float) -> float:
    """4-step explicit Adams-Bashforth: y_{n+1} = y_n + (h/24)*(55*f_n - 59*f_{n-1} + 37*f_{n-2} - 9*f_{n-3})."""
    return y3 + (h / 24.0) * (55.0 * f3 - 59.0 * f2 + 37.0 * f1 - 9.0 * f0)


def adams_moulton_2step(y1: float, f_next: float, f1: float, h: float) -> float:
    """2-step implicit Adams-Moulton (Trapezoidal): y_{n+1} = y_n + (h/2)*(f_{n+1} + f_n)."""
    return y1 + 0.5 * h * (f_next + f1)


def adams_moulton_3step(y2: float, f_next: float, f2: float, f1: float, h: float) -> float:
    """3-step implicit Adams-Moulton: y_{n+1} = y_n + (h/12)*(5*f_{n+1} + 8*f_n - f_{n-1})."""
    return y2 + (h / 12.0) * (5.0 * f_next + 8.0 * f2 - f1)


def adams_moulton_4step(y3: float, f_next: float, f3: float, f2: float, f1: float, h: float) -> float:
    """4-step implicit Adams-Moulton: y_{n+1} = y_n + (h/24)*(9*f_{n+1} + 19*f_n - 5*f_{n-1} + f_{n-2})."""
    return y3 + (h / 24.0) * (9.0 * f_next + 19.0 * f3 - 5.0 * f2 + f1)


def milne_simpson_step(y_prev2: float, f_next: float, f_curr: float, f_prev: float, h: float) -> float:
    """Milne-Simpson corrector step: y_{n+1} = y_{n-1} + (h/3)*(f_{n+1} + 4*f_n + f_{n-1})."""
    return y_prev2 + (h / 3.0) * (f_next + 4.0 * f_curr + f_prev)


def robertson_stiff_derivative(t: float, state: List[float]) -> List[float]:
    """Robertson's classic benchmark system for stiff ODE solvers."""
    y1, y2, y3 = state
    dy1 = -0.04 * y1 + 1e4 * y2 * y3
    dy2 = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * (y2 ** 2)
    dy3 = 3e7 * (y2 ** 2)
    return [dy1, dy2, dy3]


def fitzhugh_nagumo_derivative(t: float, state: List[float],
                               I_ext: float = 0.5, a: float = 0.7, b: float = 0.8, tau: float = 12.5) -> List[float]:
    """FitzHugh-Nagumo model of neuronal excitability [voltage v, recovery w]."""
    v, w = state
    dv = v - (v ** 3) / 3.0 - w + I_ext
    dw = (v + a - b * w) / tau
    return [dv, dw]


def brusselator_derivative(t: float, state: List[float], a: float = 1.0, b: float = 3.0) -> List[float]:
    """Brusselator chemical reaction model: dx/dt = a + x^2*y - b*x - x, dy/dt = b*x - x^2*y."""
    x, y = state
    dx = a + x * x * y - (b + 1.0) * x
    dy = b * x - x * x * y
    return [dx, dy]


def duffing_oscillator_derivative(t: float, state: List[float],
                                  alpha: float = 1.0, beta: float = 5.0,
                                  gamma: float = 0.3, delta: float = 0.2, omega: float = 1.2) -> List[float]:
    """Duffing non-linear forced oscillator: x'' + delta*x' - alpha*x + beta*x^3 = gamma*cos(omega*t)."""
    x, v = state
    dx = v
    dv = -delta * v + alpha * x - beta * (x ** 3) + gamma * math.cos(omega * t)
    return [dx, dv]


def rossler_attractor_derivative(t: float, state: List[float],
                                 a: float = 0.2, b: float = 0.2, c: float = 5.7) -> List[float]:
    """Rossler chaotic attractor [dx, dy, dz]."""
    x, y, z = state
    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)
    return [dx, dy, dz]


def harmonic_oscillator_derivative(t: float, state: List[float], k: float = 1.0, m: float = 1.0) -> List[float]:
    """Simple linear harmonic oscillator [position x, velocity v]."""
    x, v = state
    return [v, -(k / m) * x]


def hamiltonian_total_energy(q: float, p: float, V_func: Callable[[float], float], m: float = 1.0) -> float:
    """Total Hamiltonian energy H(q, p) = p^2 / (2m) + V(q)."""
    return (p * p) / (2.0 * m) + V_func(q)


def verlet_compute_kinetic_energy(v: float, m: float = 1.0) -> float:
    """Kinetic energy E_k = 0.5 * m * v^2."""
    return 0.5 * m * v * v


def yoshida_4th_order_symplectic_step(q: float, p: float, h: float,
                                      grad_V: Callable[[float], float], m: float = 1.0) -> Tuple[float, float]:
    """Yoshida 4th-order symplectic integrator step for Hamiltonian H = T(p) + V(q)."""
    cuberoot2 = 2.0 ** (1.0 / 3.0)
    w0 = -cuberoot2 / (2.0 - cuberoot2)
    w1 = 1.0 / (2.0 - cuberoot2)
    c = [0.5 * w1, 0.5 * (w0 + w1), 0.5 * (w0 + w1), 0.5 * w1]
    d = [w1, w0, w1]

    # Sub-steps
    q = q + c[0] * h * (p / m)
    for i in range(3):
        p = p - d[i] * h * grad_V(q)
        q = q + c[i + 1] * h * (p / m)
    return q, p


def pde_laplace_2d_gauss_seidel_step(grid: List[List[float]]) -> List[List[float]]:
    """In-place Gauss-Seidel relaxation step for 2D Laplace equation."""
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [row[:] for row in grid]
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            new_grid[i][j] = 0.25 * (new_grid[i + 1][j] + new_grid[i - 1][j] + new_grid[i][j + 1] + new_grid[i][j - 1])
    return new_grid


def pde_poisson_2d_sor_step(grid: List[List[float]], source: List[List[float]],
                            h: float, omega: float = 1.5) -> List[List[float]]:
    """Successive Over-Relaxation (SOR) step for Poisson equation d^2u/dx^2 + d^2u/dy^2 = f."""
    rows = len(grid)
    cols = len(grid[0])
    new_grid = [row[:] for row in grid]
    h_sq = h * h
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gs_val = 0.25 * (new_grid[i + 1][j] + new_grid[i - 1][j] + new_grid[i][j + 1] + new_grid[i][j - 1] - h_sq * source[i][j])
            new_grid[i][j] = (1.0 - omega) * new_grid[i][j] + omega * gs_val
    return new_grid


def pde_advection_1d_upwind_step(u: List[float], c: float, dx: float, dt: float) -> List[float]:
    """1D linear advection du/dt + c*du/dx = 0 using upwind scheme."""
    courant = c * dt / dx
    n = len(u)
    u_next = [0.0] * n
    if c >= 0:
        for i in range(1, n):
            u_next[i] = u[i] - courant * (u[i] - u[i - 1])
        u_next[0] = u[0]
    else:
        for i in range(n - 1):
            u_next[i] = u[i] - courant * (u[i + 1] - u[i])
        u_next[-1] = u[-1]
    return u_next


def pde_advection_1d_lax_friedrichs_step(u: List[float], c: float, dx: float, dt: float) -> List[float]:
    """1D linear advection via Lax-Friedrichs conservative scheme."""
    courant = c * dt / dx
    n = len(u)
    u_next = [0.0] * n
    for i in range(1, n - 1):
        u_next[i] = 0.5 * (u[i + 1] + u[i - 1]) - 0.5 * courant * (u[i + 1] - u[i - 1])
    u_next[0] = u[0]
    u_next[-1] = u[-1]
    return u_next


def airy_ode_derivative(x: float, state: List[float]) -> List[float]:
    """Airy differential equation: y'' - x*y = 0 [y, y']."""
    y, yp = state
    return [yp, x * y]


def bessel_ode_derivative(x: float, state: List[float], n: float = 0.0) -> List[float]:
    """Bessel differential equation: x^2 y'' + x y' + (x^2 - n^2) y = 0 [y, y']."""
    y, yp = state
    if abs(x) < 1e-12:
        return [yp, 0.0]
    ypp = -(1.0 / x) * yp - (1.0 - (n * n) / (x * x)) * y
    return [yp, ypp]


def legendre_ode_derivative(x: float, state: List[float], n: int = 1) -> List[float]:
    """Legendre differential equation: (1 - x^2) y'' - 2x y' + n(n + 1) y = 0 [y, y']."""
    y, yp = state
    denom = 1.0 - x * x
    if abs(denom) < 1e-12:
        return [yp, 0.0]
    ypp = (2.0 * x * yp - n * (n + 1) * y) / denom
    return [yp, ypp]


def chebyshev_ode_derivative(x: float, state: List[float], n: int = 1) -> List[float]:
    """Chebyshev differential equation: (1 - x^2) y'' - x y' + n^2 y = 0 [y, y']."""
    y, yp = state
    denom = 1.0 - x * x
    if abs(denom) < 1e-12:
        return [yp, 0.0]
    ypp = (x * yp - (n * n) * y) / denom
    return [yp, ypp]


def hermite_ode_derivative(x: float, state: List[float], n: int = 1) -> List[float]:
    """Hermite differential equation: y'' - 2x y' + 2n y = 0 [y, y']."""
    y, yp = state
    ypp = 2.0 * x * yp - 2.0 * n * y
    return [yp, ypp]
def laguerre_ode_derivative(x: float, state: List[float], n: int = 1) -> List[float]:
    """Laguerre differential equation: x y'' + (1 - x) y' + n y = 0 [y, y']."""
    y, yp = state
    if abs(x) < 1e-12:
        return [yp, 0.0]
    ypp = -((1.0 - x) / x) * yp - (n / x) * y
    return [yp, ypp]


def ode_richardson_extrapolation_step(f: Callable[[float, float], float],
                                      t: float, y: float, h: float) -> Tuple[float, float]:
    """Richardson extrapolation using two half-steps and one full RK4 step: returns (extrapolated_y, estimated_error)."""
    y_full = ode_rk4_step(f, t, y, h)
    y_half1 = ode_rk4_step(f, t, y, 0.5 * h)
    y_half2 = ode_rk4_step(f, t + 0.5 * h, y_half1, 0.5 * h)
    error = (y_half2 - y_full) / 15.0
    extrapolated = y_half2 + error
    return extrapolated, abs(error)
