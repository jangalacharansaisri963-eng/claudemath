"""Probability module for claudemath.

Pure-Python implementation of discrete and continuous probability distributions (PDF, PMF,
CDF, Quantile/PPF, Mean, Variance), Bayes theorem, conditional probability,
expectation, variance, covariance, and probabilistic inequality bounds.
"""

from typing import List, Tuple, Optional, Callable
import math


# ----------------------------------------------------
# 1. Probability Fundamentals and Conditional Rules
# ----------------------------------------------------

def probability_union_two(p_a: float, p_b: float, p_a_and_b: float) -> float:
    """P(A or B) = P(A) + P(B) - P(A and B)."""
    return p_a + p_b - p_a_and_b


def conditional_probability(p_a_and_b: float, p_b: float) -> float:
    """Conditional probability P(A | B) = P(A and B) / P(B)."""
    if p_b <= 0:
        raise ValueError("P(B) must be strictly positive")
    return p_a_and_b / p_b


def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
    """Bayes' Theorem: P(A | B) = P(B | A) * P(A) / P(B)."""
    if p_b <= 0:
        raise ValueError("P(B) must be positive")
    return (p_b_given_a * p_a) / p_b


def law_of_total_probability(p_b_given_ai: List[float], p_ai: List[float]) -> float:
    """P(B) = sum(P(B | A_i) * P(A_i))."""
    if len(p_b_given_ai) != len(p_ai) or not p_ai:
        raise ValueError("Inputs must have matching non-empty lengths")
    return sum(cond * prior for cond, prior in zip(p_b_given_ai, p_ai))


def odds_from_probability(p: float) -> float:
    """Compute odds O = p / (1 - p)."""
    if not (0.0 <= p < 1.0):
        raise ValueError("Probability p must be in [0, 1)")
    return p / (1.0 - p)


def probability_from_odds(odds: float) -> float:
    """Compute probability p = odds / (1 + odds)."""
    if odds < 0:
        raise ValueError("Odds must be non-negative")
    return odds / (1.0 + odds)


def odds_ratio(p1: float, p2: float) -> float:
    """Odds ratio OR = odds(p1) / odds(p2)."""
    o1 = odds_from_probability(p1)
    o2 = odds_from_probability(p2)
    if o2 == 0:
        raise ZeroDivisionError("Odds of denominator is zero")
    return o1 / o2


def relative_risk(p_exposed: float, p_unexposed: float) -> float:
    """Relative risk RR = p_exposed / p_unexposed."""
    if p_unexposed <= 0:
        raise ValueError("p_unexposed must be positive")
    return p_exposed / p_unexposed


def markov_inequality_bound(mean_x: float, a: float) -> float:
    """Markov's inequality: P(X >= a) <= E[X] / a for non-negative X."""
    if a <= 0 or mean_x < 0:
        raise ValueError("a must be positive and mean_x non-negative")
    return min(1.0, mean_x / a)


def chebyshev_inequality_bound(variance_x: float, k: float) -> float:
    """Chebyshev's inequality: P(|X - mu| >= k*sigma) <= 1 / k^2."""
    if k <= 0:
        raise ValueError("k must be positive")
    return min(1.0, 1.0 / (k * k))


# ----------------------------------------------------
# 2. Discrete Distributions
# ----------------------------------------------------

# --- Bernoulli ---
def bernoulli_pmf(k: int, p: float) -> float:
    """Bernoulli PMF: P(X = 1) = p, P(X = 0) = 1 - p."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if k == 1:
        return p
    if k == 0:
        return 1.0 - p
    return 0.0


def bernoulli_mean(p: float) -> float:
    """Bernoulli distribution mean."""
    return p


def bernoulli_variance(p: float) -> float:
    """Bernoulli distribution variance p * (1 - p)."""
    return p * (1.0 - p)


# --- Binomial ---
def binomial_pmf(k: int, n: int, p: float) -> float:
    """Binomial PMF: C(n, k) * p^k * (1 - p)^{n - k}."""
    if k < 0 or k > n:
        return 0.0
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Binomial CDF: sum_{i=0}^k PMF(i)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(binomial_pmf(i, n, p) for i in range(k + 1))


def binomial_mean(n: int, p: float) -> float:
    """Binomial mean n * p."""
    return n * p


def binomial_variance(n: int, p: float) -> float:
    """Binomial variance n * p * (1 - p)."""
    return n * p * (1.0 - p)


# --- Poisson ---
def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: e^{-lambda} * lambda^k / k!."""
    if k < 0:
        return 0.0
    if lam <= 0:
        raise ValueError("lambda must be positive")
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    """Poisson CDF: sum_{i=0}^k PMF(i)."""
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_mean(lam: float) -> float:
    """Poisson mean lambda."""
    return lam


def poisson_variance(lam: float) -> float:
    """Poisson variance lambda."""
    return lam


# --- Geometric ---
def geometric_pmf(k: int, p: float) -> float:
    """Geometric PMF (number of trials until 1st success): (1 - p)^{k - 1} * p."""
    if k < 1:
        return 0.0
    return ((1.0 - p) ** (k - 1)) * p


def geometric_cdf(k: int, p: float) -> float:
    """Geometric CDF: 1 - (1 - p)^k."""
    if k < 1:
        return 0.0
    return 1.0 - (1.0 - p) ** k


def geometric_mean(p: float) -> float:
    """Geometric mean 1 / p."""
    if p <= 0:
        raise ValueError("p must be positive")
    return 1.0 / p


def geometric_variance(p: float) -> float:
    """Geometric variance (1 - p) / p^2."""
    if p <= 0:
        raise ValueError("p must be positive")
    return (1.0 - p) / (p * p)


# --- Hypergeometric ---
def hypergeometric_pmf(k: int, N: int, K: int, n: int) -> float:
    """Hypergeometric PMF: C(K, k) * C(N-K, n-k) / C(N, n)."""
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def hypergeometric_mean(N: int, K: int, n: int) -> float:
    """Hypergeometric mean n * K / N."""
    return n * K / N


def hypergeometric_variance(N: int, K: int, n: int) -> float:
    """Hypergeometric variance n * (K/N) * (1 - K/N) * ((N - n)/(N - 1))."""
    p = K / N
    return n * p * (1.0 - p) * ((N - n) / (N - 1))


# --- Negative Binomial ---
def negative_binomial_pmf(k: int, r: int, p: float) -> float:
    """Negative binomial PMF (number of failures k before r successes): C(k+r-1, k) * (1-p)^k * p^r."""
    if k < 0:
        return 0.0
    return math.comb(k + r - 1, k) * ((1.0 - p) ** k) * (p ** r)


def negative_binomial_mean(r: int, p: float) -> float:
    """Negative binomial mean r * (1 - p) / p."""
    return r * (1.0 - p) / p


def negative_binomial_variance(r: int, p: float) -> float:
    """Negative binomial variance r * (1 - p) / p^2."""
    return r * (1.0 - p) / (p * p)


# ----------------------------------------------------
# 3. Continuous Distributions
# ----------------------------------------------------

# --- Standard Normal & Normal ---
def standard_normal_pdf(z: float) -> float:
    """Standard normal PDF phi(z) = (1 / sqrt(2*pi)) * e^{-z^2 / 2}."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def standard_normal_cdf(z: float) -> float:
    """Standard normal CDF Phi(z) via error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def standard_normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (quantile / probit) using rational approximation."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    # Beasley-Springer-Moro or Abramowitz & Stegun approximation
    if p < 0.5:
        # F^{-1}(p) = -F^{-1}(1-p)
        return -standard_normal_quantile(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    # Coefficients for approximation
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    num = c0 + c1 * t + c2 * (t ** 2)
    den = 1.0 + d1 * t + d2 * (t ** 2) + d3 * (t ** 3)
    return t - num / den


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian normal PDF N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return standard_normal_pdf((x - mu) / sigma) / sigma


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian normal CDF."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return standard_normal_cdf((x - mu) / sigma)


def normal_quantile(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Gaussian normal quantile (inverse CDF)."""
    return mu + sigma * standard_normal_quantile(p)


# --- Uniform ---
def uniform_pdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Continuous uniform PDF on [a, b]."""
    if b <= a:
        raise ValueError("b must be strictly greater than a")
    return 1.0 / (b - a) if a <= x <= b else 0.0


def uniform_cdf(x: float, a: float = 0.0, b: float = 1.0) -> float:
    """Continuous uniform CDF on [a, b]."""
    if b <= a:
        raise ValueError("b must be strictly greater than a")
    if x < a:
        return 0.0
    if x > b:
        return 1.0
    return (x - a) / (b - a)


def uniform_mean(a: float, b: float) -> float:
    """Uniform distribution mean (a + b) / 2."""
    return (a + b) * 0.5


def uniform_variance(a: float, b: float) -> float:
    """Uniform distribution variance (b - a)^2 / 12."""
    return ((b - a) ** 2) / 12.0


# --- Exponential ---
def exponential_pdf(x: float, rate: float = 1.0) -> float:
    """Exponential distribution PDF: rate * e^{-rate * x}."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    if x < 0:
        return 0.0
    return rate * math.exp(-rate * x)


def exponential_cdf(x: float, rate: float = 1.0) -> float:
    """Exponential distribution CDF: 1 - e^{-rate * x}."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-rate * x)


def exponential_quantile(p: float, rate: float = 1.0) -> float:
    """Exponential distribution quantile: -ln(1 - p) / rate."""
    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0, 1)")
    return -math.log(1.0 - p) / rate


def exponential_mean(rate: float) -> float:
    """Exponential distribution mean 1 / rate."""
    return 1.0 / rate


def exponential_variance(rate: float) -> float:
    """Exponential distribution variance 1 / rate^2."""
    return 1.0 / (rate * rate)


# --- Gamma ---
def gamma_pdf(x: float, shape: float, scale: float = 1.0) -> float:
    """Gamma distribution PDF: x^{k-1} * e^{-x/theta} / (theta^k * Gamma(k))."""
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    if x <= 0:
        return 0.0
    log_p = (shape - 1.0) * math.log(x) - (x / scale) - shape * math.log(scale) - math.lgamma(shape)
    return math.exp(log_p)


def gamma_mean(shape: float, scale: float = 1.0) -> float:
    """Gamma distribution mean shape * scale."""
    return shape * scale


def gamma_variance(shape: float, scale: float = 1.0) -> float:
    """Gamma distribution variance shape * scale^2."""
    return shape * (scale ** 2)


# --- Beta ---
def beta_pdf(x: float, alpha: float, beta: float) -> float:
    """Beta distribution PDF: x^{alpha-1} * (1-x)^{beta-1} / B(alpha, beta)."""
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if not (0.0 < x < 1.0):
        return 0.0
    log_b = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_p = (alpha - 1.0) * math.log(x) + (beta - 1.0) * math.log(1.0 - x) - log_b
    return math.exp(log_p)


def beta_mean(alpha: float, beta: float) -> float:
    """Beta distribution mean alpha / (alpha + beta)."""
    return alpha / (alpha + beta)


def beta_variance(alpha: float, beta: float) -> float:
    """Beta distribution variance (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))."""
    s = alpha + beta
    return (alpha * beta) / (s * s * (s + 1.0))


# --- Cauchy ---
def cauchy_pdf(x: float, x0: float = 0.0, gamma: float = 1.0) -> float:
    """Cauchy distribution PDF: 1 / (pi * gamma * (1 + ((x - x0)/gamma)^2))."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return 1.0 / (math.pi * gamma * (1.0 + ((x - x0) / gamma) ** 2))


def cauchy_cdf(x: float, x0: float = 0.0, gamma: float = 1.0) -> float:
    """Cauchy distribution CDF: 0.5 + (1/pi) * arctan((x - x0)/gamma)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return 0.5 + (1.0 / math.pi) * math.atan((x - x0) / gamma)


def cauchy_quantile(p: float, x0: float = 0.0, gamma: float = 1.0) -> float:
    """Cauchy distribution quantile."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    return x0 + gamma * math.tan(math.pi * (p - 0.5))


# --- Log-Normal ---
def lognormal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Log-normal distribution PDF."""
    if x <= 0:
        return 0.0
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (1.0 / (x * sigma * math.sqrt(2.0 * math.pi))) * math.exp(-((math.log(x) - mu) ** 2) / (2.0 * sigma * sigma))


def lognormal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Log-normal distribution CDF."""
    if x <= 0:
        return 0.0
    return standard_normal_cdf((math.log(x) - mu) / sigma)


def lognormal_mean(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Log-normal distribution mean exp(mu + sigma^2 / 2)."""
    return math.exp(mu + 0.5 * sigma * sigma)


def lognormal_variance(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Log-normal distribution variance exp(2*mu + sigma^2) * (exp(sigma^2) - 1)."""
    return math.exp(2.0 * mu + sigma * sigma) * (math.exp(sigma * sigma) - 1.0)


# --- Weibull ---
def weibull_pdf(x: float, k: float, lam: float = 1.0) -> float:
    """Weibull distribution PDF."""
    if k <= 0 or lam <= 0:
        raise ValueError("k and lambda must be positive")
    if x < 0:
        return 0.0
    return (k / lam) * ((x / lam) ** (k - 1.0)) * math.exp(-((x / lam) ** k))


def weibull_cdf(x: float, k: float, lam: float = 1.0) -> float:
    """Weibull distribution CDF: 1 - e^{-(x/lam)^k}."""
    if k <= 0 or lam <= 0:
        raise ValueError("k and lambda must be positive")
    if x < 0:
        return 0.0
    return 1.0 - math.exp(-((x / lam) ** k))


def weibull_quantile(p: float, k: float, lam: float = 1.0) -> float:
    """Weibull distribution quantile."""
    if not (0.0 <= p < 1.0):
        raise ValueError("p must be in [0, 1)")
    return lam * ((-math.log(1.0 - p)) ** (1.0 / k))


# --- Laplace ---
def laplace_pdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """Laplace (double exponential) PDF: (1 / (2b)) * e^{-|x - mu| / b}."""
    if b <= 0:
        raise ValueError("b must be positive")
    return (0.5 / b) * math.exp(-abs(x - mu) / b)


def laplace_cdf(x: float, mu: float = 0.0, b: float = 1.0) -> float:
    """Laplace distribution CDF."""
    if b <= 0:
        raise ValueError("b must be positive")
    if x < mu:
        return 0.5 * math.exp((x - mu) / b)
    return 1.0 - 0.5 * math.exp(-(x - mu) / b)


def laplace_mean(mu: float = 0.0, b: float = 1.0) -> float:
    """Laplace distribution mean."""
    return mu


def laplace_variance(mu: float = 0.0, b: float = 1.0) -> float:
    """Laplace distribution variance 2 * b^2."""
    return 2.0 * b * b


# --- Pareto ---
def pareto_pdf(x: float, alpha: float, x_m: float = 1.0) -> float:
    """Pareto Type I distribution PDF: alpha * x_m^alpha / x^{alpha + 1} for x >= x_m."""
    if alpha <= 0 or x_m <= 0:
        raise ValueError("alpha and x_m must be positive")
    if x < x_m:
        return 0.0
    return (alpha * (x_m ** alpha)) / (x ** (alpha + 1.0))


def pareto_cdf(x: float, alpha: float, x_m: float = 1.0) -> float:
    """Pareto Type I distribution CDF: 1 - (x_m / x)^alpha."""
    if alpha <= 0 or x_m <= 0:
        raise ValueError("alpha and x_m must be positive")
    if x < x_m:
        return 0.0
    return 1.0 - (x_m / x) ** alpha


def pareto_mean(alpha: float, x_m: float = 1.0) -> float:
    """Pareto mean alpha * x_m / (alpha - 1) for alpha > 1."""
    if alpha <= 1.0:
        raise ValueError("Mean infinite for alpha <= 1")
    return alpha * x_m / (alpha - 1.0)


# --- Chi-Square ---
def chi_square_pdf(x: float, k: int) -> float:
    """Chi-square distribution PDF with k degrees of freedom."""
    if k <= 0:
        raise ValueError("Degrees of freedom k must be positive")
    return gamma_pdf(x, shape=k * 0.5, scale=2.0)


def chi_square_mean(k: int) -> float:
    """Chi-square mean equals degrees of freedom k."""
    return float(k)


def chi_square_variance(k: int) -> float:
    """Chi-square variance 2 * k."""
    return 2.0 * k
