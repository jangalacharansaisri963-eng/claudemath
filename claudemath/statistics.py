"""Statistics module for claudemath.

Pure-Python implementation of descriptive statistics, dispersion metrics,
higher-order moments, quantiles, correlation measures, linear/polynomial regression,
distribution scaling/transforms, and hypothesis test statistics.
"""

from typing import List, Tuple, Optional, Dict
import math


# ----------------------------------------------------
# 1. Measures of Central Tendency
# ----------------------------------------------------

def mean(data: List[float]) -> float:
    """Compute arithmetic mean."""
    if not data:
        raise ValueError("data cannot be empty")
    return sum(data) / len(data)


def weighted_mean(data: List[float], weights: List[float]) -> float:
    """Compute weighted arithmetic mean."""
    if len(data) != len(weights) or not data:
        raise ValueError("data and weights must have matching non-empty lengths")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("sum of weights must be positive")
    return sum(x * w for x, w in zip(data, weights)) / w_sum


def geometric_mean(data: List[float]) -> float:
    """Compute geometric mean of positive numbers."""
    if not data:
        raise ValueError("data cannot be empty")
    for x in data:
        if x <= 0:
            raise ValueError("geometric mean requires strictly positive numbers")
    log_sum = sum(math.log(x) for x in data)
    return math.exp(log_sum / len(data))


def harmonic_mean(data: List[float]) -> float:
    """Compute harmonic mean of positive numbers."""
    if not data:
        raise ValueError("data cannot be empty")
    for x in data:
        if x <= 0:
            raise ValueError("harmonic mean requires strictly positive numbers")
    return len(data) / sum(1.0 / x for x in data)


def contraharmonic_mean(data: List[float]) -> float:
    """Compute contraharmonic mean: sum(x^2) / sum(x)."""
    if not data:
        raise ValueError("data cannot be empty")
    denom = sum(data)
    if denom == 0:
        raise ZeroDivisionError("sum of data is zero")
    return sum(x * x for x in data) / denom


def quadratic_mean(data: List[float]) -> float:
    """Compute quadratic mean (root mean square / RMS)."""
    if not data:
        raise ValueError("data cannot be empty")
    return math.sqrt(sum(x * x for x in data) / len(data))


def median(data: List[float]) -> float:
    """Compute sample median."""
    if not data:
        raise ValueError("data cannot be empty")
    s = sorted(data)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def median_low(data: List[float]) -> float:
    """Return low median of data."""
    if not data:
        raise ValueError("data cannot be empty")
    s = sorted(data)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return float(s[n // 2 - 1])


def median_high(data: List[float]) -> float:
    """Return high median of data."""
    if not data:
        raise ValueError("data cannot be empty")
    s = sorted(data)
    return float(s[len(s) // 2])


def mode(data: List[float]) -> float:
    """Return single mode (most common value)."""
    if not data:
        raise ValueError("data cannot be empty")
    counts: Dict[float, int] = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def multimode(data: List[float]) -> List[float]:
    """Return all modes in order of first appearance."""
    if not data:
        return []
    counts: Dict[float, int] = {}
    for x in data:
        counts[x] = counts.get(x, 0) + 1
    max_count = max(counts.values())
    return [k for k, v in counts.items() if v == max_count]


def midrange(data: List[float]) -> float:
    """Compute midrange: (min(x) + max(x)) / 2."""
    if not data:
        raise ValueError("data cannot be empty")
    return (min(data) + max(data)) / 2.0


def trimmed_mean(data: List[float], proportion_to_cut: float = 0.1) -> float:
    """Compute trimmed mean removing proportion_to_cut from each tail."""
    if not data:
        raise ValueError("data cannot be empty")
    if not (0.0 <= proportion_to_cut < 0.5):
        raise ValueError("proportion_to_cut must be in [0, 0.5)")
    s = sorted(data)
    n = len(s)
    k = int(n * proportion_to_cut)
    trimmed = s[k:n - k]
    return sum(trimmed) / len(trimmed)


def winsorized_mean(data: List[float], proportion_to_cut: float = 0.1) -> float:
    """Compute Winsorized mean replacing extreme tail values."""
    if not data:
        raise ValueError("data cannot be empty")
    if not (0.0 <= proportion_to_cut < 0.5):
        raise ValueError("proportion_to_cut must be in [0, 0.5)")
    s = sorted(data)
    n = len(s)
    k = int(n * proportion_to_cut)
    if k == 0:
        return mean(data)
    low_val = s[k]
    high_val = s[n - 1 - k]
    w = [low_val] * k + s[k:n - k] + [high_val] * k
    return mean(w)


def trimean(data: List[float]) -> float:
    """Compute Tukey's trimean: (Q1 + 2*Q2 + Q3) / 4."""
    q1 = quantile(data, 0.25)
    q2 = median(data)
    q3 = quantile(data, 0.75)
    return (q1 + 2.0 * q2 + q3) / 4.0


def midhinge(data: List[float]) -> float:
    """Compute midhinge: (Q1 + Q3) / 2."""
    q1 = quantile(data, 0.25)
    q3 = quantile(data, 0.75)
    return (q1 + q3) / 2.0


# ----------------------------------------------------
# 2. Measures of Dispersion and Variability
# ----------------------------------------------------

def population_variance(data: List[float]) -> float:
    """Compute population variance sigma^2."""
    if not data:
        raise ValueError("data cannot be empty")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)


def sample_variance(data: List[float]) -> float:
    """Compute unbiased sample variance s^2 (ddof = 1)."""
    if len(data) < 2:
        raise ValueError("sample variance requires at least 2 points")
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def population_std_dev(data: List[float]) -> float:
    """Compute population standard deviation sigma."""
    return math.sqrt(population_variance(data))


def sample_std_dev(data: List[float]) -> float:
    """Compute sample standard deviation s."""
    return math.sqrt(sample_variance(data))


def standard_error_mean(data: List[float]) -> float:
    """Standard error of the mean SE = s / sqrt(n)."""
    if len(data) < 2:
        raise ValueError("data must have at least 2 points")
    return sample_std_dev(data) / math.sqrt(len(data))


def mean_absolute_deviation(data: List[float]) -> float:
    """Compute mean absolute deviation around the mean."""
    if not data:
        raise ValueError("data cannot be empty")
    m = mean(data)
    return sum(abs(x - m) for x in data) / len(data)


def median_absolute_deviation(data: List[float]) -> float:
    """Compute median absolute deviation (MAD) around the median."""
    if not data:
        raise ValueError("data cannot be empty")
    med = median(data)
    devs = [abs(x - med) for x in data]
    return median(devs)


def range_stat(data: List[float]) -> float:
    """Compute range max(x) - min(x)."""
    if not data:
        raise ValueError("data cannot be empty")
    return max(data) - min(data)


def iqr(data: List[float]) -> float:
    """Interquartile range IQR = Q3 - Q1."""
    return quantile(data, 0.75) - quantile(data, 0.25)


def semi_iqr(data: List[float]) -> float:
    """Semi-interquartile range = (Q3 - Q1) / 2."""
    return iqr(data) / 2.0


def coefficient_of_variation(data: List[float]) -> float:
    """Coefficient of variation CV = s / mean."""
    m = mean(data)
    if m == 0:
        raise ZeroDivisionError("mean is zero")
    return sample_std_dev(data) / m


def variance_to_mean_ratio(data: List[float]) -> float:
    """Index of dispersion VMR = variance / mean."""
    m = mean(data)
    if m == 0:
        raise ZeroDivisionError("mean is zero")
    return sample_variance(data) / m


def relative_mean_difference(data: List[float]) -> float:
    """Gini relative mean difference."""
    n = len(data)
    if n < 2:
        return 0.0
    m = mean(data)
    if m == 0:
        return 0.0
    total = sum(abs(x - y) for x in data for y in data)
    return total / (n * n * m)


def gini_coefficient(data: List[float]) -> float:
    """Gini coefficient of inequality."""
    n = len(data)
    if n == 0:
        raise ValueError("data cannot be empty")
    s = sorted(data)
    if s[0] < 0:
        raise ValueError("Gini coefficient requires non-negative values")
    total_val = sum(s)
    if total_val == 0:
        return 0.0
    index_sum = sum((i + 1) * x for i, x in enumerate(s))
    return (2.0 * index_sum) / (n * total_val) - (n + 1.0) / n


# ----------------------------------------------------
# 3. Higher Moments and Distribution Shape
# ----------------------------------------------------

def raw_moment(data: List[float], order: int) -> float:
    """Compute raw moment E[X^k]."""
    if not data:
        raise ValueError("data cannot be empty")
    return sum(x ** order for x in data) / len(data)


def central_moment(data: List[float], order: int) -> float:
    """Compute central moment mu_k = E[(X - mu)^k]."""
    if not data:
        raise ValueError("data cannot be empty")
    m = mean(data)
    return sum((x - m) ** order for x in data) / len(data)


def standardized_moment(data: List[float], order: int) -> float:
    """Compute standardized moment mu_k / sigma^k."""
    sd = population_std_dev(data)
    if sd == 0:
        raise ZeroDivisionError("standard deviation is zero")
    return central_moment(data, order) / (sd ** order)


def skewness_sample(data: List[float]) -> float:
    """Compute sample Fisher-Pearson skewness coefficient g1."""
    n = len(data)
    if n < 3:
        raise ValueError("skewness requires at least 3 data points")
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / n
    m3 = sum((x - m) ** 3 for x in data) / n
    if m2 == 0:
        return 0.0
    g1 = m3 / (m2 ** 1.5)
    return g1 * math.sqrt(n * (n - 1)) / (n - 2)


def skewness_pearson(data: List[float]) -> float:
    """Pearson's first skewness coefficient (mean - mode) / s."""
    sd = sample_std_dev(data)
    if sd == 0:
        return 0.0
    return (mean(data) - mode(data)) / sd


def skewness_bowley(data: List[float]) -> float:
    """Bowley quartile skewness coefficient: ((Q3 - Q2) - (Q2 - Q1)) / (Q3 - Q1)."""
    q1 = quantile(data, 0.25)
    q2 = median(data)
    q3 = quantile(data, 0.75)
    denom = q3 - q1
    if denom == 0:
        return 0.0
    return (q3 + q1 - 2.0 * q2) / denom


def kurtosis_sample(data: List[float]) -> float:
    """Sample kurtosis (unadjusted, normal distribution ~ 3)."""
    n = len(data)
    if n < 4:
        raise ValueError("kurtosis requires at least 4 points")
    return standardized_moment(data, 4)


def excess_kurtosis(data: List[float]) -> float:
    """Sample excess kurtosis (normal distribution = 0)."""
    n = len(data)
    if n < 4:
        raise ValueError("excess kurtosis requires at least 4 points")
    m = mean(data)
    m2 = sum((x - m) ** 2 for x in data) / n
    m4 = sum((x - m) ** 4 for x in data) / n
    if m2 == 0:
        return 0.0
    g2 = m4 / (m2 ** 2) - 3.0
    factor = (n - 1) / ((n - 2) * (n - 3))
    return factor * ((n + 1) * g2 + 6.0)


# ----------------------------------------------------
# 4. Quantiles, Ranking, and Percentiles
# ----------------------------------------------------

def quantile(data: List[float], q: float) -> float:
    """Compute quantile q in [0, 1] using linear interpolation."""
    if not data:
        raise ValueError("data cannot be empty")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1]")
    s = sorted(data)
    n = len(s)
    if n == 1:
        return float(s[0])
    idx = q * (n - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    frac = idx - low
    return s[low] + frac * (s[high] - s[low])


def percentile(data: List[float], p: float) -> float:
    """Compute percentile p in [0, 100]."""
    return quantile(data, p / 100.0)


def decile(data: List[float], d: int) -> float:
    """Compute d-th decile (d from 1 to 9)."""
    if not (1 <= d <= 9):
        raise ValueError("d must be between 1 and 9")
    return quantile(data, d / 10.0)


def quartile_1(data: List[float]) -> float:
    """First quartile Q1 (25th percentile)."""
    return quantile(data, 0.25)


def quartile_3(data: List[float]) -> float:
    """Third quartile Q3 (75th percentile)."""
    return quantile(data, 0.75)


def iqr_outlier_bounds(data: List[float], multiplier: float = 1.5) -> Tuple[float, float]:
    """Return Tukey fences (lower_bound, upper_bound) for outlier detection."""
    q1 = quartile_1(data)
    q3 = quartile_3(data)
    step = multiplier * (q3 - q1)
    return q1 - step, q3 + step


def percentile_rank(data: List[float], value: float) -> float:
    """Compute percentile rank of value within data [0, 100]."""
    if not data:
        raise ValueError("data cannot be empty")
    count_less = sum(1 for x in data if x < value)
    count_equal = sum(1 for x in data if x == value)
    return 100.0 * (count_less + 0.5 * count_equal) / len(data)


def rank_data(data: List[float]) -> List[float]:
    """Rank data handling ties with fractional averages (1-based)."""
    indexed = sorted(enumerate(data), key=lambda x: x[1])
    ranks = [0.0] * len(data)
    i = 0
    n = len(data)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def empirical_cdf(data: List[float], x: float) -> float:
    """Empirical cumulative distribution function F_n(x)."""
    if not data:
        raise ValueError("data cannot be empty")
    return sum(1 for v in data if v <= x) / len(data)


# ----------------------------------------------------
# 5. Correlation and Covariance
# ----------------------------------------------------

def population_covariance(x: List[float], y: List[float]) -> float:
    """Population covariance between x and y."""
    if len(x) != len(y) or not x:
        raise ValueError("x and y must have equal non-empty length")
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / len(x)


def sample_covariance(x: List[float], y: List[float]) -> float:
    """Sample covariance between x and y (ddof = 1)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length >= 2")
    mx, my = mean(x), mean(y)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (len(x) - 1)


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient r."""
    cov = sample_covariance(x, y)
    sx = sample_std_dev(x)
    sy = sample_std_dev(y)
    if sx == 0 or sy == 0:
        raise ValueError("Standard deviation of x or y is zero")
    return max(-1.0, min(1.0, cov / (sx * sy)))


def spearman_correlation(x: List[float], y: List[float]) -> float:
    """Compute Spearman's rank correlation coefficient rho."""
    rx = rank_data(x)
    ry = rank_data(y)
    return pearson_correlation(rx, ry)


def kendall_tau_correlation(x: List[float], y: List[float]) -> float:
    """Compute Kendall's rank correlation coefficient tau-b."""
    n = len(x)
    if n != len(y) or n < 2:
        raise ValueError("x and y must have equal length >= 2")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            else:
                if dx == 0:
                    ties_x += 1
                if dy == 0:
                    ties_y += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def r_squared(y_true: List[float], y_pred: List[float]) -> float:
    """Coefficient of determination R^2."""
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("lengths must match")
    ss_tot = sum((y - mean(y_true)) ** 2 for y in y_true)
    if ss_tot == 0:
        return 1.0
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1.0 - (ss_res / ss_tot)


def adjusted_r_squared(y_true: List[float], y_pred: List[float], num_predictors: int) -> float:
    """Adjusted R^2 accounting for number of predictors."""
    n = len(y_true)
    if n <= num_predictors + 1:
        raise ValueError("Sample size too small for degree of freedom adjustment")
    r2 = r_squared(y_true, y_pred)
    return 1.0 - ((1.0 - r2) * (n - 1) / (n - num_predictors - 1))


def autocorrelation_lag_k(x: List[float], lag: int) -> float:
    """Compute sample autocorrelation at lag k."""
    n = len(x)
    if lag >= n or lag < 0:
        raise ValueError("lag must be non-negative and < len(x)")
    if lag == 0:
        return 1.0
    m = mean(x)
    var = sum((v - m) ** 2 for v in x)
    if var == 0:
        return 0.0
    cov = sum((x[i] - m) * (x[i + lag] - m) for i in range(n - lag))
    return cov / var


# ----------------------------------------------------
# 6. Scaling, Normalization, and Transforms
# ----------------------------------------------------

def z_score(value: float, mean_val: float, std_val: float) -> float:
    """Compute standard score z = (x - mu) / sigma."""
    if std_val == 0:
        raise ZeroDivisionError("Standard deviation is zero")
    return (value - mean_val) / std_val


def z_score_normalize(data: List[float]) -> List[float]:
    """Standardize data to zero mean and unit variance."""
    m = mean(data)
    s = sample_std_dev(data)
    return [z_score(x, m, s) for x in data]


def min_max_normalize(data: List[float], feature_range: Tuple[float, float] = (0.0, 1.0)) -> List[float]:
    """Scale data linearly into target feature range."""
    min_x, max_x = min(data), max(data)
    diff = max_x - min_x
    if diff == 0:
        return [feature_range[0]] * len(data)
    a, b = feature_range
    return [a + ((x - min_x) / diff) * (b - a) for x in data]


def robust_scale_normalize(data: List[float]) -> List[float]:
    """Scale data using median and interquartile range."""
    med = median(data)
    iqr_val = iqr(data)
    if iqr_val == 0:
        return [0.0] * len(data)
    return [(x - med) / iqr_val for x in data]


def logit_transform(p: float) -> float:
    """Logit function log(p / (1 - p))."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be strictly in (0, 1)")
    return math.log(p / (1.0 - p))


def expit_transform(x: float) -> float:
    """Standard logistic sigmoid function 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + math.exp(-x))


# ----------------------------------------------------
# 7. Linear Regression and Error Metrics
# ----------------------------------------------------

def simple_linear_regression(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Fit ordinary least squares line y = slope * x + intercept."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must have equal length >= 2")
    mx, my = mean(x), mean(y)
    var_x = sum((xi - mx) ** 2 for xi in x)
    if var_x == 0:
        raise ValueError("Variance of x is zero; line is vertical")
    cov_xy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = cov_xy / var_x
    intercept = my - slope * mx
    return slope, intercept


def regression_residuals(x: List[float], y: List[float], slope: float, intercept: float) -> List[float]:
    """Compute residuals e_i = y_i - (slope * x_i + intercept)."""
    return [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]


def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Squared Error (MSE)."""
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    """Root Mean Squared Error (RMSE)."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Error (MAE)."""
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_percentage_error(y_true: List[float], y_pred: List[float]) -> float:
    """Mean Absolute Percentage Error (MAPE)."""
    return 100.0 * sum(abs((t - p) / t) for t, p in zip(y_true, y_pred) if t != 0) / len(y_true)


# ----------------------------------------------------
# 8. Hypothesis Test Statistics and Effect Sizes
# ----------------------------------------------------

def t_statistic_one_sample(data: List[float], mu0: float = 0.0) -> float:
    """Compute Student's t statistic for one sample mean against mu0."""
    se = standard_error_mean(data)
    if se == 0:
        raise ZeroDivisionError("Standard error is zero")
    return (mean(data) - mu0) / se


def t_statistic_two_sample_ind(data1: List[float], data2: List[float], equal_var: bool = True) -> float:
    """Compute independent two-sample Student's t-statistic (or Welch's t)."""
    n1, n2 = len(data1), len(data2)
    m1, m2 = mean(data1), mean(data2)
    s1_sq = sample_variance(data1)
    s2_sq = sample_variance(data2)
    if equal_var:
        sp_sq = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2)
        se = math.sqrt(sp_sq * (1.0 / n1 + 1.0 / n2))
    else:
        se = math.sqrt(s1_sq / n1 + s2_sq / n2)
    if se == 0:
        raise ZeroDivisionError("Standard error is zero")
    return (m1 - m2) / se


def t_statistic_paired(data1: List[float], data2: List[float]) -> float:
    """Compute paired samples Student's t-statistic."""
    if len(data1) != len(data2) or len(data1) < 2:
        raise ValueError("Paired data must have equal size >= 2")
    diffs = [x - y for x, y in zip(data1, data2)]
    return t_statistic_one_sample(diffs, 0.0)


def z_statistic(sample_mean_val: float, pop_mean_val: float, pop_std_val: float, n: int) -> float:
    """Compute Z statistic: (x_bar - mu) / (sigma / sqrt(n))."""
    if pop_std_val <= 0 or n <= 0:
        raise ValueError("pop_std_val and n must be positive")
    return (sample_mean_val - pop_mean_val) / (pop_std_val / math.sqrt(n))


def chi_square_goodness_of_fit(observed: List[float], expected: List[float]) -> float:
    """Compute Pearson's chi-square test statistic sum((O - E)^2 / E)."""
    if len(observed) != len(expected) or not observed:
        raise ValueError("observed and expected must have matching lengths")
    chi2 = 0.0
    for o, e in zip(observed, expected):
        if e <= 0:
            raise ValueError("expected frequencies must be positive")
        chi2 += ((o - e) ** 2) / e
    return chi2


def f_statistic_var_ratio(data1: List[float], data2: List[float]) -> float:
    """Compute F-test statistic for ratio of variances s1^2 / s2^2."""
    v2 = sample_variance(data2)
    if v2 == 0:
        raise ZeroDivisionError("variance of data2 is zero")
    return sample_variance(data1) / v2


def anova_one_way_f_stat(groups: List[List[float]]) -> float:
    """Compute F-statistic for one-way Analysis of Variance (ANOVA)."""
    k = len(groups)
    if k < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    all_data = [x for g in groups for x in g]
    grand_mean = mean(all_data)
    total_n = len(all_data)
    # Between-group sum of squares
    ss_between = sum(len(g) * ((mean(g) - grand_mean) ** 2) for g in groups)
    df_between = k - 1
    # Within-group sum of squares
    ss_within = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups)
    df_within = total_n - k
    if df_within <= 0 or ss_within == 0:
        raise ValueError("Insufficient within-group degrees of freedom")
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return ms_between / ms_within


def cohens_d(data1: List[float], data2: List[float]) -> float:
    """Compute Cohen's d effect size for two independent groups."""
    n1, n2 = len(data1), len(data2)
    s1_sq = sample_variance(data1)
    s2_sq = sample_variance(data2)
    sp = math.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2))
    if sp == 0:
        return 0.0
    return (mean(data1) - mean(data2)) / sp


def hedges_g(data1: List[float], data2: List[float]) -> float:
    """Compute Hedges' g (unbiased effect size estimate)."""
    d = cohens_d(data1, data2)
    df = len(data1) + len(data2) - 2
    j = 1.0 - (3.0 / (4.0 * df - 1.0))
    return d * j


def confidence_interval_mean_z(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Confidence interval for mean using normal approximation."""
    m = mean(data)
    se = standard_error_mean(data)
    # Approximate z values for common confidence levels
    z_map = {0.90: 1.64485, 0.95: 1.95996, 0.99: 2.57583}
    z = z_map.get(confidence, 1.95996)
    margin = z * se
    return m - margin, m + margin
def interquartile_mean(data: List[float]) -> float:
    """Compute mean of values in the interquartile range (between Q1 and Q3)."""
    q1 = quartile_1(data)
    q3 = quartile_3(data)
    mid = [x for x in data if q1 <= x <= q3]
    return sum(mid) / len(mid) if mid else mean(data)


def proportion_confidence_interval(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p_hat = successes / n
    z = 1.95996 if confidence == 0.95 else 1.64485
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def shannon_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p))."""
    h = 0.0
    for p in probabilities:
        if p < 0:
            raise ValueError("Probabilities must be non-negative")
        if p > 0:
            h -= p * math.log2(p)
    return h


def kl_divergence(p: List[float], q: List[float]) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q) = sum(p_i * ln(p_i / q_i))."""
    if len(p) != len(q):
        raise ValueError("Distributions must have equal length")
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total


def cross_entropy(p: List[float], q: List[float]) -> float:
    """Compute cross entropy H(P, Q) = -sum(p_i * ln(q_i))."""
    if len(p) != len(q):
        raise ValueError("Distributions must have equal length")
    return sum(-pi * math.log(qi) for pi, qi in zip(p, q) if pi > 0 and qi > 0)


def point_biserial_correlation(binary_x: List[int], continuous_y: List[float]) -> float:
    """Compute point-biserial correlation between binary 0/1 variable and continuous y."""
    y0 = [y for x, y in zip(binary_x, continuous_y) if x == 0]
    y1 = [y for x, y in zip(binary_x, continuous_y) if x == 1]
    if not y0 or not y1:
        raise ValueError("binary_x must contain both 0 and 1 values")
    m0, m1 = mean(y0), mean(y1)
    sy = sample_std_dev(continuous_y)
    n = len(continuous_y)
    p = len(y1) / n
    q = 1.0 - p
    return ((m1 - m0) / sy) * math.sqrt(p * q)


def durbin_watson_statistic(residuals: List[float]) -> float:
    """Compute Durbin-Watson statistic for autocorrelation in residuals."""
    if len(residuals) < 2:
        raise ValueError("Requires at least 2 residuals")
    diff_sq = sum((residuals[i] - residuals[i - 1]) ** 2 for i in range(1, len(residuals)))
    sum_sq = sum(r * r for r in residuals)
    if sum_sq == 0:
        return 0.0
    return diff_sq / sum_sq


def standard_error_estimate(y_true: List[float], y_pred: List[float]) -> float:
    """Standard error of regression estimate s_e = sqrt(sum((y - y_hat)^2) / (n - 2))."""
    n = len(y_true)
    if n <= 2:
        raise ValueError("Requires at least 3 points")
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    return math.sqrt(ss_res / (n - 2))


def polynomial_regression_fit(x: List[float], y: List[float], degree: int) -> List[float]:
    """Fit degree-d polynomial y ~ c_0 + c_1*x + ... + c_d*x^d returning coefficients [c_0, ..., c_d]."""
    m = degree + 1
    # Build Vandermonde-like normal equations X^T X c = X^T y
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for xi, yi in zip(x, y):
        powers = [xi ** k for k in range(2 * degree + 1)]
        for i in range(m):
            b[i] += yi * powers[i]
            for j in range(m):
                A[i][j] += powers[i + j]
    # Gaussian solve
    M = [A[i] + [b[i]] for i in range(m)]
    for i in range(m):
        max_r = max(range(i, m), key=lambda r: abs(M[r][i]))
        M[i], M[max_r] = M[max_r], M[i]
        pivot = M[i][i]
        for j in range(i + 1, m):
            factor = M[j][i] / pivot
            for k in range(i, m + 1):
                M[j][k] -= factor * M[i][k]
    coeffs = [0.0] * m
    for i in range(m - 1, -1, -1):
        s = sum(M[i][j] * coeffs[j] for j in range(i + 1, m))
        coeffs[i] = (M[i][m] - s) / M[i][i]
    return coeffs
