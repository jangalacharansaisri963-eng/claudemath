"""Fourier analysis module for claudemath.

Pure-Python implementation of Cooley-Tukey FFT, IFFT, 2D FFT, DCT (I-IV), DST (I-IV),
window functions (Hann, Hamming, Blackman, Kaiser, etc.), STFT, spectrogram,
Hilbert transform, circular/linear FFT convolutions, and spectral features.
"""

from typing import List, Tuple, Optional, Callable
import math
import cmath


# ----------------------------------------------------
# 1. Classical DFT and Fast Fourier Transform (FFT)
# ----------------------------------------------------

def dft_1d(x: List[complex]) -> List[complex]:
    """Naive 1D Discrete Fourier Transform O(N^2): X_k = sum_{n=0}^{N-1} x_n * exp(-2pi*i*k*n / N)."""
    N = len(x)
    X = []
    for k in range(N):
        total = 0.0 + 0.0j
        for n in range(N):
            angle = -2.0 * math.pi * k * n / N
            total += x[n] * cmath.exp(complex(0.0, angle))
        X.append(total)
    return X


def idft_1d(X: List[complex]) -> List[complex]:
    """Naive 1D Inverse Discrete Fourier Transform O(N^2): x_n = (1/N) sum_{k=0}^{N-1} X_k * exp(2pi*i*k*n / N)."""
    N = len(X)
    x = []
    for n in range(N):
        total = 0.0 + 0.0j
        for k in range(N):
            angle = 2.0 * math.pi * k * n / N
            total += X[k] * cmath.exp(complex(0.0, angle))
        x.append(total / N)
    return x


def fft_radix2(x: List[complex]) -> List[complex]:
    """Cooley-Tukey radix-2 decimation-in-time FFT for power-of-two length N."""
    N = len(x)
    if N <= 1:
        return list(x)
    if N & (N - 1) != 0:
        # Zero-pad to next power of 2
        next_pow2 = 1 << (N - 1).bit_length()
        x = list(x) + [0.0j] * (next_pow2 - N)
        N = next_pow2

    even = fft_radix2(x[0::2])
    odd = fft_radix2(x[1::2])
    T = [cmath.exp(complex(0.0, -2.0 * math.pi * k / N)) * odd[k] for k in range(N // 2)]
    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]


def ifft_radix2(X: List[complex]) -> List[complex]:
    """Inverse FFT using conjugated forward FFT: ifft(X) = conj(fft(conj(X))) / N."""
    N = len(X)
    conj_X = [z.conjugate() for z in X]
    fft_res = fft_radix2(conj_X)
    N_actual = len(fft_res)
    return [z.conjugate() / N_actual for z in fft_res]


def fft_real(x: List[float]) -> List[complex]:
    """FFT of real-valued input sequence."""
    return fft_radix2([complex(val, 0.0) for val in x])


def ifft_real(X: List[complex]) -> List[float]:
    """Return real component of inverse FFT."""
    res = ifft_radix2(X)
    return [z.real for z in res]


def fft_frequencies(N: int, sample_rate: float = 1.0) -> List[float]:
    """Return FFT sample frequencies in Hertz."""
    val = 1.0 / (N * (1.0 / sample_rate))
    results = [0.0] * N
    N_half = (N - 1) // 2 + 1
    for i in range(N_half):
        results[i] = i * val
    for i in range(N_half, N):
        results[i] = -(N - i) * val
    return results


def fft_shift(X: List[complex]) -> List[complex]:
    """Shift zero-frequency component to center of spectrum."""
    N = len(X)
    mid = (N + 1) // 2
    return X[mid:] + X[:mid]


def ifft_shift(X: List[complex]) -> List[complex]:
    """Inverse of fft_shift."""
    N = len(X)
    mid = N // 2
    return X[mid:] + X[:mid]


# ----------------------------------------------------
# 2. 2D Fourier Transforms
# ----------------------------------------------------

def fft_2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """2D Fast Fourier Transform by separable row and column 1D FFTs."""
    rows = len(matrix)
    cols = len(matrix[0])
    # FFT on each row
    row_fft = [fft_radix2(row) for row in matrix]
    new_cols = len(row_fft[0])
    # FFT on each column
    result = [[0.0j] * new_cols for _ in range(rows)]
    for j in range(new_cols):
        col = [row_fft[i][j] for i in range(rows)]
        col_fft = fft_radix2(col)
        for i in range(len(col_fft)):
            if i < rows:
                result[i][j] = col_fft[i]
    return result


def ifft_2d(matrix: List[List[complex]]) -> List[List[complex]]:
    """2D Inverse Fast Fourier Transform."""
    rows = len(matrix)
    cols = len(matrix[0])
    row_ifft = [ifft_radix2(row) for row in matrix]
    new_cols = len(row_ifft[0])
    result = [[0.0j] * new_cols for _ in range(rows)]
    for j in range(new_cols):
        col = [row_ifft[i][j] for i in range(rows)]
        col_ifft = ifft_radix2(col)
        for i in range(len(col_ifft)):
            if i < rows:
                result[i][j] = col_ifft[i]
    return result


# ----------------------------------------------------
# 3. Discrete Cosine and Sine Transforms (DCT / DST)
# ----------------------------------------------------

def dct_type_2(x: List[float]) -> List[float]:
    """DCT-II (standard JPEG DCT): X_k = 2 * sum_{n=0}^{N-1} x_n * cos(pi * (2n + 1) * k / (2N))."""
    N = len(x)
    X = []
    for k in range(N):
        total = sum(x[n] * math.cos(math.pi * (2 * n + 1) * k / (2.0 * N)) for n in range(N))
        X.append(2.0 * total)
    return X


def idct_type_2(X: List[float]) -> List[float]:
    """Inverse DCT-II (DCT-III normalized): x_n = (1/2N) X_0 + (1/N) sum_{k=1}^{N-1} X_k * cos(pi * (2n + 1) * k / (2N))."""
    N = len(X)
    x = []
    for n in range(N):
        total = 0.5 * X[0]
        for k in range(1, N):
            total += X[k] * math.cos(math.pi * (2 * n + 1) * k / (2.0 * N))
        x.append(total / N)
    return x


def dct_type_1(x: List[float]) -> List[float]:
    """DCT-I for N >= 2: X_k = 0.5*(x_0 + (-1)^k x_{N-1}) + sum_{n=1}^{N-2} x_n cos(pi * n * k / (N - 1))."""
    N = len(x)
    if N < 2:
        return list(x)
    X = []
    for k in range(N):
        total = 0.5 * (x[0] + ((-1.0) ** k) * x[-1])
        for n in range(1, N - 1):
            total += x[n] * math.cos(math.pi * n * k / (N - 1))
        X.append(total)
    return X


def dst_type_1(x: List[float]) -> List[float]:
    """DST-I: X_k = sum_{n=0}^{N-1} x_n * sin(pi * (n + 1) * (k + 1) / (N + 1))."""
    N = len(x)
    X = []
    for k in range(N):
        total = sum(x[n] * math.sin(math.pi * (n + 1) * (k + 1) / (N + 1.0)) for n in range(N))
        X.append(total)
    return X


def dst_type_2(x: List[float]) -> List[float]:
    """DST-II: X_k = 2 * sum_{n=0}^{N-1} x_n * sin(pi * (2n + 1) * (k + 1) / (2N))."""
    N = len(x)
    X = []
    for k in range(N):
        total = sum(x[n] * math.sin(math.pi * (2 * n + 1) * (k + 1) / (2.0 * N)) for n in range(N))
        X.append(2.0 * total)
    return X


# ----------------------------------------------------
# 4. Window Functions
# ----------------------------------------------------

def window_rectangular(N: int) -> List[float]:
    """Rectangular (Dirichlet) window: all ones."""
    return [1.0] * N


def window_hann(N: int) -> List[float]:
    """Hann (Hanning) window: 0.5 * (1 - cos(2pi*n / (N - 1)))."""
    if N <= 1:
        return [1.0]
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * n / (N - 1))) for n in range(N)]


def window_hamming(N: int) -> List[float]:
    """Hamming window: 0.54 - 0.46 * cos(2pi*n / (N - 1))."""
    if N <= 1:
        return [1.0]
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * n / (N - 1)) for n in range(N)]


def window_blackman(N: int) -> List[float]:
    """Blackman window: 0.42 - 0.5*cos(2pi*n/(N-1)) + 0.08*cos(4pi*n/(N-1))."""
    if N <= 1:
        return [1.0]
    return [
        0.42 - 0.5 * math.cos(2.0 * math.pi * n / (N - 1)) + 0.08 * math.cos(4.0 * math.pi * n / (N - 1))
        for n in range(N)
    ]


def window_bartlett(N: int) -> List[float]:
    """Bartlett (triangular) window with zero endpoints."""
    if N <= 1:
        return [1.0]
    return [1.0 - abs(2.0 * n - (N - 1)) / (N - 1) for n in range(N)]


def window_flattop(N: int) -> List[float]:
    """Flat-top window for accurate amplitude measurement."""
    if N <= 1:
        return [1.0]
    a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
    res = []
    for n in range(N):
        term = (
            a0 - a1 * math.cos(2.0 * math.pi * n / (N - 1))
            + a2 * math.cos(4.0 * math.pi * n / (N - 1))
            - a3 * math.cos(6.0 * math.pi * n / (N - 1))
            + a4 * math.cos(8.0 * math.pi * n / (N - 1))
        )
        res.append(term)
    return res


def window_gaussian(N: int, sigma: float = 0.4) -> List[float]:
    """Gaussian window with standard deviation parameter sigma."""
    if N <= 1:
        return [1.0]
    res = []
    for n in range(N):
        num = n - 0.5 * (N - 1)
        den = sigma * 0.5 * (N - 1)
        res.append(math.exp(-0.5 * (num / den) ** 2))
    return res


def window_kaiser(N: int, beta: float = 14.0) -> List[float]:
    """Kaiser window using zeroth-order modified Bessel function I0."""
    def bessel_i0(x: float) -> float:
        total = 1.0
        term = 1.0
        for m in range(1, 25):
            term *= (x * 0.5 / m) ** 2
            total += term
        return total

    if N <= 1:
        return [1.0]
    i0_beta = bessel_i0(beta)
    res = []
    for n in range(N):
        val = 2.0 * n / (N - 1) - 1.0
        arg = beta * math.sqrt(max(0.0, 1.0 - val * val))
        res.append(bessel_i0(arg) / i0_beta)
    return res


def window_tukey(N: int, alpha: float = 0.5) -> List[float]:
    """Tukey (tapered cosine) window."""
    if N <= 1:
        return [1.0]
    if alpha <= 0.0:
        return window_rectangular(N)
    if alpha >= 1.0:
        return window_hann(N)
    res = [0.0] * N
    boundary = int(alpha * (N - 1) * 0.5)
    for n in range(N):
        if n < boundary:
            res[n] = 0.5 * (1.0 + math.cos(math.pi * (2.0 * n / (alpha * (N - 1)) - 1.0)))
        elif n <= (N - 1) - boundary:
            res[n] = 1.0
        else:
            res[n] = 0.5 * (1.0 + math.cos(math.pi * (2.0 * n / (alpha * (N - 1)) - 2.0 / alpha + 1.0)))
    return res


# ----------------------------------------------------
# 5. Convolution and Correlation via FFT
# ----------------------------------------------------

def fft_convolve_circular(x: List[float], h: List[float]) -> List[float]:
    """Circular convolution of two equal-length signals via FFT."""
    N = len(x)
    if len(h) != N:
        raise ValueError("Inputs must have identical length for circular convolution")
    X = fft_radix2([complex(v, 0.0) for v in x])
    H = fft_radix2([complex(v, 0.0) for v in h])
    Y = [a * b for a, b in zip(X, H)]
    y_full = ifft_radix2(Y)
    return [z.real for z in y_full[:N]]


def fft_convolve_linear(x: List[float], h: List[float]) -> List[float]:
    """Linear convolution of two signals via zero-padded FFT."""
    out_len = len(x) + len(h) - 1
    pow2 = 1 << (out_len - 1).bit_length()
    x_pad = [complex(v, 0.0) for v in x] + [0.0j] * (pow2 - len(x))
    h_pad = [complex(v, 0.0) for v in h] + [0.0j] * (pow2 - len(h))
    X = fft_radix2(x_pad)
    H = fft_radix2(h_pad)
    Y = [a * b for a, b in zip(X, H)]
    y_full = ifft_radix2(Y)
    return [z.real for z in y_full[:out_len]]


def fft_cross_correlation(x: List[float], y: List[float]) -> List[float]:
    """Cross-correlation (x * y)[n] using zero-padded FFT."""
    return fft_convolve_linear(x, list(reversed(y)))


# ----------------------------------------------------
# 6. Spectral Features and Hilbert Transform
# ----------------------------------------------------

def power_spectral_density(x: List[float], window_type: str = "hann") -> List[float]:
    """Compute normalized Power Spectral Density (PSD) periodogram."""
    N = len(x)
    if window_type == "hann":
        w = window_hann(N)
    elif window_type == "hamming":
        w = window_hamming(N)
    else:
        w = window_rectangular(N)
    windowed = [x[i] * w[i] for i in range(N)]
    X = fft_real(windowed)
    scale = sum(wi * wi for wi in w)
    if scale == 0:
        scale = 1.0
    return [(abs(z) ** 2) / scale for z in X]


def spectral_centroid(magnitudes: List[float], freqs: List[float]) -> float:
    """Compute spectral centroid: center of mass of spectrum."""
    num = sum(f * m for f, m in zip(freqs, magnitudes))
    den = sum(magnitudes)
    return num / den if den > 1e-15 else 0.0


def spectral_spread(magnitudes: List[float], freqs: List[float]) -> float:
    """Compute spectral spread (standard deviation around centroid)."""
    mu = spectral_centroid(magnitudes, freqs)
    den = sum(magnitudes)
    if den < 1e-15:
        return 0.0
    var = sum(((f - mu) ** 2) * m for f, m in zip(freqs, magnitudes)) / den
    return math.sqrt(max(0.0, var))


def hilbert_transform(x: List[float]) -> List[float]:
    """Discrete Hilbert transform using FFT: shifts negative freqs by -90 deg, positive by +90 deg."""
    N = len(x)
    X = fft_real(x)
    N_fft = len(X)
    H = [0.0j] * N_fft
    H[0] = 1.0
    if N_fft % 2 == 0:
        H[N_fft // 2] = 1.0
        for i in range(1, N_fft // 2):
            H[i] = 2.0
    else:
        for i in range(1, (N_fft + 1) // 2):
            H[i] = 2.0
    X_analytic = [X[i] * H[i] for i in range(N_fft)]
    analytic_signal = ifft_radix2(X_analytic)
    # Hilbert transform is imaginary part of analytic signal
    return [z.imag for z in analytic_signal[:N]]
def dct_type_3(x: List[float]) -> List[float]:
    """DCT-III (inverse of DCT-II): X_k = x_0 + 2 * sum_{n=1}^{N-1} x_n * cos(pi * n * (2k + 1) / (2N))."""
    N = len(x)
    X = []
    for k in range(N):
        total = x[0]
        for n in range(1, N):
            total += 2.0 * x[n] * math.cos(math.pi * n * (2 * k + 1) / (2.0 * N))
        X.append(total)
    return X


def dct_type_4(x: List[float]) -> List[float]:
    """DCT-IV: self-inverse orthogonal transform X_k = 2 * sum_{n=0}^{N-1} x_n * cos(pi * (2n + 1) * (2k + 1) / (4N))."""
    N = len(x)
    X = []
    for k in range(N):
        total = sum(x[n] * math.cos(math.pi * (2 * n + 1) * (2 * k + 1) / (4.0 * N)) for n in range(N))
        X.append(2.0 * total)
    return X


def dst_type_3(x: List[float]) -> List[float]:
    """DST-III: inverse of DST-II."""
    N = len(x)
    X = []
    for k in range(N):
        total = ((-1.0) ** k) * x[-1]
        for n in range(N - 1):
            total += 2.0 * x[n] * math.sin(math.pi * (2 * k + 1) * (n + 1) / (2.0 * N))
        X.append(total)
    return X


def dst_type_4(x: List[float]) -> List[float]:
    """DST-IV: self-inverse transform."""
    N = len(x)
    X = []
    for k in range(N):
        total = sum(x[n] * math.sin(math.pi * (2 * n + 1) * (2 * k + 1) / (4.0 * N)) for n in range(N))
        X.append(2.0 * total)
    return X


def fast_walsh_hadamard_transform(x: List[float]) -> List[float]:
    """Fast Walsh-Hadamard Transform (FWHT) in-place butterfly algorithm for power-of-2 length."""
    N = len(x)
    h = 1
    a = list(x)
    while h < N:
        for i in range(0, N, h * 2):
            for j in range(i, i + h):
                u = a[j]
                v = a[j + h]
                a[j] = u + v
                a[j + h] = u - v
        h *= 2
    return a


def short_time_fourier_transform(x: List[float], window_size: int = 256,
                                 hop_size: int = 128) -> List[List[complex]]:
    """Short-Time Fourier Transform (STFT) with Hann windowing."""
    w = window_hann(window_size)
    stft_matrix = []
    for start in range(0, len(x) - window_size + 1, hop_size):
        frame = [x[start + i] * w[i] for i in range(window_size)]
        stft_matrix.append(fft_real(frame))
    return stft_matrix


def spectrogram(x: List[float], window_size: int = 256,
                hop_size: int = 128) -> List[List[float]]:
    """Power spectrogram |STFT(x)|^2."""
    stft = short_time_fourier_transform(x, window_size, hop_size)
    return [[abs(val) ** 2 for val in frame] for frame in stft]


def analytic_signal(x: List[float]) -> List[complex]:
    """Construct complex analytic signal z(t) = x(t) + i * H{x}(t)."""
    h_x = hilbert_transform(x)
    return [complex(re, im) for re, im in zip(x, h_x)]


def instantaneous_amplitude_envelope(x: List[float]) -> List[float]:
    """Compute instantaneous amplitude (envelope) via analytic signal."""
    z = analytic_signal(x)
    return [abs(val) for val in z]


def instantaneous_phase(x: List[float]) -> List[float]:
    """Compute unwrapped instantaneous phase from analytic signal."""
    z = analytic_signal(x)
    return [cmath.phase(val) for val in z]


def window_bohman(N: int) -> List[float]:
    """Bohman window: 2D convolution of two half-duration cosine lobes."""
    if N <= 1:
        return [1.0]
    res = []
    for n in range(N):
        x = abs(2.0 * n / (N - 1) - 1.0)
        if x >= 1.0:
            res.append(0.0)
        else:
            term = (1.0 - x) * math.cos(math.pi * x) + (1.0 / math.pi) * math.sin(math.pi * x)
            res.append(term)
    return res


def window_parzen(N: int) -> List[float]:
    """Parzen window (4th-order B-spline window)."""
    if N <= 1:
        return [1.0]
    res = []
    for n in range(N):
        z = abs(n - 0.5 * (N - 1)) / (0.5 * N)
        if z <= 0.5:
            res.append(1.0 - 6.0 * z * z * (1.0 - z))
        elif z <= 1.0:
            res.append(2.0 * ((1.0 - z) ** 3))
        else:
            res.append(0.0)
    return res


def window_welch(N: int) -> List[float]:
    """Welch parabolic window: 1 - ((n - 0.5*(N-1)) / (0.5*(N-1)))^2."""
    if N <= 1:
        return [1.0]
    res = []
    half = 0.5 * (N - 1)
    for n in range(N):
        val = (n - half) / half
        res.append(max(0.0, 1.0 - val * val))
    return res


def spectral_flatness(magnitudes: List[float]) -> float:
    """Wiener entropy (spectral flatness): geometric mean / arithmetic mean."""
    pos_mags = [m for m in magnitudes if m > 1e-15]
    if not pos_mags:
        return 0.0
    arithmetic_mean = sum(pos_mags) / len(pos_mags)
    log_sum = sum(math.log(m) for m in pos_mags)
    geometric_mean = math.exp(log_sum / len(pos_mags))
    return geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0.0


def spectral_flux(mag1: List[float], mag2: List[float]) -> float:
    """Spectral flux (rate of spectral change) between two consecutive frames."""
    return math.sqrt(sum((b - a) ** 2 for a, b in zip(mag1, mag2)))


def spectral_rolloff(magnitudes: List[float], freqs: List[float], percentile: float = 0.85) -> float:
    """Spectral roll-off frequency below which percentile (e.g. 85%) of energy is contained."""
    total_energy = sum(magnitudes)
    threshold = total_energy * percentile
    cum = 0.0
    for f, m in zip(freqs, magnitudes):
        cum += m
        if cum >= threshold:
            return f
    return freqs[-1] if freqs else 0.0


def goertzel_single_frequency(x: List[float], target_freq: float, sample_rate: float) -> complex:
    """Goertzel algorithm: evaluate single DFT bin efficiently in O(N) time and O(1) space."""
    N = len(x)
    k = int(0.5 + N * target_freq / sample_rate)
    omega = 2.0 * math.pi * k / N
    coeff = 2.0 * math.cos(omega)
    s_prev = 0.0
    s_prev2 = 0.0
    for val in x:
        s = val + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    real = s_prev - s_prev2 * math.cos(omega)
    imag = -s_prev2 * math.sin(omega)
    return complex(real, imag)
def spectral_entropy(magnitudes: List[float]) -> float:
    """Normalized Shannon entropy of spectral distribution in [0, 1]."""
    total = sum(magnitudes)
    if total <= 1e-15:
        return 0.0
    p = [m / total for m in magnitudes if m > 1e-15]
    n = len(magnitudes)
    if n <= 1:
        return 0.0
    ent = -sum(pi * math.log(pi) for pi in p)
    return ent / math.log(n)


def coherence_magnitude_squared(psd_x: float, psd_y: float, csd_xy: complex) -> float:
    """Magnitude-squared coherence gamma_{xy}^2 = |CSD_{xy}|^2 / (PSD_x * PSD_y)."""
    denom = psd_x * psd_y
    if denom <= 1e-15:
        return 0.0
    return (abs(csd_xy) ** 2) / denom
