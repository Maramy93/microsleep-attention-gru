"""Feature extraction for EEG microsleep forecasting."""

import numpy as np
from scipy import signal as sg
from scipy.stats import skew, kurtosis


def nonlinear_features(x: np.ndarray) -> np.ndarray:
    """Return lightweight nonlinear features.

    These are stable approximations for repository use. Replace or extend with
    sample entropy, approximate entropy, or domain-specific nonlinear features
    when reproducing the full paper pipeline.
    """
    x = np.asarray(x).squeeze()
    dx = np.diff(x)
    ddx = np.diff(dx) if len(dx) > 1 else np.array([0])
    eps = 1e-10

    activity = np.var(x)
    mobility = np.sqrt(np.var(dx) / (activity + eps))
    complexity = np.sqrt(np.var(ddx) / (np.var(dx) + eps)) / (mobility + eps)
    line_length = np.sum(np.abs(dx))
    energy = np.sum(x ** 2) / max(len(x), 1)

    hist, _ = np.histogram(x, bins=20, density=True)
    hist = hist + eps
    entropy = -np.sum(hist * np.log(hist))

    return np.array([activity, mobility, complexity, line_length, entropy])


def frequency_features(x: np.ndarray, fs: int = 200) -> np.ndarray:
    """Extract absolute power, relative power, ratios, and spectral features."""
    x = np.asarray(x).squeeze()
    nperseg = min(256, max(16, len(x) // 4))
    freqs, psd = sg.welch(x, fs=fs, nperseg=nperseg, scaling="density", detrend="constant")

    bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 26),
        "gamma": (26, min(50, fs / 2 - 1)),
    }

    powers = []
    for low, high in bands.values():
        idx = (freqs >= low) & (freqs <= high)
        powers.append(np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0)

    delta, theta, alpha, beta, gamma = powers
    total = sum(powers) + 1e-10
    rel = [p / total for p in powers]

    theta_ratio = theta / (alpha + beta + 1e-10)
    alpha_theta_ratio = alpha / (delta + theta + 1e-10)

    cumsum = np.cumsum(psd)
    sef_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
    sef_95 = freqs[sef_idx[0]] if len(sef_idx) else freqs[-1]

    alpha_idx = (freqs >= 8) & (freqs <= 12)
    alpha_peak = freqs[alpha_idx][np.argmax(psd[alpha_idx])] if np.any(alpha_idx) else 10.0

    return np.array(powers + rel + [theta_ratio, alpha_theta_ratio, sef_95, alpha_peak, total])


def temporal_features(x: np.ndarray) -> np.ndarray:
    """Extract time-domain statistical and Hjorth features."""
    x = np.asarray(x).squeeze()
    mean_val = np.mean(x)
    std_val = np.std(x)
    var_val = np.var(x)
    skewness = skew(x) if len(x) > 3 and std_val > 0 else 0
    kurt = kurtosis(x) if len(x) > 3 and std_val > 0 else 0
    zcr = np.mean(np.diff(np.signbit(x - mean_val))) if len(x) > 1 else 0
    ptp = np.ptp(x)
    rms = np.sqrt(np.mean(x ** 2))

    dx = np.diff(x)
    ddx = np.diff(dx) if len(dx) > 1 else np.array([0])
    mobility = np.var(dx) / var_val if var_val > 0 else 0
    complexity = (np.var(ddx) / np.var(dx)) / mobility if np.var(dx) > 0 and mobility > 0 else 0

    return np.array([mean_val, std_val, var_val, skewness, kurt, zcr, ptp, rms, var_val, mobility, complexity])


def extract_features(subject_windows, fs: int = 200):
    """Extract 31 features per EEG window for every subject."""
    all_features = []
    for windows in subject_windows:
        feature_matrix = []
        for window in windows:
            feats = np.concatenate([
                nonlinear_features(window),
                frequency_features(window, fs),
                temporal_features(window),
            ])
            feature_matrix.append(feats)
        all_features.append(np.asarray(feature_matrix))
    return all_features


def create_temporal_feature_windows(features: np.ndarray, labels: np.ndarray, window_size: int = 5):
    """Convert sample-level features into short temporal feature sequences."""
    if len(features) < window_size:
        return np.empty((0, window_size, features.shape[1])), np.empty((0,))

    temporal_x, temporal_y = [], []
    for i in range(len(features) - window_size + 1):
        temporal_x.append(features[i:i + window_size])
        temporal_y.append(labels[i + window_size - 1])
    return np.asarray(temporal_x), np.asarray(temporal_y)
