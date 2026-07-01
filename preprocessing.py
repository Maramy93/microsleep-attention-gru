"""EEG preprocessing, filtering, and forecasting window generation."""

from typing import List, Tuple

import numpy as np
from scipy import signal as sg
from sklearn.utils import resample


def adaptive_filter_signal(eeg_signal: np.ndarray, fs: int = 200) -> np.ndarray:
    """Apply notch and bandpass filtering with fallbacks for short windows."""
    eeg_signal = np.asarray(eeg_signal).squeeze()
    if len(eeg_signal) < 50:
        return eeg_signal - np.mean(eeg_signal)

    try:
        quality = 30 if len(eeg_signal) >= 100 else 10
        b_notch, a_notch = sg.iirnotch(50, quality, fs)
        eeg_signal = sg.filtfilt(b_notch, a_notch, eeg_signal)

        order = 4 if len(eeg_signal) >= 200 else 2
        b_band, a_band = sg.butter(order, [0.5, 50], btype="band", fs=fs)
        eeg_signal = sg.filtfilt(b_band, a_band, eeg_signal)
    except ValueError:
        eeg_signal = eeg_signal - np.mean(eeg_signal)
        std = np.std(eeg_signal)
        if std > 0:
            eeg_signal = eeg_signal / std

    return eeg_signal


def create_windows_with_labels_forecast(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step_size: int,
    forecast_horizon_samples: int,
    fs: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create EEG windows and assign future labels for forecasting."""
    windows, window_labels = [], []
    data = np.asarray(data).squeeze()
    labels = np.asarray(labels).squeeze()
    start = 0

    while start + window_size + forecast_horizon_samples <= len(data):
        current_window = data[start:start + window_size]
        current_window = adaptive_filter_signal(current_window, fs=fs)
        future_index = start + window_size + forecast_horizon_samples - 1
        future_label = labels[future_index]
        windows.append(current_window)
        window_labels.append(future_label)
        start += step_size

    return np.asarray(windows), np.asarray(window_labels)


def preprocess_subjects(
    eegs: List[np.ndarray],
    labels: List[np.ndarray],
    forecast_seconds: float = 0,
    fs: int = 200,
    window_seconds: float = 1.5,
    overlap_fraction: float = 0.33,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Preprocess all subjects using binary labels and forecast horizons."""
    window_size = int(window_seconds * fs)
    step_size = max(1, int(window_size * overlap_fraction))
    forecast_horizon_samples = int(forecast_seconds * fs)

    processed_eegs, processed_labels = [], []

    for eeg, label in zip(eegs, labels):
        if len(eeg) < window_size + forecast_horizon_samples + 100:
            continue

        binary_labels = np.where(np.asarray(label).squeeze() > 0, 1, 0)
        win_data, win_labels = create_windows_with_labels_forecast(
            eeg, binary_labels, window_size, step_size, forecast_horizon_samples, fs
        )

        if len(win_data) == 0:
            continue

        idx_0 = np.where(win_labels == 0)[0]
        idx_1 = np.where(win_labels == 1)[0]
        if len(idx_0) == 0 or len(idx_1) == 0:
            continue

        min_size = min(len(idx_0), len(idx_1))
        max_samples = min(min_size * 2, max(len(idx_0), len(idx_1)))
        idx_0 = resample(idx_0, replace=False, n_samples=min(len(idx_0), max_samples), random_state=42)
        idx_1 = resample(idx_1, replace=False, n_samples=min(len(idx_1), max_samples), random_state=42)
        final_idx = np.concatenate([idx_0, idx_1])

        processed_eegs.append(win_data[final_idx])
        processed_labels.append(win_labels[final_idx])

    return processed_eegs, processed_labels
