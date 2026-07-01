"""Training script for Attention-GRU microsleep forecasting."""

import argparse
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .data_loader import load_data
from .features import create_temporal_feature_windows, extract_features
from .metrics import calculate_metrics, optimize_threshold
from .model import build_attention_gru
from .preprocessing import preprocess_subjects


def balance_training_data(x_train, y_train, balance_ratio=0.8):
    """Moderate oversampling of the minority class."""
    x_major = x_train[y_train == 0]
    x_minor = x_train[y_train == 1]
    y_major = y_train[y_train == 0]
    y_minor = y_train[y_train == 1]

    target_minor = int(len(x_major) * balance_ratio)
    if len(x_minor) < target_minor:
        x_minor, y_minor = resample(x_minor, y_minor, replace=True, n_samples=target_minor, random_state=42)

    x_bal = np.vstack([x_major, x_minor])
    y_bal = np.hstack([y_major, y_minor])
    idx = np.random.default_rng(42).permutation(len(y_bal))
    return x_bal[idx], y_bal[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to MWT .mat files")
    parser.add_argument("--forecast_seconds", type=float, default=0)
    args = parser.parse_args()

    eeg_01, _, labels_01, _ = load_data(args.data_dir, read_raw=False)

    processed_eegs, processed_labels = preprocess_subjects(
        eeg_01, labels_01, forecast_seconds=args.forecast_seconds
    )
    feature_subjects = extract_features(processed_eegs)

    x_all, y_all = [], []
    for feats, labs in zip(feature_subjects, processed_labels):
        x_temp, y_temp = create_temporal_feature_windows(feats, labs, window_size=5)
        if len(x_temp):
            x_all.append(x_temp)
            y_all.append(y_temp)

    if not x_all:
        raise RuntimeError("No training windows created. Check dataset path and label structure.")

    x = np.concatenate(x_all, axis=0)
    y = np.concatenate(y_all, axis=0).astype(int)
    print("Dataset shape:", x.shape, Counter(y))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    x_train, y_train = balance_training_data(x_train, y_train)
    print("Balanced training distribution:", Counter(y_train))

    scaler = RobustScaler()
    train_shape, test_shape = x_train.shape, x_test.shape
    x_train = scaler.fit_transform(x_train.reshape(-1, train_shape[-1])).reshape(train_shape)
    x_test = scaler.transform(x_test.reshape(-1, test_shape[-1])).reshape(test_shape)

    model = build_attention_gru(input_shape=(x_train.shape[1], x_train.shape[2]))
    model.summary()

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max"),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ],
    )

    y_proba = model.predict(x_test).flatten()
    threshold, _ = optimize_threshold(y_test, y_proba)
    metrics = calculate_metrics(y_test, y_proba, threshold)

    print("\nEvaluation metrics")
    print("Optimal threshold:", round(threshold, 3))
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"{key}: {value:.4f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
