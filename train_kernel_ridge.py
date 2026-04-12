#!/usr/bin/env python3
"""Train Kernel Ridge Regression on California Housing.

Usage:
    python train_kernel_ridge.py --kernel rbf --alpha 1.0 --test-size 0.2 --seed 42
"""
import argparse

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Kernel Ridge Regression on California Housing."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="Kernel type (default: rbf)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default=None,
        help="Kernel coefficient: None (auto), or a float value (default: None)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse gamma
    gamma = None
    if args.gamma is not None:
        try:
            gamma = float(args.gamma)
        except ValueError:
            gamma = args.gamma

    print("Loading California Housing dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Use a subset of 2000 samples for speed
    n_samples = min(2000, X.shape[0])
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(X.shape[0], size=n_samples, replace=False)
    X, y = X[indices], y[indices]
    print(f"  Using {n_samples} samples, Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\nTraining KernelRidge(kernel={args.kernel}, alpha={args.alpha}, gamma={gamma})...")
    model = KernelRidge(kernel=args.kernel, alpha=args.alpha, gamma=gamma)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R2:   {r2:.4f}")

    model_path = "kernel_ridge_housing_model.joblib"
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
