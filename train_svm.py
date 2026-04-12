#!/usr/bin/env python3
"""Train an SVM classifier on the Digits dataset (sklearn).

Usage:
    python train_svm.py --kernel rbf --C 1.0 --gamma scale --test-size 0.2 --seed 42
"""
import argparse

import joblib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an SVM classifier on the Digits dataset."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["rbf", "linear", "poly"],
        help="SVM kernel type (default: rbf)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization parameter (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient: 'scale', 'auto', or a float value (default: scale)",
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

    # Parse gamma: allow float values as well as string presets
    try:
        gamma = float(args.gamma)
    except ValueError:
        gamma = args.gamma

    print(f"Loading Digits dataset...")
    digits = load_digits()
    X, y = digits.data, digits.target
    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    print(f"\nTraining SVC(kernel={args.kernel}, C={args.C}, gamma={gamma})...")
    model = SVC(kernel=args.kernel, C=args.C, gamma=gamma, random_state=args.seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_path = "svm_digits_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
