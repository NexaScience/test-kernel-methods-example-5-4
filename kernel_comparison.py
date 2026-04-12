#!/usr/bin/env python3
"""Compare different SVM kernels on a chosen dataset using cross-validation.

Usage:
    python kernel_comparison.py --dataset digits --seed 42
"""
import argparse

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DATASETS = {
    "digits": load_digits,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}

KERNELS = ["rbf", "linear", "poly"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SVM kernels on a classification dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="digits",
        choices=list(DATASETS.keys()),
        help="Dataset to use (default: digits)",
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

    print(f"Loading dataset: {args.dataset}")
    data = DATASETS[args.dataset]()
    X, y = data.data, data.target

    # Scale features for fair comparison
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(np.unique(y))}")
    print(f"\nRunning 5-fold cross-validation for each kernel...\n")

    results = {}
    for kernel in KERNELS:
        model = SVC(kernel=kernel, random_state=args.seed)
        scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        results[kernel] = scores
        print(f"  {kernel:8s}  mean={scores.mean():.4f}  std={scores.std():.4f}")

    # Print comparison table
    print("\n" + "=" * 44)
    print(f"{'Kernel':<10} {'Mean Acc':>10} {'Std':>10} {'Min':>10}")
    print("-" * 44)
    for kernel in KERNELS:
        scores = results[kernel]
        print(f"{kernel:<10} {scores.mean():>10.4f} {scores.std():>10.4f} {scores.min():>10.4f}")
    print("=" * 44)

    best_kernel = max(results, key=lambda k: results[k].mean())
    print(f"\nBest kernel: {best_kernel} (mean accuracy: {results[best_kernel].mean():.4f})")


if __name__ == "__main__":
    main()
