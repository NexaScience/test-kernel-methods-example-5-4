"""Kernel Ridge Regression with RBF kernel on California Housing."""
import argparse
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="Kernel Ridge Regression")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = fetch_california_housing(return_X_y=True)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(X), size=2000, replace=False)
    X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = KernelRidge(kernel="rbf")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Kernel Ridge (RBF) RMSE: {rmse:.4f}  R2: {r2:.4f}")


if __name__ == "__main__":
    main()
