import argparse, sys, platform, time, traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("Kernel Ridge Regression - Verbose Mode")
    print("=" * 60)
    print(f"\n[ENV] Python: {sys.version}, Platform: {platform.platform()}")

    try:
        import numpy as np
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.preprocessing import StandardScaler
        import sklearn
        print(f"[ENV] sklearn: {sklearn.__version__}, numpy: {np.__version__}")
    except ImportError as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    print("[DEVICE] CPU only")

    print("\n[DATA] Loading California Housing (subsample 2000)...")
    np.random.seed(args.seed)
    data = fetch_california_housing()
    idx = np.random.choice(len(data.data), 2000, replace=False)
    X, y = data.data[idx], data.target[idx]
    print(f"[DATA] Samples: {X.shape[0]}, Features: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    print(f"\n[TRAIN] KernelRidge(kernel='rbf')...")
    start = time.time()
    try:
        model = KernelRidge(kernel="rbf")
        model.fit(X_train, y_train)
        print(f"[TRAIN] Completed in {time.time()-start:.2f}s")
    except Exception as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    y_pred = model.predict(X_test)
    try:
        from sklearn.metrics import root_mean_squared_error
        rmse = root_mean_squared_error(y_test, y_pred)
    except ImportError:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"\n[EVAL] Kernel Ridge (RBF) RMSE: {rmse:.4f}  R2: {r2:.4f}")
    print(f"[DONE] Completed (seed={args.seed})")

if __name__ == "__main__":
    main()
