import argparse, sys, platform, time, traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("SVM Kernel Comparison - Verbose Mode")
    print("=" * 60)
    print(f"\n[ENV] Python: {sys.version}, Platform: {platform.platform()}")

    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score
        from sklearn.preprocessing import StandardScaler
        import sklearn
        print(f"[ENV] sklearn: {sklearn.__version__}")
    except ImportError as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    print("[DEVICE] CPU only")

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=args.seed)
    print(f"[DATA] Digits: {digits.data.shape[0]} samples, Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kernels = ["rbf", "linear", "poly"]
    print(f"\n[COMPARE] Testing kernels: {kernels}")
    results = {}
    for kernel in kernels:
        print(f"\n[TRAIN] SVC(kernel='{kernel}', seed={args.seed})...")
        start = time.time()
        try:
            model = SVC(kernel=kernel, random_state=args.seed)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            elapsed = time.time() - start
            results[kernel] = acc
            print(f"[TRAIN] Completed in {elapsed:.2f}s, SVs: {model.n_support_.sum()}")
            print(f"[EVAL]  {kernel} accuracy: {acc:.4f}")
        except Exception as e:
            print(f"[ERROR] {kernel} failed: {e}")
            traceback.print_exc()
            results[kernel] = None

    print("\n[RESULT] Comparison Summary:")
    for k, v in sorted(results.items(), key=lambda x: -(x[1] or 0)):
        status = f"{v:.4f}" if v else "FAILED"
        print(f"[RESULT]   {k:8s}: {status}")

    print(f"\n[DONE] Completed (seed={args.seed})")

if __name__ == "__main__":
    main()
