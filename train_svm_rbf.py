import argparse, sys, platform, time, traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("SVM RBF Kernel - Verbose Mode")
    print("=" * 60)
    print(f"\n[ENV] Python: {sys.version}")
    print(f"[ENV] Platform: {platform.platform()}")

    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.preprocessing import StandardScaler
        import sklearn
        print(f"[ENV] sklearn: {sklearn.__version__}")
    except ImportError as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    print("[DEVICE] sklearn SVM uses CPU only")

    print("\n[DATA] Loading Digits dataset...")
    try:
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"[DATA] Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {len(set(y))}")
    except Exception as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    print(f"[SPLIT] Train: {len(X_train)}, Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("[DATA] StandardScaler applied")

    print(f"\n[TRAIN] SVC(kernel='rbf', seed={args.seed})...")
    start = time.time()
    try:
        model = SVC(kernel="rbf", random_state=args.seed)
        model.fit(X_train, y_train)
        print(f"[TRAIN] Completed in {time.time()-start:.2f}s")
        print(f"[TRAIN] Support vectors: {model.n_support_.sum()}")
    except Exception as e:
        print(f"[ERROR] {e}"); traceback.print_exc(); sys.exit(1)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[EVAL] RBF SVM accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(f"[DONE] Completed (seed={args.seed})")

if __name__ == "__main__":
    main()
