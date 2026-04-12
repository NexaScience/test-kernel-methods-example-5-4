"""SVM with polynomial kernel on the Digits dataset."""
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser(description="SVM with polynomial kernel")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed
    )
    clf = SVC(kernel="poly", random_state=args.seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Poly SVM accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
