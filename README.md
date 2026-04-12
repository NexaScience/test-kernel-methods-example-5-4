# Kernel Methods Example

A minimal collection of scripts demonstrating classic kernel methods (SVM, Kernel Ridge Regression) using scikit-learn.

## Setup

```bash
pip install -r requirements.txt
```

## Scripts

### train_svm.py

Train an SVM classifier on the Digits dataset.

```bash
python train_svm.py --kernel rbf --C 1.0 --gamma scale --test-size 0.2 --seed 42
python train_svm.py --kernel linear
python train_svm.py --kernel poly --C 10.0
```

### train_kernel_ridge.py

Train Kernel Ridge Regression on a subset of California Housing.

```bash
python train_kernel_ridge.py --kernel rbf --alpha 1.0 --test-size 0.2 --seed 42
python train_kernel_ridge.py --kernel linear --alpha 0.1
python train_kernel_ridge.py --kernel poly --alpha 0.5
```

### kernel_comparison.py

Compare SVM kernels (rbf, linear, poly) via 5-fold cross-validation.

```bash
python kernel_comparison.py --dataset digits --seed 42
python kernel_comparison.py --dataset wine
python kernel_comparison.py --dataset breast_cancer
```
