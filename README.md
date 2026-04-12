# Kernel Methods Example

Minimal scikit-learn examples demonstrating kernel-based methods.

## Scripts

| Script | Description |
|---|---|
| `train_svm_rbf.py` | SVM with RBF kernel on Digits |
| `train_svm_linear.py` | SVM with linear kernel on Digits |
| `train_svm_poly.py` | SVM with polynomial kernel on Digits |
| `kernel_comparison.py` | Compare all three SVM kernels |
| `train_kernel_ridge.py` | Kernel Ridge Regression on California Housing (2000 samples) |

## Usage

```bash
pip install -r requirements.txt
python train_svm_rbf.py --seed 42
python kernel_comparison.py
python train_kernel_ridge.py
```

Each script accepts only `--seed` (default 42) for reproducibility.
