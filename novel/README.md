# Novel PPML Protocol Research

Exploratory work on an optimized PPML protocol tailored for business-specific deployment scenarios, conducted alongside the SecureML and BLAZE evaluations.

## Contents

| File | Description |
|------|-------------|
| `novel.cpp` | C++ prototype for privately evaluating a **Double ReLU** activation using secret-shared fixed-point arithmetic |
| `novel.ipynb` | Jupyter notebook with step-by-step derivation and Beaver-triple multiplication walkthrough |

## Double ReLU Activation

The target function \( f(z) \) is a smoothed ReLU variant:

- \( f(z) = 0 \) if \( z < -\frac{1}{2} \)
- \( f(z) = z + \frac{1}{2} \) if \( -\frac{1}{2} \leq z \leq \frac{1}{2} \)
- \( f(z) = 1 \) if \( z > \frac{1}{2} \)

This activation was explored as a more numerically stable alternative for secure neural-network layers in fixed-point MPC settings.

## Fixed-Point Settings

`novel.cpp` uses a 16-bit scaling factor (`65536`), distinct from the 13-bit default in the SecureML implementation — reflecting precision trade-off experiments during protocol optimization.
