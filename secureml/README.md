# SecureML Implementation

C++ implementation of the **SecureML** two-party protocol for privacy-preserving linear and logistic regression, based on [Mohassel & Zhang (IEEE S&P 2017)](https://ieeexplore.ieee.org/document/7958569).

## Layout

```
secureml/
├── include/   # Headers (defines, utils, regression, data loading)
└── src/       # Implementation and entry-point drivers
```

## Build

Built from the repository root:

```bash
make linear    # → bin/linear
make logistic  # → bin/logistic
```

If the compiler cannot find Eigen:

```bash
make EIGEN_INCLUDE=/opt/homebrew/include/eigen3 linear
```

## Entry Points

| Binary | Source | Model |
|--------|--------|-------|
| `bin/linear` | `src/linear.cpp` | Secure linear regression |
| `bin/logistic` | `src/logistic.cpp` | Secure logistic regression |

## Core Modules

- **`utils`** — Secret sharing, Beaver-triple multiplication, fixed-point conversion
- **`linear_regression` / `logistic_regression`** — Ideal (plaintext) and secure training loops
- **`read_data`** — CSV loaders for MNIST, binary MNIST, and medical insurance data

## Fixed-Point Precision

Default scaling factor is `8192` (13-bit precision), defined in `include/defines.hpp`. Truncation and ring arithmetic are applied after each secure multiplication to prevent overflow.
