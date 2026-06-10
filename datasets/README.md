# Datasets

Sample datasets used to evaluate PPML protocol efficiency and model accuracy.

## Included

### Medical Insurance (`medical/`)

Pre-split train/test CSV files derived from a health insurance cost dataset. Ready to use with SecureML — select option **2** at the dataset prompt.

| File | Description |
|------|-------------|
| `insurance_train.csv` | Training split |
| `insurance_test.csv` | Test split |
| `insurance.csv` | Full dataset |
| `insurance_int.csv` | Integer-encoded variant |

### MNIST (`mnist.zip`)

Full MNIST CSV archive. **Not extracted by default** (large file).

```bash
cd datasets
unzip mnist.zip
```

After extraction, use option **1** in the SecureML binaries.

### Binary MNIST

A reduced binary-classification MNIST subset. Generate it with the included notebook:

[`Binary MNIST generation.ipynb`](Binary%20MNIST%20generation.ipynb)

After generation, files are written to `datasets/binary_mnist/` (gitignored). Use option **3** in SecureML.

## Notes

- Large generated directories (`mnist/`, `binary_mnist/`) are listed in `.gitignore`.
- All dataset paths are relative to the **repository root** — run `bin/linear` and `bin/logistic` from the top-level directory.
