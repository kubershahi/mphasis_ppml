# Experiments

Auxiliary prototypes, unit tests, and one-off experiments from the PPML research internship. These are **not** part of the main SecureML build.

## Contents

| File | Description |
|------|-------------|
| `trun.cpp` | Truncation and fixed-point linear regression experiments on medical data |
| `ot-test.cpp` | Oblivious-transfer integration test (requires emp-ot) |
| `test.cpp` | Eigen matrix and integer-ring sanity checks |
| `map.cpp`, `func.cpp`, `extras.cpp` | Low-level arithmetic and sharing helpers |

## Build OT Test

```bash
cd experiments
make        # builds ot-test → file
make clean
```

Requires `g++-11` (as configured in the local Makefile) and emp-ot headers.

## Run Truncation Experiment

```bash
# from repository root
g++ -std=c++14 -Isecureml/include -I/opt/homebrew/include/eigen3 -O2 \
  experiments/trun.cpp secureml/src/read_data.cpp -o bin/trun
./bin/trun
```
