# BLAZE Protocol (Python)

Python prototype of the **BLAZE** three-party PPML framework from [Kumar et al. (2020)](https://eprint.iacr.org/2020/042.pdf).

## Layout

| File | Role |
|------|------|
| `shares.py` | Angular and special three-party share types |
| `primitives.py` | Integer-ring embedding, additive sharing, reconstruction |
| `protocols.py` | Secure multiplication, truncation, and protocol composition |
| `config.py` | Ring parameters (64-bit modulus, 13-bit precision) |
| `main.py` | Unit tests for primitives and sharing semantics |
| `main2.py` | Extended protocol integration tests |

## Run

```bash
pip install -r requirements.txt
python3 main.py
python3 main2.py
```

## Notes

This is a local simulation of three-party computation — parties are represented as Python objects rather than networked processes. Networking stubs exist in `protocols.py` for future distributed deployment.
