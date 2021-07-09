from dataclasses import dataclass
from gmpy2 import mpz
import numpy as np

@dataclass
class angular_share:
	a: mpz = -1
	b: mpz = -1

@dataclass
class special_share:
	a: np.uint64 = -1
	b: np.uint64 = -1
	c: np.uint64 = -1