from dataclasses import dataclass
from gmpy2 import mpz

@dataclass
class angular_share:
	a: mpz = -1
	b: mpz = -1

@dataclass
class special_share:
	a: mpz = -1
	b: mpz = -1
	c: mpz = -1