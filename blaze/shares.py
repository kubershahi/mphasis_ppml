from dataclasses import dataclass
from gmpy2 import mpz
import numpy as np

@dataclass
class angular_share:
	a: mpz = -1
	b: mpz = -1

@dataclass
class special_share:
	def __init__(self, num):
		self.party_num = num
		self.a = np.uint64(0)
		self.b = np.uint64(0)
		self.c = np.uint64(0)