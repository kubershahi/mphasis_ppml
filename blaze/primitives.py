#!/usr/bin/python
import sys
from config import config as conf
import pickle 
import random
import numpy as np
from gmpy2 import mpz
import math
import os
import hashlib

class primitives:

	def float2int(x): # embed float value onto the integer ring
		x = np.array(conf.converttoint64*(x), dtype = np.uint64)
		return x
	

	def int2float(x,scale=1<<conf.precision): # convert a value in the integer ring back to float
		y=0
		if(x > (2**(conf.l-1))-1):
			x = (2**conf.l) - x
			y = np.uint64(x)
			y = y*(-1)
		else:
			y = np.uint64(x)
			
		return float(y)/(scale)

	def byte_xor(ba1, ba2):
		return bytes([_a ^ _b for _a, _b in zip(ba1, ba2)])


	def randsample_2(send_partyNum): # allow any two of the three parties to sample a random value together
		conf.max_dec = primitives.int2float(2**(conf.l-1)-1)

		my_r = primitives.float2int(random.uniform(-1*conf.max_dec,conf.max_dec)) # sample a random float value from the possible range and embed it on the ring
		adv_r = primitives.send_recv_val(my_r,send_partyNum)

		r = (mpz(my_r)+mpz(adv_r))%(conf.modl)

		return r


	def Hash(x):

		m = hashlib.sha256(bytes(str(x).encode('utf-8'))).digest()
		# m = m.update(bytes(str(x).encode('utf-8')))
		# print("m: ",m)
		# m = m.digest()
		return m