#!/usr/bin/python
import sys
from config import config as conf
from shares import special_share as spc
import pickle 
import random
import numpy as np
from gmpy2 import mpz
import math
import os
import hashlib

class primitives:

	def float2int(x): # embed float value onto the integer ring
		#x = np.array(conf.converttoint64*(x), dtype = np.uint64)
		x = np.uint64(conf.converttoint64*(x))
		return x
	

	def int2float(x,scale=1<<conf.precision): # convert a value in the integer ring back to float
		y = 0
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

	def print_special(p0, p1, p2):
		print(f"{p0.a} --> {primitives.int2float(p0.a)}")
		print(f"{p0.b} --> {primitives.int2float(p0.b)}")
		print(f"{p0.c} --> {primitives.int2float(p0.c)}")
		print(f"{p1.a} --> {primitives.int2float(p1.a)}")
		print(f"{p1.b} --> {primitives.int2float(p1.b)}")
		print(f"{p1.c} --> {primitives.int2float(p1.c)}")
		print(f"{p2.a} --> {primitives.int2float(p2.a)}")
		print(f"{p2.b} --> {primitives.int2float(p2.b)}")
		print(f"{p2.c} --> {primitives.int2float(p2.c)}")

	def additive_sharing(a):
		a1 = primitives.float2int(random.randint(1, 10))
		a2 = a - a1
		return (a1,a2)

	def special_sharing(a):
		p0 = spc(0)
		p1 = spc(1)
		p2 = spc(2)
		alpha_p = primitives.float2int(random.randint(1, 10))
		gamma_p = primitives.float2int(random.randint(1, 10))
		beta_p = alpha_p + a

		(alpha_p_1, alpha_p_2) = primitives.additive_sharing(alpha_p)

		p0.a = (alpha_p_1)
		p0.b = (alpha_p_2)
		p0.c = (beta_p + gamma_p) % (conf.modl)

		p1.a = (alpha_p_1)
		p1.b = (beta_p)
		p1.c = (gamma_p)

		p2.a = (alpha_p_2)
		p2.b = (beta_p)
		p2.c = (gamma_p)
		return (p0, p1, p2)

	def rec_additive(share_1, share_2):
		return share_1 + share_2

	def rec_special(share_1, share_2):
		if (share_1.party_num == 0 and (share_2.party_num == 1 or share_2.party_num == 2)):
			alpha_val = share_1.a + share_1.b
		elif (share_1.party_num == 1 and share_2.party_num == 2):
			alpha_val = share_1.a + share_2.a
		
		val = share_2.b - alpha_val
		return val

