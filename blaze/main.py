import sys
from primitives import primitives as prim
from protocols import protocols
from shares import angular_share as ang
from shares import special_share as spc
from config import config as conf
import numpy as np
import random
from gmpy2 import mpz

def additive_sharing(a):
	a1 = random.randint(1, 10)
	a2 = a - a1
	return (a1,a2)

def main():

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TEST BLAZE MULT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

	print("=== Testing Integer Enbedding ===")
	
	print("Number is 9")
	f = prim.float2int(9)
	print(f"Integer Ring embedding is: {f}")
	i = prim.int2float(f)
	print(f"OG Float is: {i}")

	print("Number is 12.21")
	f = prim.float2int(12.21)
	print(f"Integer Ring embedding is: {f}")
	i = prim.int2float(f)
	print(f"OG Float is: {i}")

	print("Number is -12.21")
	f = prim.float2int(-12.21)
	print(f"Integer Ring embedding is: {f}")
	i = prim.int2float(f)
	print(f"OG Float is: {i}")

	print("======================")

	

	x0 = spc()
	x1 = spc()
	x2 = spc()

	y0 = spc()
	y1 = spc()
	y2 = spc()

	z0 = spc()
	z1 = spc()
	z2 = spc()

	x = prim.float2int(3)


	my_r = primitives.float2int(random.uniform(-1*conf.max_dec,conf.max_dec)) # sample a random float value from the possible range and embed it on the ring

	r = (mpz(my_r)+mpz(adv_r))%(conf.modl)

	alpha_x = prim.float2int(random.randint(1, 10))
	y = prim.float2int(4)
	alpha_y = prim.float2int(random.randint(1, 10))
	gamma_x = prim.float2int(random.randint(1, 10))
	gamma_y = prim.float2int(random.randint(1, 10))

	beta_x = alpha_x + x
	beta_y = alpha_y + y

	(alpha_x_1, alpha_x_2) = additive_sharing(alpha_x)
	(alpha_y_1, alpha_y_2) = additive_sharing(alpha_y)

	x0.a = mpz(alpha_x_1)
	x0.b = mpz(alpha_x_2)
	x0.c = (mpz(beta_x) + mpz(gamma_x)) % (conf.modl)

	x1.a = mpz(alpha_x_1)
	x1.b = mpz(beta_x)
	x1.c = mpz(gamma_x)

	x2.a = mpz(alpha_x_2)
	x2.b = mpz(beta_x)
	x2.c = mpz(gamma_x)

	y0.a = mpz(alpha_y_1)
	y0.b = mpz(alpha_y_2)
	y0.c = (mpz(beta_y) + mpz(gamma_y)) % (conf.modl)

	y1.a = mpz(alpha_y_1)
	y1.b = mpz(beta_y)
	y1.c = mpz(gamma_y)

	y2.a = mpz(alpha_y_2)
	y2.b = mpz(beta_y)
	y2.c = mpz(gamma_y)


	#print(type(a.x1))
	print(f"Special Shares of x: {prim.int2float(x)}")
	print(x0.a)
	print(x0.b)
	print(x0.c)
	print(x1.a)
	print(x1.b)
	print(x1.c)
	print(x2.a)
	print(x2.b)
	print(x2.c)

	print(f"Special Shares of y: {prim.int2float(y)}")
	print(y0.a)
	print(y0.b)
	print(y0.c)
	print(y1.a)
	print(y1.b)
	print(y1.c)
	print(y2.a)
	print(y2.b)
	print(y2.c)

	# muliplication

	alpha_z = prim.float2int(random.randint(1, 10))
	(alpha_z_1, alpha_z_2) = additive_sharing(alpha_z)

	# P0 calculates:
	alpha_x_alpha_y = alpha_x * alpha_y
	(alpha_x_alpha_y_1, alpha_x_alpha_y_2) = additive_sharing(alpha_x_alpha_y)

	# P1 calculates:
	beta_z_1 = - (beta_x * alpha_y_1) - (beta_y * alpha_x_1) + alpha_x_alpha_y_1 + alpha_z_1

	# P2 calculates:
	beta_z_2 = (beta_x * beta_y) - (beta_x * alpha_y_2) - (beta_y * alpha_x_2) + alpha_x_alpha_y_2 + alpha_z_2

	beta_z = beta_z_1 + beta_z_2
	gamma_z = prim.float2int(random.randint(1, 10)) # change later

	z0.a = mpz(alpha_z_1)
	z0.b = mpz(alpha_z_2)
	z0.c = (mpz(beta_z) + mpz(gamma_z)) % (conf.modl)

	z1.a = mpz(alpha_z_1)
	z1.b = mpz(beta_z)
	z1.c = mpz(gamma_z)

	z2.a = mpz(alpha_z_2)
	z2.b = mpz(beta_z)
	z2.c = mpz(gamma_z)

	print(f"Special Shares of z = x.y: {prim.int2float(x * y)}")
	print(z0.a)
	print(z0.b)
	print(z0.c)
	print(z1.a)
	print(z1.b)
	print(z1.c)
	print(z2.a)
	print(z2.b)
	print(z2.c)

	# Truncation
	# ======================

	r = prim.float2int(12.3)
	print(r)
	print(bin(r))
	r0 = r
	(r1, r2) = additive_sharing(r)

	rd = int(r//(2**13))
	print(rd)
	print(bin(rd))

	alpha_rd = prim.float2int(random.randint(1, 20))
	beta_rd = alpha_rd + rd
	gamma_rd = prim.float2int(random.randint(1, 20))

	rd0 = spc()
	rd1 = spc()
	rd2 = spc()

	(alpha_rd_1, alpha_rd_2) = additive_sharing(alpha_rd)

	rd0.a = mpz(alpha_rd_1)
	rd0.b = mpz(alpha_rd_2)
	rd0.c = (mpz(beta_rd) + mpz(gamma_rd)) % (conf.modl)

	rd1.a = mpz(alpha_rd_1)
	rd1.b = mpz(beta_rd)
	rd1.c = mpz(gamma_rd)

	rd2.a = mpz(alpha_rd_2)
	rd2.b = mpz(beta_rd)
	rd2.c = mpz(gamma_rd)

	#print(type(a.x1))
	print(f"Special Shares of rd: {prim.int2float(rd)}")
	print(rd0.a)
	print(rd0.b)
	print(rd0.c)
	print(rd1.a)
	print(rd1.b)
	print(rd1.c)
	print(rd2.a)
	print(rd2.b)
	print(rd2.c)

	# P1 calculates:
	zr_1 = beta_z_1 - alpha_z_1

	# P2 calculates:
	zr_2 = beta_z_2 - alpha_z_2

	zr = zr_1 + zr_2 # P1 and P2 Reconstruct (z-r)
	zrd = int(rd//(2**13))

	zrd0 = spc()
	zrd1 = spc()
	zrd2 = spc()

	alpha_zrd = prim.float2int(random.randint(1, 20))
	beta_zrd = alpha_zrd + zrd
	gamma_zrd = prim.float2int(random.randint(1, 20))

	(alpha_zrd_1, alpha_zrd_2) = additive_sharing(alpha_zrd)

	zrd0.a = mpz(alpha_zrd_1)
	zrd0.b = mpz(alpha_zrd_2)
	zrd0.c = (mpz(beta_zrd) + mpz(gamma_zrd)) % (conf.modl)

	zrd1.a = mpz(alpha_zrd_1)
	zrd1.b = mpz(beta_zrd)
	zrd1.c = mpz(gamma_zrd)

	zrd2.a = mpz(alpha_zrd_2)
	zrd2.b = mpz(beta_zrd)
	zrd2.c = mpz(gamma_zrd)

	# ======================

	z0.a = mpz(zrd0.a + rd0.a)
	z0.b = mpz(zrd0.b + rd0.b)
	z0.c = mpz(zrd0.c + rd0.c)

	z1.a = mpz(zrd1.a + rd1.a)
	z1.b = mpz(zrd1.b + rd1.b)
	z1.c = mpz(zrd1.c + rd1.c)

	z2.a = mpz(zrd2.a + rd2.a)
	z2.b = mpz(zrd2.b + rd2.b)
	z2.c = mpz(zrd2.c + rd2.c)

	print(f"New Special Shares of z: {prim.int2float(x) * prim.int2float(y)}")

	print(z0.a)
	print(z0.b)
	print(z0.c)
	print(z1.a)
	print(z1.b)
	print(z1.c)
	print(z2.a)
	print(z2.b)
	print(z2.c)
	
	#c = protocols.multiplication(a,b)

	#print("Party:" + " Share 0:", c.x1)
	#print("Party:" + " Share 1:", c.x2)
	#print("Party:" + " Share 2:", c.x3)


	#if(conf.partyNum == 0):
		#alp = (c.x1 + c.x2) % (conf.modl)
		# print("\n\nalpha: ",alp,end="\n")
		#prim.send_val(alp,1)
		#prim.send_val(alp,2)

	#else:
		#alp = prim.recv_val(0)
		#v = (c.x2 - alp) % (conf.modl)

		#print("Product: ",v)



if __name__ == '__main__':
	main()