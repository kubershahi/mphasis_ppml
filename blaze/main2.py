import sys
from primitives import primitives as prim
from protocols import protocols
from shares import angular_share as ang
from shares import special_share as spc
from config import config as conf
import numpy as np
import random
from gmpy2 import mpz

np.seterr(over='ignore')

def additive_sharing(a):
	a1 = prim.float2int(random.randint(1, 10))
	a2 = a - a1
	return (a1,a2)

def rec_special(share_1, share_2):
	if (share_1.party_num == 0 and (share_2.party_num == 1 or share_2.party_num == 2)):
		alpha_val = share_1.a + share_1.b
	elif (share_1.party_num == 1 and share_2.party_num == 2):
		alpha_val = share_1.a + share_2.a
	
	val = share_2.b - alpha_val
	return val

def mult(z0, z1, z2, alpha_x, alpha_y, alpha_x_1, alpha_x_2, alpha_y_1, alpha_y_2, beta_x, beta_y):
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

	z0.a = (alpha_z_1)
	z0.b = (alpha_z_2)
	z0.c = ((beta_z) + (gamma_z)) % (conf.modl)

	z1.a = (alpha_z_1)
	z1.b = (beta_z)
	z1.c = (gamma_z)

	z2.a = (alpha_z_2)
	z2.b = (beta_z)
	z2.c = (gamma_z)
	
	return (z0, z1, z2)

def main():

	print("=== Testing Integer Enbedding ===")
	
	nums = np.array([9, 12.21, -12.21])
	i = np.array([0.0,0.0,0.0])
	print("Numbers are 9, 12.21, -12.21")
	f = prim.float2int(nums)
	print(f"Integer Ring embedding is: {f}")
	for j in range(0, len(f)):
		i[j] = prim.int2float(f[j])
	print(f"OG Float is: {i}")

	print("====================== \n \n")

	print("=== Testing Special Sharing Semantics ===")

	x0 = spc(0)
	x1 = spc(1)
	x2 = spc(2)

	y0 = spc(0)
	y1 = spc(1)
	y2 = spc(2)

	x = prim.float2int(3)
	y = prim.float2int(4)

	alpha_x = prim.float2int(random.randint(1, 10))
	alpha_y = prim.float2int(random.randint(1, 10))
	gamma_x = prim.float2int(random.randint(1, 10))
	gamma_y = prim.float2int(random.randint(1, 10))

	beta_x = alpha_x + x
	beta_y = alpha_y + y

	(alpha_x_1, alpha_x_2) = additive_sharing(alpha_x)
	(alpha_y_1, alpha_y_2) = additive_sharing(alpha_y)

	x0.a = (alpha_x_1)
	x0.b = (alpha_x_2)
	x0.c = (beta_x + gamma_x) % (conf.modl)

	x1.a = (alpha_x_1)
	x1.b = (beta_x)
	x1.c = (gamma_x)

	x2.a = (alpha_x_2)
	x2.b = (beta_x)
	x2.c = (gamma_x)

	y0.a = (alpha_y_1)
	y0.b = (alpha_y_2)
	y0.c = (beta_y + gamma_y) % (conf.modl)

	y1.a = (alpha_y_1)
	y1.b = (beta_y)
	y1.c = (gamma_y)

	y2.a = (alpha_y_2)
	y2.b = (beta_y)
	y2.c = (gamma_y)


	#print(type(a.x1))
	print(f"Special Shares of x: {prim.int2float(x)}")
	print(f"{x0.a} --> {prim.int2float(x0.a)}")
	print(f"{x0.b} --> {prim.int2float(x0.b)}")
	print(f"{x0.c} --> {prim.int2float(x0.c)}")
	print(f"{x1.a} --> {prim.int2float(x1.a)}")
	print(f"{x1.b} --> {prim.int2float(x1.b)}")
	print(f"{x1.c} --> {prim.int2float(x1.c)}")
	print(f"{x2.a} --> {prim.int2float(x2.a)}")
	print(f"{x2.b} --> {prim.int2float(x2.b)}")
	print(f"{x2.c} --> {prim.int2float(x2.c)}")
	print(f"Reconstructed value of x: {prim.int2float(rec_special(x0, x1))} \n")

	print(f"Special Shares of y: {prim.int2float(y)}")
	print(f"{y0.a} --> {prim.int2float(y0.a)}")
	print(f"{y0.b} --> {prim.int2float(y0.b)}")
	print(f"{y0.c} --> {prim.int2float(y0.c)}")
	print(f"{y1.a} --> {prim.int2float(y1.a)}")
	print(f"{y1.b} --> {prim.int2float(y1.b)}")
	print(f"{y1.c} --> {prim.int2float(y1.c)}")
	print(f"{y2.a} --> {prim.int2float(y2.a)}")
	print(f"{y2.b} --> {prim.int2float(y2.b)}")
	print(f"{y2.c} --> {prim.int2float(y2.c)}")
	print(f"Reconstructed value of x: {prim.int2float(rec_special(y0, y1))} \n")

	
	print("====================== \n \n")
	# muliplication
	print("=== Testing Multipication (Honest Parties) ===")

	# z = x.y
	z0 = spc(0)
	z1 = spc(1)
	z2 = spc(2)

	(z0, z1, z2) = mult(z0, z1, z2, alpha_x, alpha_y, alpha_x_1, alpha_x_2, alpha_y_1, alpha_y_2, beta_x, beta_y)

	print(f"Special Shares of z = x.y: {prim.int2float(x) * prim.int2float(y)}")
	print(f"{z0.a} --> {prim.int2float(z0.a)}")
	print(f"{z0.b} --> {prim.int2float(z0.b)}")
	print(f"{z0.c} --> {prim.int2float(z0.c)}")
	print(f"{z1.a} --> {prim.int2float(z1.a)}")
	print(f"{z1.b} --> {prim.int2float(z1.b)}")
	print(f"{z1.c} --> {prim.int2float(z1.c)}")
	print(f"{z2.a} --> {prim.int2float(z2.a)}")
	print(f"{z2.b} --> {prim.int2float(z2.b)}")
	print(f"{z2.c} --> {prim.int2float(z2.c)}")
	print(f"Reconstructed value of z: {prim.int2float(prim.int2float(rec_special(z0, z1)))}")

	print("====================== \n \n")
	sys.exit("End of testing (works upto here)")
	
	# Truncation
	print("=== Testing Truncation ===")

	r = prim.float2int(12.3)
	r0 = r
	(r1, r2) = additive_sharing(r)

	rd = int(r//(2**13))

	alpha_rd = prim.float2int(random.randint(1, 20))
	beta_rd = alpha_rd + rd
	gamma_rd = prim.float2int(random.randint(1, 20))

	rd0 = spc(0)
	rd1 = spc(1)
	rd2 = spc(2)

	(alpha_rd_1, alpha_rd_2) = additive_sharing(alpha_rd)

	rd0.a = (alpha_rd_1)
	rd0.b = (alpha_rd_2)
	rd0.c = (beta_rd + gamma_rd) % (conf.modl)

	rd1.a = (alpha_rd_1)
	rd1.b = (beta_rd)
	rd1.c = (gamma_rd)

	rd2.a = (alpha_rd_2)
	rd2.b = (beta_rd)
	rd2.c = (gamma_rd)

	#print(type(a.x1))
	print(f"Special Shares of rd: {prim.int2float(rd)}")

	# P1 calculates:
	zr_1 = beta_z_1 - alpha_z_1

	# P2 calculates:
	zr_2 = beta_z_2 - alpha_z_2

	zr = zr_1 + zr_2 # P1 and P2 Reconstruct (z-r)
	zrd = int(rd//(2**13))

	zrd0 = spc(0)
	zrd1 = spc(1)
	zrd2 = spc(2)

	alpha_zrd = prim.float2int(random.randint(1, 20))
	beta_zrd = alpha_zrd + zrd
	gamma_zrd = prim.float2int(random.randint(1, 20))

	(alpha_zrd_1, alpha_zrd_2) = additive_sharing(alpha_zrd)

	zrd0.a = (alpha_zrd_1)
	zrd0.b = (alpha_zrd_2)
	zrd0.c = (beta_zrd + gamma_zrd) % (conf.modl)

	zrd1.a = (alpha_zrd_1)
	zrd1.b = (beta_zrd)
	zrd1.c = (gamma_zrd)

	zrd2.a = (alpha_zrd_2)
	zrd2.b = (beta_zrd)
	zrd2.c = (gamma_zrd)

	# ======================

	z0.a = (zrd0.a + rd0.a)
	z0.b = (zrd0.b + rd0.b)
	z0.c = (zrd0.c + rd0.c)

	z1.a = (zrd1.a + rd1.a)
	z1.b = (zrd1.b + rd1.b)
	z1.c = (zrd1.c + rd1.c)

	z2.a = (zrd2.a + rd2.a)
	z2.b = (zrd2.b + rd2.b)
	z2.c = (zrd2.c + rd2.c)

	print(f"New Special Shares of z: {prim.int2float(x) * prim.int2float(y)}")
	print(f"{z0.a} --> {prim.int2float(z0.a)}")
	print(f"{z0.b} --> {prim.int2float(z0.b)}")
	print(f"{z0.c} --> {prim.int2float(z0.c)}")
	print(f"{z1.a} --> {prim.int2float(z1.a)}")
	print(f"{z1.b} --> {prim.int2float(z1.b)}")
	print(f"{z1.c} --> {prim.int2float(z1.c)}")
	print(f"{z2.a} --> {prim.int2float(z2.a)}")
	print(f"{z2.b} --> {prim.int2float(z2.b)}")
	print(f"{z2.c} --> {prim.int2float(z2.c)}")
	print(f"Reconstructed value of z: {prim.int2float(prim.int2float(rec_special(z0, z1)))}")

	


if __name__ == '__main__':
	main()