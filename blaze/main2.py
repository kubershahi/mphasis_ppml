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

def add(x0, x1, x2, y0, y1, y2):

	(z0, z1, z2) = prim.special_sharing(prim.float2int(0))
	z0.a = x0.a + y0.a
	z0.b = x0.b + y0.b
	z0.c = x0.c + y0.c

	z1.a = x1.a + y1.a
	z1.b = x1.b + y1.b
	z1.c = x1.c + y1.c

	z2.a = x2.a + y2.a
	z2.b = x2.b + y2.b
	z2.c = x2.c + y2.c

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
	x = prim.float2int(30.1242)
	y = prim.float2int(14.9938)

	(x0, x1, x2) = prim.special_sharing(x)
	(y0, y1, y2) = prim.special_sharing(y)

	print(f"Special Shares of x: {prim.int2float(x)}")
	prim.print_special(x0, x1, x2)
	print(f"Reconstructed value of x: {prim.int2float(prim.rec_special(x0, x1))} \n")

	print(f"Special Shares of y: {prim.int2float(y)}")
	prim.print_special(y0, y1, y2)
	print(f"Reconstructed value of x: {prim.int2float(prim.rec_special(y0, y1))} \n")
	print("====================== \n \n")
	
	# muliplication
	print("=== Testing Multipication (Honest Parties) ===")
	# z = x.y
	z0 = spc(0)
	z1 = spc(1)
	z2 = spc(2)
	(z0, z1, z2) = protocols.mult(z0, z1, z2, x0, x1, x2, y0, y1, y2) 

	print(f"Special Shares of z = x.y: {prim.int2float(x) * prim.int2float(y)}")
	prim.print_special(z0, z1, z2)
	print(f"Reconstructed value of z: {prim.int2float(prim.int2float(prim.rec_special(z0, z1)))}")
	print("====================== \n \n")
	
	
	

	# Dot Product
	x = np.array([1,2,3])
	y = np.array([4,5,6])
	z = np.array([0,0,0])
	x = prim.float2int(x)
	y = prim.float2int(y)

	x0 = np.array([spc(0),spc(0),spc(0)])
	x1 = np.array([spc(1),spc(1),spc(1)])
	x2 = np.array([spc(2),spc(2),spc(2)])

	y0 = np.array([spc(0),spc(0),spc(0)])
	y1 = np.array([spc(1),spc(1),spc(1)])
	y2 = np.array([spc(2),spc(2),spc(2)])

	z0 = np.array([spc(0),spc(0),spc(0)])
	z1 = np.array([spc(1),spc(1),spc(1)])
	z2 = np.array([spc(2),spc(2),spc(2)])

	z0_sum = spc(0)
	z1_sum = spc(1)
	z2_sum = spc(2)

	for i in range(0, len(x)):
		(x0[i], x1[i], x2[i]) = prim.special_sharing(x[i])
		(y0[i], y1[i], y2[i]) = prim.special_sharing(y[i])
		(z0[i], z1[i], z2[i]) = protocols.mult(z0[i], z1[i], z2[i], x0[i], x1[i], x2[i], y0[i], y1[i], y2[i])
		print(f"Reconstructed value of x_i. y_i: {prim.int2float(prim.int2float(prim.rec_special(z0[i], z1[i])))}")
		(z0_sum, z1_sum, z2_sum) = add(z0_sum, z1_sum, z2_sum, z0[i], z1[i], z2[i])

	print(f"Reconstructed value of z (dot product): {prim.int2float(prim.int2float(prim.rec_special(z0_sum, z1_sum)))}")
	sys.exit("End of testing (works upto here)")

	# Truncation
	print("=== Testing Truncation ===")

	r = prim.float2int(12.3)
	r0 = r
	(r1, r2) = prim.additive_sharing(r)

	rd = int(r//(2**13))

	alpha_rd = prim.float2int(random.randint(1, 20))
	beta_rd = alpha_rd + rd
	gamma_rd = prim.float2int(random.randint(1, 20))

	rd0 = spc(0)
	rd1 = spc(1)
	rd2 = spc(2)

	(alpha_rd_1, alpha_rd_2) = prim.additive_sharing(alpha_rd)

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

	(alpha_zrd_1, alpha_zrd_2) = prim.additive_sharing(alpha_zrd)

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
	print(f"Reconstructed value of z: {prim.int2float(prim.int2float(prim.rec_special(z0, z1)))}")

	


if __name__ == '__main__':
	main()