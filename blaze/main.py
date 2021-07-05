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

	x0 = spc()
	x1 = spc()
	x2 = spc()

	y0 = spc()
	y1 = spc()
	y2 = spc()

	z0 = spc()
	z1 = spc()
	z2 = spc()

	x = 3
	alpha_x = random.randint(1, 10)
	y = 4
	alpha_y = random.randint(1, 10)
	gamma_x = random.randint(1, 10)
	gamma_y = random.randint(1, 10)

	beta_x = alpha_x + x
	beta_y = alpha_y + y

	(alpha_x_1, alpha_x_2) = additive_sharing(alpha_x)
	(alpha_y_1, alpha_y_2) = additive_sharing(alpha_y)

	x0.a = mpz(alpha_x_1)
	x0.b = mpz(alpha_x_2)
	x0.c = mpz(beta_x + gamma_x)

	x1.a = mpz(alpha_x_1)
	x1.b = mpz(beta_x)
	x1.c = mpz(gamma_x)

	x2.a = mpz(alpha_x_2)
	x2.b = mpz(beta_x)
	x2.c = mpz(gamma_x)

	y0.a = mpz(alpha_y_1)
	y0.b = mpz(alpha_y_2)
	y0.c = mpz(beta_y + gamma_y)

	y1.a = mpz(alpha_y_1)
	y1.b = mpz(beta_y)
	y1.c = mpz(gamma_y)

	y2.a = mpz(alpha_y_2)
	y2.b = mpz(beta_y)
	y2.c = mpz(gamma_y)


	#print(type(a.x1))
	print(f"Special Shares of x: {x}")
	print(x0.a)
	print(x0.b)
	print(x0.c)
	print(x1.a)
	print(x1.b)
	print(x1.c)
	print(x2.a)
	print(x2.b)
	print(x2.c)

	print(f"Special Shares of y: {y}")
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

	alpha_z = random.randint(1, 10)
	(alpha_z_1, alpha_z_2) = additive_sharing(alpha_z)

	# P0 calculates:
	alpha_x_alpha_y = alpha_x * alpha_y
	(alpha_x_alpha_y_1, alpha_x_alpha_y_2) = additive_sharing(alpha_x_alpha_y)

	# P1 calculates:
	beta_z_1 = - (beta_x * alpha_y_1) - (beta_y * alpha_x_1) + alpha_x_alpha_y_1 + alpha_z_1

	# P2 calculates:
	beta_z_2 = (beta_x * beta_y) - (beta_x * alpha_y_2) - (beta_y * alpha_x_2) + alpha_x_alpha_y_2 + alpha_z_2

	beta_z = beta_z_1 + beta_z_2
	gamma_z = random.randint(1, 10) # change later

	z0.a = mpz(alpha_z_1)
	z0.b = mpz(alpha_z_2)
	z0.c = mpz(beta_z + gamma_z)

	z1.a = mpz(alpha_z_1)
	z1.b = mpz(beta_z)
	z1.c = mpz(gamma_z)

	z2.a = mpz(alpha_z_2)
	z2.b = mpz(beta_z)
	z2.c = mpz(gamma_z)

	print(f"Special Shares of z = x.y: {x * y}")
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


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TEST MULZK ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	# a = ang()
	# b = ang()

	# if(conf.PRIMARY):
	# 	a.x1 = mpz(2)
	# 	a.x2 = mpz(8191) + mpz(2) + mpz(2)
	# 	# a.x3 = mpz(2)

	# 	b.x1 = mpz(2)
	# 	b.x2 = mpz(8191) + mpz(2) + mpz(2)
	# 	# b.x3 = mpz(2)

	# else:
	# 	a.x1 = mpz(2)
	# 	a.x2 = mpz(2)
	# 	# a.x3 = mpz((8191 + 2) + 2)

	# 	b.x1 = mpz(2)
	# 	b.x2 = mpz(2)
	# 	# b.x3 = mpz((8191 + 2) + 2)

	# c = protocols.mulZK(a,b)
	# print(str(conf.partyNum)+" Share 0:",c.x1)
	# print(str(conf.partyNum)+" Share 1:",c.x2)

	# if(conf.partyNum == 0):
	# 	alp = (c.x1 + c.x2) % (conf.modl)
	# 	print("\n\nalpha: ",alp,end="\n")
	# 	prim.send_val(alp,1)
	# 	prim.send_val(alp,2)

	# else:
	# 	alp = prim.recv_val(0)
	# 	v = (c.x2 - alp) % (conf.modl)

	# 	print("secret share: ",v)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
	
	#print("Server "+ str(conf.partyNum)+": Total number of bytes sent: "+str(conf.num_bytes_sent)+" bytes.")
	#print("Server "+ str(conf.partyNum)+": Total number of bytes received: "+str(conf.num_bytes_received)+" bytes.")


if __name__ == '__main__':
	main()