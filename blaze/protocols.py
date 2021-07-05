#!/usr/bin/python
import sys
from shares import angular_share as ang
from shares import special_share as spc
from primitives import primitives as prim
import socket
import pickle
import random
import numpy as np
from gmpy2 import mpz
import time
# import math
# import os

class protocols:

	def mulZK(d,e): 

	# mulZK is from: D. Boneh, E. Boyle, H. Corrigan-Gibbs, N. Gilboa, and Y. Ishai,
	# “Zero-knowledge proofs on secret-shared data via fully linear pcps,”
	# in CRYPTO, 2019, pp. 67–97.
	
	# The implementation of verification by zero knowledge proofs is not done.
	# This function will be used as a black box. We will be assuming that 
	# this protocol is honestly executed by each server.

		f = ang() # creating a data class object for an angular share

		# Step 1
		if(conf.partyNum == 0):
			lambda_1 = prim.randsample_2(1)
			# print("0: Sampled lambda_1")
			lambda_2 = prim.randsample_2(2)
			# print("0: Sampled lambda_2")
			lambda_ = mpz((lambda_1+lambda_2)%(conf.modl))
			# print("0: Computing lambda...")
			# print("lambda: ", lambda_)

			lambda_de_1 = prim.randsample_2(1) # sampling lambda_de with P1
			# print("0: Sampled lambda_de_1")
			# Step 2
			lambda_de = mpz((d.x1 + d.x2)*(e.x1 + e.x2)%(conf.modl))
			# print("lambda_de: ", lambda_de)
			lambda_de_2 = mpz((lambda_de - lambda_de_1)%(conf.modl))

			prim.send_val(lambda_de_2,2)
			# print("0: Sent lambda_de_2")

		if(conf.partyNum == 1):
			lambda_ = prim.randsample_2(0)
			# print("1: Sampled lambda_1")
			lambda_de = prim.randsample_2(0)
			# print("1: Sampled lambda_de")

		if(conf.partyNum == 2):
			lambda_ = prim.randsample_2(0)
			# print("2: Sampled lambda_2")
			# lambda_de_1 = prim.randsample_2(1)
			# print("2: Sampled lambda_de")
			# Step 2
			lambda_de = prim.recv_val(0) # this value would be lambda_de_2
			# print("2: Received lambda_de")


		# Step 3
		if(conf.PRIMARY):
			# print("lambda_de_"+str(conf.partyNum)+": ", lambda_de)
			# print(" lamda_"+str(conf.partyNum)+": ",lambda_)
			f_lf = mpz(((mpz(conf.partyNum) - mpz(1))*(d.x2*e.x2) - (d.x1*e.x2) - (e.x1*d.x2) + (lambda_de + lambda_))%(conf.modl))
			
			# print(str(conf.partyNum)+" partyNum - 1: ", (mpz(conf.partyNum) - mpz(1)))
			other_f_lf = prim.send_recv_val(f_lf,conf.adv_party)

			f_lf = mpz((f_lf + other_f_lf)%(conf.modl))
			# print("f_lf: ",f_lf)
			# setting the share

			f.x1 = lambda_
			f.x2 = f_lf

		else:
			# setting the share
			f.x1 = lambda_1
			f.x2 = lambda_2

		return f
	

	def multiplication(a,b):

		c = spc()

		################# PREPROCESSING #########################
		
		print("Preprocessing...")
		# Step 1
		if(conf.partyNum == 0):
			alpha_1 = prim.randsample_2(1)
			# print("Sampled alpha_1")
			alpha_2 = prim.randsample_2(2)
			# print("Sampled alpha_2")
			# print("Computing alpha...")
			alpha = (alpha_1+alpha_2)%(conf.modl)

		if(conf.partyNum == 1):
			alpha = prim.randsample_2(0)
			# print("Sampled alpha_1")
			gamma = prim.randsample_2(2)
			# print("Sampled gamma")

		if(conf.partyNum == 2):
			alpha = prim.randsample_2(0)
			# print("Sampled alpha_2")
			gamma = prim.randsample_2(1)
			# print("Sampled gamma: ", gamma)

		
		# Step 2
		d = ang()
		e = ang()

		if(conf.partyNum == 0):
			d.x1 = a.x1
			d.x2 = a.x2
			e.x1 = b.x1
			e.x2 = b.x2

		if(conf.PRIMARY):
			d.x1 = a.x1
			d.x2 = a.x3
			e.x1 = b.x1
			e.x2 = b.x3

		# Step 3
		f = protocols.mulZK(d,e)

		# Step 4
		if(conf.partyNum == 0):
			chi_1 = f.x1
			chi_2 = f.x2
			chi = (chi_1+chi_2)%(conf.modl)

		if(conf.PRIMARY):
			chi = f.x1

			psi = f.x2 - ((a.x3)*(b.x3))%(conf.modl) # is truncation needed here?

		# Step 5
		if(conf.PRIMARY):
			r = prim.randsample_2(conf.adv_party) # P1,P2 sample random r
			psi_1 = r
			psi_2 = (psi - r)%(conf.modl)

			Psi = [psi,psi_1,psi_2] # to access all the psi's

		# Step 6
			gamma_xy = ((a.x3)*(b.x1) + (b.x3)*(a.x1) + (Psi[conf.partyNum]-chi))%(conf.modl)
			# print(str(conf.partyNum)+" gamma_xy",gamma_xy)

		##################### ONLINE #########################

		print("Running the online phase...")
		# Step 1

		if(conf.PRIMARY):
			my_beta_z = ((conf.partyNum - 1)* (a.x2)*(b.x2) - (a.x2)*(b.x1) - (b.x2)*(a.x1) + (gamma_xy + alpha))%(conf.modl)

			adv_beta_z = prim.send_recv_val(my_beta_z,conf.adv_party)
			beta_z = (my_beta_z + adv_beta_z)%(conf.modl)
			beta_z = beta_z
			# print(str(conf.partyNum)+" beta_z",beta_z)
			# print("psi: ",Psi[0])
			# print("Bx.By: ",np.multiply((a.x2),(b.x2)))

		if(conf.MODE!=1):
			# Step 2
			if(conf.partyNum == 0):
				gamma_xy = (a.x1 + a.x2)*(b.x1 + b.x2) # gamma_xy = alpha_x X alpha_y
				
				# print("0 gamma_xy",gamma_xy)
				beta_zstar = ((-1 * (a.x3)*(b.x1 + b.x2)) - (b.x3)*(a.x1 + a.x2) + alpha + 2*gamma_xy + chi)%(conf.modl)
				print('Party 0: beta_zstar: ',(beta_zstar))
				print('Party 0: Hash(beta_zstar): ',prim.Hash(beta_zstar))
				prim.send_val(prim.Hash(beta_zstar),1)
				prim.send_val(prim.Hash(beta_zstar),2)


			# Step 3
			if(conf.PRIMARY):
				
				beta_zstar = prim.recv_val(0)
				# print('beta_zstar: ',beta_zstar)
				print("Party "+str(conf.partyNum)+": Hash= ",prim.Hash((beta_z - ((a.x2)*(b.x2)) + psi) % (conf.modl))) 
				print("Party "+str(conf.partyNum)+' beta_zstar computed: ',(beta_z - ((a.x2)*(b.x2)) + psi) % (conf.modl))
				assert beta_zstar == prim.Hash((beta_z - ((a.x2)*(b.x2)) + psi) % (conf.modl))
				if(conf.partyNum == 1):
					# send(beta_z + gamma) to P_0
					prim.send_val((beta_z + gamma)%(conf.modl),0)
				else:
					# P2 send(prim.Hash(beta_z + gamma)) to P_0
					prim.send_val(prim.Hash((beta_z + gamma)%(conf.modl)),0)

			if(conf.partyNum == 0):
				# receive() from P_1
				bg = prim.recv_val(1)
				# receive() from P_2
				h_bg = prim.recv_val(2)
				# assert that they're are consistent
				assert prim.Hash(bg) == h_bg

		else:
			# Step 2
			if(conf.partyNum == 0):

				bg = prim.recv_val(1)

				gamma_xy = (a.x1 + a.x2) * (b.x1 + b.x2) # gamma_xy = alpha_x X alpha_y
				# print("0 gamma_xy",gamma_xy)
				beta_zstar = (-1 * a.x3 * (b.x1 + b.x2) - b.x3 * (a.x1 + a.x2) + alpha + (2*gamma_xy) + chi)%(conf.modl)
				print('Party 0: beta_zstar: ',(beta_zstar))
				print('Party 0: Hash(beta_zstar): ',prim.Hash(beta_zstar))
				print('Party 0: Hash(beta_zstar)^Hash(bg): ',prim.byte_xor(prim.Hash(beta_zstar),prim.Hash(bg)))
				prim.send_val(prim.byte_xor(prim.Hash(beta_zstar),prim.Hash(bg)),1)
				prim.send_val(prim.byte_xor(prim.Hash(beta_zstar),prim.Hash(bg)),2)
			
			# Step 3
			if(conf.PRIMARY):

				bg = (beta_z + gamma)%(conf.modl)
				if(conf.partyNum == 1):
					# send(beta_z + gamma) to P_0
					prim.send_val((bg)%(conf.modl),0)
				h_bzs = (beta_z - (a.x2) * (b.x2) + psi)%(conf.modl)

				print(h_bzs)
				print(prim.Hash(bg))
				beta_zstar = prim.recv_val(0)
				# print('beta_zstar: ',beta_zstar)
				# print("Party "+str(conf.partyNum)+": Hash= ",prim.Hash((beta_z - (a.x2) * (b.x2) + psi)%(conf.modl)))  
				# print("Party "+str(conf.partyNum)+' beta_zstar computed: ',(beta_z - (a.x2) * (b.x2) + psi)%(conf.modl))
				# print("Party "+str(conf.partyNum)+": Hash(beta_zstar)^Hash(bg)= ",prim.byte_xor(prim.Hash(h_bzs),bytes(prim.Hash(bg))))
				assert beta_zstar == prim.byte_xor(prim.Hash(h_bzs),bytes(prim.Hash(bg)))
					
		if(conf.PRIMARY):
			c.x1 = alpha
			c.x2 = beta_z
			c.x3 = gamma
		else:
			c.x1 = alpha_1
			c.x2 = alpha_2
			c.x3 = bg

		return c