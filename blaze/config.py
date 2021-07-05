class config:
	
	l = 64
	lby2 = 32
	modl = 2**l
	max_dec = 0 # maximum positive decimal value possible to express in the integer ring

	precision = 13
	converttoint64 = (1<<precision)
	trunc_parameter = (1>>precision)
	
	epochs = 10
	