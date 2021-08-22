CC = g++-11
# CFLAGS  = -g -Wall

all:  ot-test.o
	$(CC) $(CFLAGS) ot-test.o -o file

ot-test.o:	ot-test.cpp
	$(CC) $(CFLAGS) -c ot-test.cpp

clean: 
	$(RM) file *.o *~