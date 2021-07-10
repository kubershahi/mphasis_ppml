# Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#
CC = g++
# CFLAGS  = -g -Wall

all:  read_data.o trun.o 
	$(CC) $(CFLAGS) read_data.o trun.o -o file

read_data.o:	read_data.cpp read_data.hpp
	$(CC) $(CFLAGS) -c read_data.cpp

trun.o:	trun.cpp read_data.hpp
	$(CC) $(CFLAGS) -c trun.cpp

# To start over from scratch, type 'make clean'. This removes the executable file, 
# as well as old .o objectfiles and *~ backup files:
clean: 
	$(RM) file *.o *~
