# Makefile variables for the compiler and compiler flags
# to use Makefile variables later in the Makefile: $()
#
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings
#
CC = g++
# CFLAGS  = -g -Wall

all:  read_data.o main.o utils.o linear_regression.o
	$(CC) $(CFLAGS) read_data.o utils.o linear_regression.o main.o -o file

linear_regression.o:	linear_regression.cpp linear_regression.hpp
	$(CC) $(CFLAGS) -c linear_regression.cpp

logistic_regression.o:	logistic_regression.cpp logistic_regression.hpp
	$(CC) $(CFLAGS) -c logistic_regression.cpp

utils.o:	utils.cpp utils.hpp
	$(CC) $(CFLAGS) -c utils.cpp

read_data.o:	read_data.cpp read_data.hpp
	$(CC) $(CFLAGS) -c read_data.cpp

main.o:	main.cpp read_data.hpp utils.hpp linear_regression.hpp defines.hpp
	$(CC) $(CFLAGS) -c main.cpp

# To start over from scratch, type 'make clean'. This removes the executable file, 
# as well as old .o objectfiles and *~ backup files:
clean: 
	$(RM) file *.o *~
