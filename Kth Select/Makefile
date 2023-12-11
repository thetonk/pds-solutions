##
# Kth select openMPI
#
# @file
# @version 0.1
SHELL := /bin/bash
CC := gcc
MPICC = mpicc
RM = rm -f

.PHONY: all main openmpi clean purge
all: main openmpi

main: main.c
	$(CC) $^ -o test

openmpi: main-openmpi.c
	$(MPICC) $^ -o test-openmpi

clean:
	$(RM) *.o

purge:
	$(RM) *.o test*

# end
