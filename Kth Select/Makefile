##
# Kth select openMPI
#
# @file
# @version 0.1
SHELL := /bin/bash
CC := gcc
MPICC = mpicc
RM = rm -f

.PHONY: all main openmpi clean
all: main openmpi

main: main.c include/*.c
	$(MPICC) $^ -o test -lcurl

openmpi: main-openmpi.c include/*.c
	$(MPICC) $^ -o test-openmpi -lcurl

#this is a test
#curl-openmpi: curl-openmpi.c include/curlutils.c
#	$(MPICC) $^ -o test-curl-openmpi -lcurl

clean:
	$(RM) *.o test*

# end
