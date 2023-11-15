##
# This is a makefile woo
#
# @file graph_minor
# @version 0.1
SHELL := /bin/bash

CC = gcc
RM = rm -f

.PHONY: all
all: main

%.o: %.c
	$(CC) -c $<

.PHONY: main

main: mmio.o main.o
	$(CC) $^ -o test

main-pthreads: mmio.o main-pthreads.o
	$(CC) $^ -o test-pthread

openmp:
	echo "not implemented yet"

opencilk:
	echo "not implemented yet"

.PHONY: clean
clean:
	$(RM) *.o

purge:
	$(RM) *.o test test-pthread
# end
