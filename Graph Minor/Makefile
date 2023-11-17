##
# This is a makefile woo
#
# @file graph_minor
# @version 0.1
SHELL := /bin/bash

CC = gcc
CLANG = clang
RM = rm -f

.PHONY: all
all: main

main-openmp.o: main-openmp.c
	$(CC) -fopenmp -c $^

main-opencilk.o: main-opencilk.c
	$(CLANG) -c -fopencilk -O3 $^

%.o: %.c
	$(CC) -c $<

.PHONY: main

main: mmio.o main.o
	$(CC) $^ -o test

pthreads: mmio.o main-pthreads.o
	$(CC) -lpthread $^ -o test-pthread

openmp: mmio.o main-openmp.o
	$(CC) $^ -fopenmp -o test-openmp
	
# add -fsanitize=cilk flag to test data races
opencilk: mmio.o main-opencilk.o
	$(CLANG) -fopencilk $^ -o test-opencilk

.PHONY: clean
clean:
	$(RM) *.o *.out

purge:
	$(RM) *.o test test-pthread test-openmp test-opencilk
# end
