##
# Ising Model CUDA
#
# @file
# @version 0.1
SHELL := /bin/bash

NVCC = /usr/local/cuda-12.0/bin/nvcc
CC = gcc -O3
RM = rm -f

.PHONY: hello clean sequential cudaV1 test

hello: hello.cu
	$(NVCC) -O3 -arch native $^ -o hello

sequential: ising.c
	$(CC) $^ -o test-ising

cudaV1: isingV1.cu
	$(NVCC) -O3 -arch native $^ -o test-isingV1

test-sequential: sequential
	./test-ising 10

test-cudaV1: cudaV1
	./test-isingV1 10 30

clean:
	$(RM) hello test-*
# end
