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
all: sequential cudaV1 cudaV2 cudaV3
hello: hello.cu
	$(NVCC) -O3 -arch native $^ -o hello

sequential: ising.c
	$(CC) $^ -o test-ising

cudaV1: isingV1.cu
	$(NVCC) -arch native $^ -o test-isingV1

cudaV2: isingV2.cu
	$(NVCC) -arch native $^ -o test-isingV2

cudaV3: isingV3.cu
	$(NVCC) -arch native $^ -o test-isingV3

test-sequential: sequential
	./test-ising 10

test-cudaV1: cudaV1
	./test-isingV1 10 30

test-cudaV2: cudaV2
	./test-isingV2 10 5

test-cudaV3: cudaV3
	./test-isingV3
clean:
	$(RM) hello test-*
# end
