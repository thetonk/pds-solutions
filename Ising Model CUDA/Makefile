##
# Ising Model CUDA
#
# @file
# @version 0.1
SHELL := /bin/bash

NVCC = /usr/local/cuda-12.0/bin/nvcc
CC = gcc -O3
RM = rm -f

.PHONY: hello clean sequential cudaV1 test cudaV2 cudaV3
all: sequential cudaV1 cudaV2 cudaV3

sequential: ising.c
	$(CC) $^ -o test-ising

cudaV1: isingV1.cu
	$(NVCC) -O3 -arch native $^ -o test-isingV1

cudaV2: isingV2.cu
	$(NVCC) -O3 -arch native $^ -o test-isingV2

cudaV3: isingV3.cu
	$(NVCC) -O3 -arch native $^ -o test-isingV3

clean:
	$(RM) hello test-*
# end
