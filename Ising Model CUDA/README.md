# Ising Model - CUDA
This folder contains the source code of the assignment and the report, which can be found [here](report/report.pdf).

## Building instructions
A makefile is provided for this purpose. All the executables (sequential, CUDA V1,V2,V3) can be easily built at once using `make all`.
To build them separately;

### Sequential
Can be built with `make sequential`. The binary name will be `test-ising`.

### CUDA V1
Can be built with `make cudaV1`. The binary name will be `test-isingV1`.

### CUDA V2
Can be built with `make cudaV2`. The binary name will be `test-isingV2`. 

### CUDA V3
Can be built with `make cudaV3`. The binary name will be `test-isingV3`.

## Running the binaries
Binaries can be run using the following syntax. All positional arguments within square braces (that is, []) are optional. Lattice is assumed to be square NxN.

### Sequential

`$ test-ising <N> [<iterations>] [<seed>]`

### CUDA

`$ <binary name> <N> [<iterations>] [<block size>] [<seed>]`

Be careful with block size though, as they may be limitations of its max size. Max block size for V1 is 1024, and 32 for V3.

## Testing
For testing purposes, a couple of simple bash scripts are provided. Feel free to check them out to see if they meet your needs!

## Building troubleshooting
Depending on your CUDA version, you may counter the following issues

### Invalid NVCC path
Change nvcc path in the Makefile with yours

### -arch native not supported
CUDA versions lower than 12.0 may not support this flag. If that's the case, you may remove it.

## Removing the binaries
Simply call `make clean`
