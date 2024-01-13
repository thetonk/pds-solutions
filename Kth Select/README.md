# K-th select, MPI implementation.

This folder contains the source code of the corresponding assignment. Includes the classic, sequential algorithm of finding the k-th smallest value, and an attempt to implement it in MPI for distributed memory environments. The report can be found [here](report/report.pdf).

## Disclaimer 
The MPI version may get stuck in an infinite loop for a few cases. The reason is currently unknown. In case it happens, please try changing the number of processes, since it has been observed that fixes the issue most of the times.

## Building instructions
A makefile is provided for this purpose, for easy compiling and testing.

### Sequential
Can be built with `make main`. The binary will be called `test`.

### OpenMPI
Can be built with `make-openmpi`. The binary will be called `test-openmpi`.

## Usage
Dataset can be passed to program from the terminal, using the following syntax;

`$ ./test-openmpi <input file path> <number of elements> <element position to search>`

You may call it using the following as well;

- For local files

`$ ./test-openmpi -f <input file path> <number of elements> <element position to search>`

-  For remote files

`$ ./test-openmpi -l <URL> <element position to search>`

Please note that for the case of MPI, `mpirun` or `mpiexec` command is required.

## Deleting build files and binaries
Simply call `make clean`.
