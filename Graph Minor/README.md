# Graph Minor Homework 1
This folder contains the source code of the corresponding assignment. The report can be found [here](report/report.pdf).

## Building instructions
A makefile is provided for this purpose.

### Sequential
Can be built with `make main`. The binary is called `test`.

### P-threads
Can be built with `make pthreads`. The binary is called `test-pthread`.

### OpenMP
Can be built with `make openmp`. The binary is called `test-openmp`.

### OpenCilk
Can be built with `make opencilk`. The binary is called `test-opencilk`.

## Running the binaries
Datasets can be passed into the program from the terminal using the following 
syntax:

`$ <binary name> <path to matrix market file> <path to file containing vector c>`

## Deleting build files
Easy as `make clean`. To delete the build files and the binaries too, use `make purge`.
