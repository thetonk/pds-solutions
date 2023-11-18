#!/bin/bash

make purge
make main pthreads openmp opencilk
make clean

./test "$1" "$2" | sort > logs/out.log
./test-pthread "$1" "$2" | sort > logs/out-pthreads.log
./test-openmp "$1" "$2" | sort > logs/out-openmp.log
./test-opencilk "$1" "$2" | sort > logs/out-opencilk.log

cd logs/
