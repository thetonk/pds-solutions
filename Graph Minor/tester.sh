#!/bin/bash

make purge
make main pthreads openmp opencilk
make clean

#./test "$1" "$2" | sort > logs/out.log
./test-pthread "$1" "$2" $3 > logs/out-pthreads.log
./test-openmp "$1" "$2" $3 > logs/out-openmp.log
./test-opencilk "$1" "$2" $3 > logs/out-opencilk.log
cd logs/
echo "Pthreads"
tail -n 1 out-pthreads.log
echo "OpenMP"
tail -n 1 out-openmp.log
echo "OpenCilk"
tail -n 1 out-opencilk.log
cd ..
make purge