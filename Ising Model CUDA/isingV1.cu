#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

#define N 10
#define BLOCK_SIZE 30

void printLattice(int8_t *lattice, size_t n){
    for(size_t row= 0; row <n; ++row){
        for(size_t col = 0; col < n ; ++col){
            printf("%2d ", lattice[n*row + col]);
        }
        putchar('\n');
    }
}

void generateRandomLattice(int8_t *lattice, size_t n){
    for(size_t row = 0; row < n; ++row){
        for(size_t col = 0; col < n; ++col){
            lattice[n*row + col] = 2*(rand() % 2) - 1;
        }
    }
}
__device__ int8_t getIndex(int8_t i, size_t n){
    return i == -1 ? n-1 : i % n;
}


__global__ void cuda_hello(){
    printf("Hello World from GPU, block %d and thread %d!\n", blockIdx.x, threadIdx.x);
}

__global__ void calculateNextLattice(int8_t *curLattice, int8_t *nexLattice, size_t n){
    size_t elementID = blockDim.x*blockIdx.x + threadIdx.x;
    size_t row = elementID / n, column = elementID % n;
    int8_t sum;
    if(elementID < n*n){
        sum = curLattice[n*row + column] + curLattice[n*getIndex(row-1, n) + column] + curLattice[n*getIndex(row+1, n) + column] +
                curLattice[n*row + getIndex(column -1, n)] +curLattice[n*row + getIndex(column+1, n)];
        //calculate sign
        nexLattice[n*row + column] = (sum >= 0) ? 1 : -1;
    }
    //printf("element id: %lu row %lu column %lu value: %d\n", elementID,row, column, nexLattice[n*row + column]);
}

int main(int argc, char *argv[]){
    //row major order will be followed
    int8_t *current_lattice_state, *next_lattice_state, *host_lattice, *temp;
    size_t block_size=BLOCK_SIZE, elementsPerRow=N,seed=69, epochs=5;
    //argument parsing
    switch(argc){
        case 5:
            elementsPerRow = atoi(argv[1]);
            block_size = atoi(argv[2]);
            epochs = atoi(argv[3]);
            seed = atoi(argv[4]);
            break;
        case 4:
            elementsPerRow = atoi(argv[1]);
            block_size = atoi(argv[2]);
            epochs = atoi(argv[3]);
            break;
        case 3:
            elementsPerRow = atoi(argv[1]);
            block_size = atoi(argv[2]);
            break;
        case 2:
            elementsPerRow = atoi(argv[1]);
            break;
    }
    if(block_size > 1024){
        //block too large, aborting
        printf("Error! Each block contains max 1024 threads! Aborting!\n");
        return 1;
    }
    //round up block count
    //assume square lattice
    const size_t block_count = (elementsPerRow*elementsPerRow+block_size-1)/block_size;
    const size_t size = elementsPerRow*elementsPerRow*sizeof(int8_t);
    struct timeval start,stop;
    long secondsElapsed, microsecondsElapsed;
    double time;
    //cuda_hello<<<1,32>>>();
    srand(seed);
    host_lattice = (int8_t*) malloc(size);
    generateRandomLattice(host_lattice, elementsPerRow);
    cudaMalloc(&current_lattice_state, size);
    cudaMalloc(&next_lattice_state, size);
    cudaMemcpy(current_lattice_state, host_lattice, size, cudaMemcpyHostToDevice);
    //printLattice(host_lattice, elementsPerRow);
    printf("AMOGUS\n");
    printf("creating %zu blocks of %zu threads!\n", block_count, block_size);
    gettimeofday(&start, 0);
    for(size_t i = 0; i < epochs; ++i){
        calculateNextLattice<<< block_count, block_size >>>(current_lattice_state, next_lattice_state, elementsPerRow);
        cudaDeviceSynchronize();
        temp = current_lattice_state;
        current_lattice_state = next_lattice_state;
        next_lattice_state = temp;
    }
    cudaMemcpy(host_lattice, current_lattice_state, size, cudaMemcpyDeviceToHost);
    gettimeofday(&stop, 0);
    secondsElapsed = stop.tv_sec - start.tv_sec;
    microsecondsElapsed = stop.tv_usec - start.tv_usec;
    time = secondsElapsed + 1e-6*microsecondsElapsed;
    //printLattice(host_lattice, elementsPerRow);
    printf("total time: %f seconds\n",time);
    cudaFree(current_lattice_state);
    cudaFree(next_lattice_state);
    free(host_lattice);
    return 0;
}
