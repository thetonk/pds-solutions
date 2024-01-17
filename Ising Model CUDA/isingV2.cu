#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

#define BLOCK_SIZE 5
#define N 10

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

__global__ void calculateNextLattice(int8_t *curLattice, int8_t *nexLattice, size_t n){
    size_t column = blockDim.x*blockIdx.x + threadIdx.x;
    int8_t sum;
    if(column < n){
        //thread is useful, make calculations
        for(size_t row = 0; row < n; ++row){
            sum = curLattice[n*row + column] + curLattice[n*getIndex(row-1, n) + column] + curLattice[n*getIndex(row+1, n) + column] +
                curLattice[n*row + getIndex(column -1, n)] +curLattice[n*row + getIndex(column+1, n)];
            //calculate sign
            nexLattice[n*row + column] = (sum >= 0) ? 1 : -1;
        }
        //printf("element id: %lu\n", column);
    }
}

//for some reason is slower (not anymore)
__global__ void calculateNextLattice2(int8_t *curLattice, int8_t *nexLattice,size_t n, size_t blockSize){
    //this time, one thread has to work in one subsquare of lattice
    size_t blockRow = blockIdx.y*blockSize*n;
    size_t blockCol = blockIdx.x*blockSize;
    size_t rowIterations = n - blockRow/n < blockSize ? n-blockRow/n : blockSize;
    size_t colIterations = n - blockCol < blockSize ? n- blockCol : blockSize;
    //printf("row iterations %lu col iterations %lu\n",rowIterations,colIterations);
    size_t row,column;
    int8_t sum;
    for(size_t local_row = blockRow; local_row < blockRow +n*rowIterations; local_row+=n){
        for(size_t local_column = blockCol; local_column < blockCol + colIterations; ++local_column){
            //elementID = local_row + local_column;
            //elementID = local_row + local_column;
            row = local_row / n;
            column = local_column;
            //printf("block row %lu block col %lu itempos %lu global row %lu global col %lu\n", blockRow,blockCol,elementID,row,column);
            sum = curLattice[n*row + column] + curLattice[n*getIndex(row-1, n) + column] + curLattice[n*getIndex(row+1, n) + column] +
                curLattice[n*row + getIndex(column -1, n)] +curLattice[n*row + getIndex(column+1, n)];
            //calculate sign
            nexLattice[n*row + column] = (sum >= 0) ? 1 : -1;
        }
    }
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
    //if(block_size > 1024){
        //block too large, aborting
        //printf("Error! Each block contains max 1024 threads! Aborting!\n");
        //return 1;
    //}
    //assume square lattice
    const size_t size = elementsPerRow*elementsPerRow*sizeof(int8_t);
    //round up block count so everything can fit. Every thread this time handles an entire column of the lattice
    const size_t block_count = (elementsPerRow+block_size-1)/block_size;
    srand(seed);
    host_lattice = (int8_t*) malloc(size);
    generateRandomLattice(host_lattice, elementsPerRow);
    //load it to GPU
    cudaMalloc(&current_lattice_state, size);
    cudaMalloc(&next_lattice_state,size);
    cudaMemcpy(current_lattice_state, host_lattice,size,cudaMemcpyHostToDevice);
    dim3 grid(block_count,block_count);
    struct timeval start,stop;
    long secondsElapsed, microsecondsElapsed;
    double time;
    printf("AMOGUS\n");
    //printf("creating %zu blocks of %zu threads!\n", block_count, block_size);
    printf("created %zu blocks containing %zu elements each!\n", block_count*block_count, block_size*block_size);
    gettimeofday(&start, 0);
    for(size_t i = 0; i < epochs; ++i){
        //calculateNextLattice<<<block_count, block_size>>>(current_lattice_state, next_lattice_state, elementsPerRow);
        calculateNextLattice2<<<grid,1>>>(current_lattice_state, next_lattice_state, elementsPerRow, block_size);
        //cudaDeviceSynchronize();
        //ping pong
        temp = current_lattice_state;
        current_lattice_state = next_lattice_state;
        next_lattice_state = temp;
    }
    cudaMemcpy(host_lattice, current_lattice_state, size, cudaMemcpyDeviceToHost);
    gettimeofday(&stop, 0);
    microsecondsElapsed = stop.tv_usec - start.tv_usec;
    secondsElapsed = stop.tv_sec - start.tv_sec;
    time = secondsElapsed + 1e-6*microsecondsElapsed;
    //printLattice(host_lattice, elementsPerRow);
    printf("time elapsed: %f seconds.\n",time);
    cudaFree(next_lattice_state);
    cudaFree(current_lattice_state);
    free(host_lattice);
    return 0;
}
