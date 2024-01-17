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

__global__ void calculateNextLattice3(int8_t *curLattice, int8_t *nexLattice,size_t n){
    //this time, each thread has an element in the sub square of the lattice
    size_t blockRow = blockIdx.y*blockDim.y;
    size_t blockCol = blockIdx.x*blockDim.x;
    size_t localRow = threadIdx.y, localCol = threadIdx.x, globalRow, globalCol;
    //populate shared memory with global memory data. Include the nearby cells of the subsquare since they will be needed
    globalCol = blockCol + localCol;
    globalRow = blockRow + localRow;
    //printf("block row %lu, global col %lu,global row: %lu, global col: %lu\n", blockRow,blockCol,globalRow, globalCol);
    //in order to read neighbours, our subsquare must be in the center. So we will need (BLOCK_SIZE+2)^2 total elements
    //in shared memory
    __shared__ int8_t subSquare[BLOCK_SIZE+2][BLOCK_SIZE+2];
    //make sure were within range
    if(globalRow < n && globalCol < n){
        //printf("block row %lu, global col %lu,global row: %lu, global col: %lu\n", blockRow,blockCol,globalRow, globalCol);
        //each thread populates loads its corresponding global memory element to shared memory
        subSquare[localRow+1][localCol+1] = curLattice[n*globalRow + globalCol];
        //load neighboring edge cells
        if(localRow == 0 || globalRow == 0){
            subSquare[localRow][localCol+1] = curLattice[n*getIndex(globalRow-1,n)+globalCol];
        }
        if(localRow == BLOCK_SIZE -1 || globalRow == n-1){
            subSquare[localRow+2][localCol+1] = curLattice[n*getIndex(globalRow+1, n) +globalCol];
        }
        if(localCol == 0 || globalCol == 0){
            subSquare[localRow+1][localCol] = curLattice[n*globalRow + getIndex(globalCol-1, n)];
        }
        if(localCol == BLOCK_SIZE -1 || globalCol == n-1){
            subSquare[localRow+1][localCol+2] = curLattice[n*globalRow + getIndex(globalCol+1, n)];
        }
        //synchronize first to make sure everything is loaded
        __syncthreads();
        //make calculation
        int8_t sum;
        sum = subSquare[localRow+1][localCol+1] + subSquare[localRow+1][localCol] + subSquare[localRow+1][localCol+2]
            + subSquare[localRow][localCol+1] + subSquare[localRow+2][localCol+1];
        nexLattice[n*globalRow + globalCol] = (sum > 0) ? 1 : -1;
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
    const size_t size = elementsPerRow*elementsPerRow*sizeof(int8_t);
    //round up block count so everything can fit. Every thread this time handles an entire column of the lattice
    const size_t block_count = (elementsPerRow+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 grid(block_count,block_count);
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);
    srand(seed);
    host_lattice = (int8_t*) malloc(size);
    generateRandomLattice(host_lattice, elementsPerRow);
    //load it to GPU
    cudaMalloc(&current_lattice_state, size);
    cudaMalloc(&next_lattice_state,size);
    cudaMemcpy(current_lattice_state, host_lattice,size,cudaMemcpyHostToDevice);
    struct timeval start,stop;
    long secondsElapsed, microsecondsElapsed;
    double totalTime;
    printf("AMOGUS\n");
    printf("creating %zu blocks containing %d threads each!\n", block_count*block_count, BLOCK_SIZE*BLOCK_SIZE);
    gettimeofday(&start, 0);
    for(size_t i = 0; i < epochs; ++i){
        calculateNextLattice3<<< grid, blocks >>>(current_lattice_state, next_lattice_state, elementsPerRow);
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
    totalTime = secondsElapsed + 1e-6*microsecondsElapsed;
    //printLattice(host_lattice, elementsPerRow);
    printf("total time: %f seconds.\n", totalTime);
    return 0;
}
