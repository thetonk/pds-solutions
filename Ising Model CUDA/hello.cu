#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU, block %d and thread %d!\n", blockIdx.x, threadIdx.x);
}

int main() {
    cudaDeviceProp p;
    cuda_hello<<<1,200>>>();
    cudaDeviceSynchronize();
    cudaGetDeviceProperties(&p,0);
    int value = p.major*10 + p.minor;
    printf("NVIDIA arch=compute_%d, code=sm_%d\n", value,value);
    return 0;
}
