#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include <cuda_runtime.h>


int main() {
    printf("Printing Stats for all CUDA devices\n");

    int devCount;
    cudaGetDeviceCount(&devCount);


    printf("The number of devices is: %d\n", devCount);
    
    cudaDeviceProp devProp;
    for (unsigned int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        printf("Max number of threads per block: %d\n", devProp.maxThreadsPerBlock);
        printf("The number of multiprocessors (SMs) is: %d\n", devProp.multiProcessorCount);
        printf("Max Thread dimentions are: x=%d, y=%d, z=%d\n", devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("Max Grid dimentions are: x=%d, y=%d, z=%d\n", devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("The max number of registers per SM = %d\n", devProp.regsPerMultiprocessor);
        printf("The size of a warp is %d threads\n", devProp.warpSize);
    }

    return 0;
}

