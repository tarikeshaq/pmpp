#include <cstdlib>
#include <cstring>
#include <stdio.h>

#include <cuda_runtime.h>



void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}


__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void gpuVecAdd(float* A_h, float* B_h, float* C_h, int n) {
    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc((void**)&A_d, sizeof (float) * n);
    cudaMalloc((void**)&B_d, sizeof (float) * n);
    cudaMalloc((void**)&C_d, sizeof (float) * n);

    cudaMemcpy(A_d, A_h, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, sizeof(float) * n, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}



int main() {
    int n = 2000;
    float* A_h = (float* )malloc(sizeof (float) * n);
    float* B_h = (float* )malloc(sizeof (float) * n);
    float* C_h = (float* )malloc(sizeof (float) * n);
    for (int i = 0; i < n; i++) {
        A_h[i] = 1;
        B_h[i] = 1;
    }

   gpuVecAdd(A_h, B_h, C_h, n);
   //vecAdd(A_h, B_h, C_h,n);

    printf("elem is %f\n", C_h[1000000]);

    free(C_h);
    free(B_h);
    free(A_h);
    return 0;
}
