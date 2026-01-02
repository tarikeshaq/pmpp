#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16

__global__ void matrixMult(float *A, float *B, float *C, int n, int m, int k) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < k && row < n) {
        float res = 0;
        for (int i = 0; i < m; i++) {
            float multi = A[row*m + i] * B[i*k + col];
            res += multi;
        }
        C[row*k + col] = res;
    }
}

__global__ void tiledMatrixMult(float *M, float *N, float *P, int w) { // Assumes a square matrix and a width multiple of the tile width for simplicity
   __shared__ float Mds[TILE_WIDTH][TILE_WIDTH + 1]; 
   __shared__ float Nds[TILE_WIDTH][TILE_WIDTH + 1]; 
   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
   int row = by*TILE_WIDTH + ty;
   int col = bx*TILE_WIDTH + tx;

   float pVal = 0;

   #pragma unroll
   for (int ph = 0; ph < w/TILE_WIDTH; ph++) {
        // First we do a load phase, where we collaborate to load
        // the global memory into the shared memory.
        // Each thread with index row,col will load A_row_{ph*TILE_WIDTH + tx}
        // and B_{ph*TILE_WIDTH + ty}_col into Mds[tx][ty] and Nds[tx][ty] respectively
        if (row < w && (ph*TILE_WIDTH + tx) < w) {
           Mds[ty][tx] = M[row*w + (ph*TILE_WIDTH + tx)];
        } else Mds[ty][tx] = 0.0f;

        if ((ph*TILE_WIDTH + ty) < w && col < w) {
           Nds[tx][ty] = N[(ph*TILE_WIDTH + ty)*w + col];
        } else Nds[tx][ty] = 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++) {
            pVal += (Mds[ty][k] * Nds[tx][k]);
        }
        __syncthreads();
    }
    if (col < w && row < w) {
        P[row*w + col] = pVal;
    }
}

void cuBlas_mult(float *A, float *B, float *C, int n, int m, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    // Row-major C = A × B is computed as column-major C^T = B^T × A^T
    // Where A is n×m, B is m×k, C is n×k in row-major storage
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                k, n, m,              // dimensions: k×n result from k×m × m×n
                &alpha,
                B, k,                 // B^T in column-major view, leading dim k
                A, m,                 // A^T in column-major view, leading dim m
                &beta,
                C, k);                // C^T in column-major view, leading dim k

    cublasDestroy(handle);
}

void populateMatrix(float* A, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A[i*m + j] = (float)(i*m + j);
        }
    }
}


void printMatrix(float *A, int n, int m) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("[");
        for (int j = 0; j < m; j++) {
            printf("%f,", A[i*m + j]);            
        }
       printf("]\n");
    }
    printf("]\n");
}


void hostMatrixMult(float *A, float *B, float *C, int n, int m, int k) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            float val = 0;
            for (int l = 0; l < m; l++) {
                val += (A[i*m + l] * B[l*k + j]);
            }
            C[i*k + j] = val;
        }
    }
}

void compareMatrix(float *A, float *B, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float diff = abs(A[i*m + j] - B[i*m + j]);

            float threshold = fmax(fabs(A[i*m + j]), fabs(B[i*m + j])) * 1e-5;
            if (diff > fmax(threshold, 1e-5)) {
                printf("Found mismatch at index %d,%d, diff is %.2f\n", i, j, diff);
            }
        }
    }
}

int main(int argv, char** argc) {
    int n = 15000;
    int m = 15000;
    int k = 15000;
    size_t A_size = n * m * sizeof(float);
    size_t B_size = m * k * sizeof(float);
    size_t C_size = n * k * sizeof(float);
    float *A = (float*)malloc(A_size);
    float *B = (float*)malloc(B_size);
    float *C = (float*)malloc(C_size);
    populateMatrix(A, n, m);
    populateMatrix(B, m, k);
    float *A_D, *B_D, *C_D;
    cudaMalloc((void**)&A_D, A_size);
    cudaMalloc((void**)&B_D, B_size);
    cudaMalloc((void**)&C_D, C_size);

    cudaMemcpy(A_D, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_D, B, B_size, cudaMemcpyHostToDevice);
    float f_tile_width = (float)TILE_WIDTH;
    dim3 dimGrid(ceil(k/f_tile_width), ceil(n/f_tile_width), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixMult<<<dimGrid, dimBlock>>>(A_D, B_D, C_D, n, m, k);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("(GPU) Time elapsed (Untiled) is: %.3f\n", elapsed);

    struct timespec start_tiled, end_tiled;
    clock_gettime(CLOCK_MONOTONIC, &start_tiled);
    tiledMatrixMult<<<dimGrid, dimBlock>>>(A_D, B_D, C_D, n);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&end_tiled);

    double elapsed_tiled = (end_tiled.tv_sec - start_tiled.tv_sec) + (end_tiled.tv_nsec - start_tiled.tv_nsec) / 1e9;

    printf("(GPU) Time elapsed (tiled) is: %.3f\n", elapsed_tiled);

    struct timespec start_cublas, end_cublas;
    clock_gettime(CLOCK_MONOTONIC, &start_cublas);
    cuBlas_mult(A_D, B_D, C_D, n, m, k);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end_cublas);

    double elapsed_cublas = (end_cublas.tv_sec - start_cublas.tv_sec) + (end_cublas.tv_nsec - start_cublas.tv_nsec) / 1e9;

    printf("(GPU) Time elapsed (cuBLAS) is: %.3f\n", elapsed_cublas);

    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(C_D);

    free(C);
    free(B);
    free(A);
}



