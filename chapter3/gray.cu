#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#include <cuda_runtime.h>

const size_t CHANNELS = 3;
const int BLUR_SIZE = 3;


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

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int offset = row*w + col;
        int pixel_num = 0;
        int pixel_val = 0;

        for (int blur_row = -BLUR_SIZE; blur_row <= BLUR_SIZE; blur_row++) {
            for (int blur_col = -BLUR_SIZE; blur_col<= BLUR_SIZE; blur_col++) {
                int pixel_row = row + blur_row;
                int pixel_col = col + blur_col;
                int pixel_offset = pixel_row*w + pixel_col;
                if (pixel_col >= 0 && pixel_col < w && pixel_row >= 0 && pixel_row < row) {
                    pixel_val += in[pixel_offset];
                    pixel_val++;
                }
          }
        }
        int average_pixel = pixel_val/pixel_num;
        out[offset] = average_pixel;
    }
}

__global__ void colorToGrayscale(unsigned char *Pout, unsigned char * Pin, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        int offset = row*width + col;

        int rgbOffset = offset * CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset];
        unsigned char b = Pin[rgbOffset];

        Pout[offset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
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
    int n = 10000;
    int m = 10000;
    int k = 10000;
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
    dim3 dimGrid(ceil(k/16.0), ceil(n/16.0), 1);
    dim3 dimBlock(16, 16, 1);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    matrixMult<<<dimGrid, dimBlock>>>(A_D, B_D, C_D, n, m, k);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC,&end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("(GPU) Time elapsed is: %.3f\n", elapsed);

    struct timespec start_h, end_h;
    clock_gettime(CLOCK_MONOTONIC, &start_h);
    hostMatrixMult(A, B, C, n, m, k);
    clock_gettime(CLOCK_MONOTONIC,&end_h);

    double elapsed_h = (end_h.tv_sec - start_h.tv_sec) + (end_h.tv_nsec - start_h.tv_nsec) / 1e9;

    printf("(CPU): Time elapsed is: %.3f\n", elapsed_h);
    float *C_H = (float*)malloc(C_size);
    cudaMemcpy(C_H, C_D, C_size, cudaMemcpyDeviceToHost);

    compareMatrix(C, C_H, n, k);

    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(C_D);

    free(C);
    free(C_H);
    free(B);
    free(A);
}

