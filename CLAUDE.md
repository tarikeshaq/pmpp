# CLAUDE.md - Project Guide for AI Assistants

## Project Overview

This is an educational CUDA programming repository implementing examples from **"Programming Massively Parallel Processors"** (PMPP) by David Kirk and Wen-mei Hwu. The project demonstrates GPU programming concepts through progressively complex examples using NVIDIA CUDA.

**Author:** Tarik Eshaq
**License:** MIT
**Status:** Active development (6 chapters, with chapter 6 as boilerplate)

## Repository Structure

```
pmpp/
├── Makefile              # Build configuration for CUDA compilation
├── LICENSE               # MIT License
├── CLAUDE.md             # This file
├── chapter2/             # Vector Addition (Basic CUDA Kernels)
│   └── vectorAdd.cu
├── chapter3/             # Image Processing (Grayscale & Blur)
│   └── gray.cu
├── chapter4/             # Device Properties (Hardware Querying)
│   └── prop.cu
├── chapter5/             # Memory Optimization (Shared Memory & cuBLAS)
│   └── mem.cu
├── chapter6/             # Performance Analysis (Boilerplate)
│   └── perf.cu
└── notes/                # Learning Documentation
    ├── Chapter1.md       # Introduction to Heterogeneous Computing
    └── Chapter2.md       # Heterogeneous Data Parallel Computing
```

## Technology Stack

- **Language:** CUDA C/C++ (`.cu` files)
- **Compiler:** NVCC (NVIDIA CUDA Compiler)
- **Build System:** GNU Make
- **GPU Libraries:** CUDA Runtime API, cuBLAS v2 (Chapter 5)

## Build Commands

```bash
make chapter3    # Compile and run grayscale/blur image processing
make chapter4    # Compile and run device properties query
make chapter5    # Compile and run memory optimization demos (requires cuBLAS)
make chapter6    # Compile and run performance analysis
make clean       # Remove all compiled binaries
```

**Note:** Chapter 5 requires cuBLAS library linking (`-lcublas`).

## Code Conventions

### Naming Patterns

- **Host pointers:** `*_h` suffix (e.g., `A_h`, `B_h`, `C_h`)
- **Device pointers:** `*_d` suffix (e.g., `A_d`, `B_d`, `C_d`)
- **Kernel functions:** `kernelName` or `kernelNameKernel`
- **Host helper functions:** Descriptive verbs (e.g., `gpuVecAdd()`, `cpuMatMul()`)

### Memory Layout

- Row-major matrix storage: `A[row * width + col]`
- Linear indexing for 2D data: `offset = row * width + col`

### Thread Indexing Patterns

**1D Kernels (Vector operations):**
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

**2D Kernels (Matrix operations):**
```cuda
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### Block/Grid Configuration

- **1D operations:** 256 threads per block
- **2D operations:** 16x16 thread blocks (`TILE_WIDTH = 16`)
- **Grid calculation:** `ceil(dimension / block_dimension)`

### Memory Management Pattern

All CUDA programs follow this pattern:
```cuda
// 1. Allocate device memory
cudaMalloc((void**)&d_ptr, size);

// 2. Transfer host → device
cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);

// 3. Execute kernel
kernel<<<grid, block>>>(d_ptr);

// 4. Transfer device → host
cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost);

// 5. Cleanup
cudaFree(d_ptr);
```

## Chapter Contents

| Chapter | File | Focus | Key Concepts |
|---------|------|-------|--------------|
| 2 | `vectorAdd.cu` | Basic GPU Computing | Kernel basics, memory transfer, grid/block config |
| 3 | `gray.cu` | Image Processing | 2D kernels, matrix multiplication, grayscale, blur |
| 4 | `prop.cu` | Hardware Properties | Device enumeration, capability querying |
| 5 | `mem.cu` | Memory Optimization | Shared memory, tiled matmul, cuBLAS integration |
| 6 | `perf.cu` | Performance (WIP) | Boilerplate for performance analysis |

## Important Constants

```cuda
#define TILE_WIDTH 16      // Chapter 5: Tiled matrix multiplication
#define BLUR_SIZE 3        // Chapter 3: Image blur kernel window
#define CHANNELS 3         // Chapter 3: RGB image channels
```

## Validation Approach

Each program includes built-in validation:
1. **CPU reference implementation** for comparison
2. **`compareMatrix()` functions** verify GPU results match CPU
3. **Relative tolerance checking** for floating-point precision:
   ```cuda
   float threshold = fmax(fabs(A[i]), fabs(B[i])) * 1e-5;
   ```
4. **Timing comparison** between GPU and CPU execution

## Synchronization

- `cudaDeviceSynchronize()` - GPU-CPU synchronization (used before timing)
- `__syncthreads()` - Intra-block barrier (used with shared memory)

## Dependencies

**Required:**
- NVIDIA CUDA Toolkit (with NVCC)
- NVIDIA GPU with CUDA support
- GNU Make
- C/C++ standard libraries

**Optional:**
- cuBLAS library (required for Chapter 5)

## Common Patterns in This Codebase

1. **Boundary checking in kernels:** `if (i < n)` to prevent out-of-bounds access
2. **Float literals:** Use `f` suffix (e.g., `1.0f`, `0.0f`)
3. **Size calculations:** `sizeof(float) * n` for explicit sizing
4. **Loop unrolling:** `#pragma unroll` for optimization hints (Chapter 5)
5. **Bank conflict avoidance:** Shared memory padding `[TILE_WIDTH + 1]`

## Current Limitations

- No CUDA error checking (assumes successful operations)
- No formal test suite (validation is inline)
- Chapter 6 is boilerplate only

## Development Tips

When adding new chapters or examples:
1. Follow the existing `chapter*/filename.cu` directory structure
2. Include both CPU and GPU implementations for comparison
3. Add timing with `clock_gettime(CLOCK_MONOTONIC)` for microsecond precision
4. Include a validation function to compare outputs
5. Update the Makefile with appropriate targets
6. Use the established naming conventions (`_h`, `_d` suffixes)
