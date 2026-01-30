// Important Note: In order to keep up with time-keeping, after having referred to "async.cu" in order to further understand the correct & appropriate usage of the 
// CUDA Directives pertaining to Asynchronous Memory Copying and other operations, I enlisted the help of Claude 3.5 Sonnet's Model to modify my Kernel Function 
// (Matrix_Multiplication) and for De-Bugging purposes as well.

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <assert.h>

#define TILE_WIDTH 16

// Error checking function
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0.0;
    
    // Loop over tiles
    for (int m = 0; m < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++m) {
        // Load tile of A into shared memory
        if (Row < N && (m * TILE_WIDTH + tx) < N)
            ds_A[ty][tx] = A[Row * N + m * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0f;
        
        // Load tile of B into shared memory
        if (Col < N && (m * TILE_WIDTH + ty) < N)
            ds_B[ty][tx] = B[(m * TILE_WIDTH + ty) * N + Col];
        else
            ds_B[ty][tx] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        
        __syncthreads();
    }
    
    // Write result
    if (Row < N && Col < N)
        C[Row * N + Col] = Pvalue;
}

int main() {
    const int n = 512;  // Matrix dimension
    const int bytes = n * n * sizeof(float);
    
    // Thread block and grid dimensions
    const dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    const dim3 numBlocks((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH);

    // Host memory allocation
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = i;
            h_B[i * n + j] = j;
            h_C[i * n + j] = 0;
        }
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes));
    checkCuda(cudaMalloc(&d_B, bytes));
    checkCuda(cudaMalloc(&d_C, bytes));

    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // Start timing
    checkCuda(cudaEventRecord(startEvent));

    // Copy matrices to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch tiled kernel
    matrixMultiplyTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Copy result back
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Stop timing
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    
    printf("Approach 2 (p2.cu) - Tiled Shared Memory\n");
    printf("Execution Time: %f ms\n", ms);
    
    // Cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}