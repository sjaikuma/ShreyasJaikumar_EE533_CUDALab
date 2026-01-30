// Important Note: In order to keep up with time-keeping, after having referred to "async.cu" in order to further understand the correct & appropriate usage of the 
// CUDA Directives pertaining to Asynchronous Memory Copying and other operations, I enlisted the help of Claude 3.5 Sonnet's Model to modify my Kernel Function 
// (Matrix_Multiplication) and for De-Bugging purposes as well.

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <assert.h>

// Error checking function
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Simple kernel for matrix multiplication
__global__ void Matrix_Multiplication(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    const int n = 512;  // Matrix dimension
    const int bytes = n * n * sizeof(float);
    
    // Thread block and grid dimensions
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((n + 15) / 16, (n + 15) / 16);  // 64x64 blocks

    // Host memory allocation (regular malloc)
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

    // Copy matrices to device (synchronous)
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    // Launch kernel (single launch, no streams)
    Matrix_Multiplication<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Copy result back (synchronous)
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Stop timing
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    
    printf("Approach 1 (p1.cu) - Naive Matrix Multiplication\n");
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