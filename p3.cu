// Important Note: In order to keep up with time-keeping, after having referred to "async.cu" in order to further understand the correct & appropriate usage of the 
// CUDA Directives pertaining to Asynchronous Memory Copying and other operations, I enlisted the help of Claude 3.5 Sonnet's Model to modify my Kernel Function 
// (Matrix_Multiplication) and for De-Bugging purposes as well.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>

// Error checking function for CUDA
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Error checking function for cuBLAS
inline cublasStatus_t checkCublas(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d\n", result);
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
    return result;
}

int main() {
    const int n = 512;  // Matrix dimension
    const int bytes = n * n * sizeof(float);
    
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

    // Copy matrices to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle));

    // Create events for timing
    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    // cuBLAS parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Start timing
    checkCuda(cudaEventRecord(startEvent));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    // Note: cuBLAS uses column-major format, so we compute B*A to get A*B in row-major
    checkCublas(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n, n, n,
                            &alpha,
                            d_B, n,
                            d_A, n,
                            &beta,
                            d_C, n));

    // Stop timing
    checkCuda(cudaEventRecord(stopEvent));
    checkCuda(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time
    float ms;
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));

    // Copy result back
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("cuBLAS Matrix Multiplication\n");
    printf("Execution Time: %f ms\n", ms);
    printf("C[451][451] = %f\n", h_C[451 * n + 451]);

    // Cleanup
    checkCuda(cudaEventDestroy(startEvent));
    checkCuda(cudaEventDestroy(stopEvent));
    checkCublas(cublasDestroy(handle));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}