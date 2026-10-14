#pragma once
#include <cstdio>
#include <cstdlib>

// ============================================================================
// CSC14120 - Parallel Programming | Final Project: Panorama Image Stitching
// CUDA Utilities & Error Checking Macros
// Author: Group XX (Lead: Member 3)
// ============================================================================

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
{                                                                             \
    const cudaError_t error = call;                                           \
    if (error != cudaSuccess) {                                               \
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n",         \
                __FILE__, __LINE__, error, cudaGetErrorString(error));        \
        exit(EXIT_FAILURE);                                                   \
    }                                                                         \
}

#define CHECK_KERNEL_ASYNC()                                                  \
{                                                                             \
    cudaError_t errSync = cudaGetLastError();                                 \
    cudaError_t errAsync = cudaDeviceSynchronize();                           \
    if (errSync != cudaSuccess) {                                             \
        fprintf(stderr, "Sync kernel error: %s\n", cudaGetErrorString(errSync)); \
    }                                                                         \
    if (errAsync != cudaSuccess) {                                            \
        fprintf(stderr, "Async kernel error: %s\n", cudaGetErrorString(errAsync)); \
    }                                                                         \
}

struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
    }

    float elapsed() {
        float ms = 0.0f;
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

#endif // __CUDACC__
