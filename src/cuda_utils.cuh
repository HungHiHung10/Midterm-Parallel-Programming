// Author: Nguyen Huu Khanh Hung - 23120271 (Member 3). Original CUDA utilities, NVIDIA Runtime API.
#pragma once
#include "logging.h"
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <stdexcept>
inline void check_cuda(cudaError_t result, const char* file, int line) {
    if (result != cudaSuccess) {
        log_message(LogLevel::Error) << file << ':' << line << ": "
                                     << cudaGetErrorString(result) << '\n';
        throw std::runtime_error(cudaGetErrorString(result));
    }
}
#define CHECK_CUDA(call) do { check_cuda((call), __FILE__, __LINE__); } while (0)
// Despite the historical name, this helper synchronizes for error diagnosis.
#define CHECK_KERNEL_ASYNC() do { CHECK_CUDA(cudaGetLastError()); \
                                  CHECK_CUDA(cudaDeviceSynchronize()); } while (0)
struct GpuTimer {
    cudaEvent_t start_event{}, stop_event{};
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        auto result = cudaEventCreate(&stop_event);
        if (result != cudaSuccess) {
            cudaEventDestroy(start_event);
            CHECK_CUDA(result);
        }
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;
    ~GpuTimer() { cudaEventDestroy(start_event); cudaEventDestroy(stop_event); }
    void start(cudaStream_t stream = 0) { CHECK_CUDA(cudaEventRecord(start_event, stream)); }
    void stop(cudaStream_t stream = 0) { CHECK_CUDA(cudaEventRecord(stop_event, stream)); }
    float elapsed() {
        float ms = 0;
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};
#endif
