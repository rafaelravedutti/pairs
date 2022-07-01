#include <cuda_runtime.h>
#include <iostream>
//---
#include "../pairs.hpp"

#pragma once

#define CUDA_ASSERT(a) { pairs::cuda_assert((a), __FILE__, __LINE__); }

namespace pairs {

inline void cuda_assert(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess) {
        std::cerr << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

__host__ void *device_alloc(size_t size) {
    void *ptr;
    CUDA_ASSERT(cudaMalloc(&ptr, size));
    return ptr;
}

__host__ void *device_realloc(void *ptr, size_t size) {
    void *new_ptr;
    CUDA_ASSERT(cudaFree(ptr));
    CUDA_ASSERT(cudaMalloc(&new_ptr, size));
    return new_ptr;
}

__host__ void copy_to_device(const void *h_ptr, void *d_ptr, size_t count) {
    CUDA_ASSERT(cudaMemcpy(d_ptr, h_ptr, count, cudaMemcpyHostToDevice));
}

__host__ void copy_to_host(const void *d_ptr, void *h_ptr, size_t count) {
    CUDA_ASSERT(cudaMemcpy(h_ptr, d_ptr, count, cudaMemcpyDeviceToHost));
}

inline __device__ int atomic_add(int *addr, int val) { return atomicAdd(addr, val); }
inline __device__ int atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    const int add_res = *addr + val;
    if(add_res >= capacity) {
        *resize = add_res;
        return *addr;
    }

    return atomic_add(addr, val);
}

}
