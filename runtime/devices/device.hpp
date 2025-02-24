#include <iostream>
#include <cstddef>

#include "../pairs_common.hpp"

#pragma once

#ifndef PAIRS_TARGET_CUDA
#   define __host__
typedef int cudaError_t;
#else
#include <cuda_runtime.h>
#endif

namespace pairs {

void cuda_assert(cudaError_t err, const char *file, int line);
__host__ void *device_alloc(size_t size);
__host__ void *device_realloc(void *ptr, size_t size);
__host__ void device_free(void *ptr);
__host__ void device_synchronize();
__host__ void copy_to_device(const void *h_ptr, void *d_ptr, size_t count);
__host__ void copy_to_host(const void *d_ptr, void *h_ptr, size_t count);
__host__ void copy_in_device(void *d_ptr1, const void *d_ptr2, size_t count);
__host__ void copy_slice_to_device(const void *h_ptr, void *d_ptr, size_t offset, size_t count);
__host__ void copy_slice_to_host(const void *d_ptr, void *h_ptr, size_t offset, size_t count);
__host__ void copy_static_symbol_to_device(void *h_ptr, const void *d_ptr, size_t count);
__host__ void copy_static_symbol_to_host(void *d_ptr, const void *h_ptr, size_t count);

#ifdef PAIRS_TARGET_OPENMP
#include <omp.h>

inline __host__ int host_atomic_add(int *addr, int val) {
    int result;
    #pragma omp critical
    {
        *addr += val;
        result = *addr;
    }
    return result - val;
}

inline __host__ real_t host_atomic_add(real_t *addr, real_t val) {
    real_t result;
    #pragma omp critical
    {
        *addr += val;
        result = *addr;
    }
    return result - val;
}
#else
inline __host__ int host_atomic_add(int *addr, int val) {
    *addr += val;
    return *addr - val;
}

inline __host__ real_t host_atomic_add(real_t *addr, real_t val) {
    real_t tmp = *addr;
    *addr += val;
    return tmp;
}
#endif

inline __host__ int host_atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    const int add_res = *addr + val;
    if(add_res >= capacity) {
        *resize = add_res;
        return *addr;
    }

    return host_atomic_add(addr, val);
}

#ifdef PAIRS_TARGET_CUDA
inline void cuda_assert(cudaError_t err, const char *file, int line) {
    if(err != cudaSuccess) {
        std::cerr << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}
__device__ double atomicAdd_double(double* address, double val);
__device__ int atomic_add(int *addr, int val);
__device__ real_t atomic_add(real_t *addr, real_t val);
__device__ int atomic_add_resize_check(int *addr, int val, int *resize, int capacity);
#else
int atomic_add(int *addr, int val);
real_t atomic_add(real_t *addr, real_t val);
int atomic_add_resize_check(int *addr, int val, int *resize, int capacity);
#endif
}
