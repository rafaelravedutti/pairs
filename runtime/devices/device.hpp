#include <iostream>
#include <cstddef>

#pragma once

#ifndef PAIRS_TARGET_CUDA
#   define __host__
#   define __device__
typedef int cudaError_t;
#endif

namespace pairs {

void cuda_assert(cudaError_t err, const char *file, int line);
__host__ void *device_alloc(size_t size);
__host__ void *device_realloc(void *ptr, size_t size);
__host__ void copy_to_device(const void *h_ptr, void *d_ptr, size_t count);
__host__ void copy_to_host(const void *d_ptr, void *h_ptr, size_t count);
__host__ void copy_static_symbol_to_device(void *h_ptr, const void *d_ptr, size_t count);
__host__ void copy_static_symbol_to_host(void *d_ptr, const void *h_ptr, size_t count);
__device__ int atomic_add(int *addr, int val);
__device__ int atomic_add_resize_check(int *addr, int val, int *resize, int capacity);

}
