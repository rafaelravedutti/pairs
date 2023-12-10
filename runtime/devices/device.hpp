#include <iostream>
#include <cstddef>

#pragma once

#ifndef PAIRS_TARGET_CUDA
#   define __host__
typedef int cudaError_t;
#endif

namespace pairs {

void cuda_assert(cudaError_t err, const char *file, int line);
__host__ void *device_alloc(size_t size);
__host__ void *device_realloc(void *ptr, size_t size);
__host__ void device_free(void *ptr);
__host__ void device_synchronize();
__host__ void copy_to_device(const void *h_ptr, void *d_ptr, size_t count);
__host__ void copy_to_host(const void *d_ptr, void *h_ptr, size_t count);
__host__ void copy_slice_to_device(const void *h_ptr, void *d_ptr, size_t offset, size_t count);
__host__ void copy_slice_to_host(const void *d_ptr, void *h_ptr, size_t offset, size_t count);
__host__ void copy_static_symbol_to_device(void *h_ptr, const void *d_ptr, size_t count);
__host__ void copy_static_symbol_to_host(void *d_ptr, const void *h_ptr, size_t count);

inline __host__ int host_atomic_add(int *addr, int val) {
    *addr += val;
    return *addr - val;
}

inline __host__ int host_atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    const int add_res = *addr + val;
    if(add_res >= capacity) {
        *resize = add_res;
        return *addr;
    }

    return host_atomic_add(addr, val);
}

#ifdef PAIRS_TARGET_CUDA
__device__ int atomic_add(int *addr, int val) { return atomicAdd(addr, val); }
__device__ int atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    const int add_res = *addr + val;
    if(add_res >= capacity) {
        *resize = add_res;
        return *addr;
    }

    return atomic_add(addr, val);
}
#else
inline int atomic_add(int *addr, int val) { return host_atomic_add(addr, val); }
inline int atomic_add_resize_check(int *addr, int val, int *resize, int capacity) {
    return host_atomic_add_resize_check(addr, val, resize, capacity);
}
#endif

}
