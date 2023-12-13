#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

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

__host__ void device_free(void *ptr) {
    CUDA_ASSERT(cudaFree(ptr));
}

__host__ void device_synchronize() {
    CUDA_ASSERT(cudaPeekAtLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
}

__host__ void copy_to_device(const void *h_ptr, void *d_ptr, size_t count) {
    CUDA_ASSERT(cudaMemcpy(d_ptr, h_ptr, count, cudaMemcpyHostToDevice));
}

__host__ void copy_to_host(const void *d_ptr, void *h_ptr, size_t count) {
    CUDA_ASSERT(cudaMemcpy(h_ptr, d_ptr, count, cudaMemcpyDeviceToHost));
}

__host__ void copy_in_device(void *d_ptr1, const void *d_ptr2, size_t count) {
    #ifdef ENABLE_CUDA_AWARE_MPI
    CUDA_ASSERT(cudaMemcpy(d_ptr1, d_ptr2, count, cudaMemcpyDeviceToDevice));
    #else
    std::memcpy(d_ptr1, d_ptr2, count);
    #endif
}

__host__ void copy_slice_to_device(const void *h_ptr, void *d_ptr, size_t offset, size_t count) {
    void *d_ptr_start = ((char *) d_ptr) + offset;
    void *h_ptr_start = ((char *) h_ptr) + offset;
    CUDA_ASSERT(cudaMemcpy(d_ptr_start, h_ptr_start, count, cudaMemcpyHostToDevice));
}

__host__ void copy_slice_to_host(const void *d_ptr, void *h_ptr, size_t offset, size_t count) {
    void *d_ptr_start = ((char *) d_ptr) + offset;
    void *h_ptr_start = ((char *) h_ptr) + offset;
    CUDA_ASSERT(cudaMemcpy(h_ptr_start, d_ptr_start, count, cudaMemcpyDeviceToHost));
}

__host__ void copy_static_symbol_to_device(void *h_ptr, const void *d_ptr, size_t count) {
    CUDA_ASSERT(cudaMemcpyToSymbol(d_ptr, h_ptr, count));
}

__host__ void copy_static_symbol_to_host(void *d_ptr, const void *h_ptr, size_t count) {
    //CUDA_ASSERT(cudaMemcpyFromSymbol(h_ptr, d_ptr, count));
}

}
