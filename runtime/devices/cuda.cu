#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

#define CUDA_ASSERT(a) { pairs::cuda_assert((a), __FILE__, __LINE__); }
#define REDUCE_BLOCK_SIZE   64

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

int cuda_compute_boundary_weights(
    real_t *position, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {

    const int nblocks = (end - start + (REDUCE_BLOCK_SIZE - 1)) / REDUCE_BLOCK_SIZE;
    int *h_weights = (int *) malloc(nblocks * sizeof(int));
    int *d_weights = (int *) device_alloc(nblocks * sizeof(int));
    int red = 0;

    CUDA_ASSERT(cudaMemset(d_weights, 0, nblocks * sizeof(int)));

    reduceBoundaryWeights(
        position, start, particle_capacity, 
    )
    CUDA_ASSERT(cudaPeekAtLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaMemcpy(h_weights, d_weights, nblocks * sizeof(int), cudaMemcpyDeviceToHost));

    reduceBoundaryWeights<nblocks, REDUCE_BLOCK_SIZE>();

    for(int i = 0; i < nblocks; i++) {
        red += h_weights[i];
    }

    return red;
}

void __device__ reduceBoundaryWeights() {
    __shared__ int red_data[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    real_t pos_x = pairs_cuda_interface::get_position(position, start + i, 0, particle_capacity);
    real_t pos_y = pairs_cuda_interface::get_position(position, start + i, 1, particle_capacity);
    real_t pos_z = pairs_cuda_interface::get_position(position, start + i, 2, particle_capacity);

    red_data[tid] = 0;

    if(i < n) {
        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                red_data[tid] = 1;
        }
    }

    __syncthreads();

    int s = blockDim.x >> 1;
    while(s > 0) {
        if(tid < s) {
            red_data[tid] += red_data[tid + s];
        }

        __syncthreads();
        s >>= 1;
    }

    if(tid == 0) {
        d_weights[blockIdx.x] = red_data[0];
    }
}

}
