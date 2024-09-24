#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "boundary_weights.hpp"
#include "pairs.hpp"
#include "pairs_common.hpp"

// Always include last generated interfaces
#include "last_generated.hpp"

#ifdef PAIRS_TARGET_CUDA

#define REDUCE_BLOCK_SIZE   64

void __global__ reduceBoundaryWeights(
    real_t *position, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax, int *d_weights) {

    __shared__ int red_data[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int particle_idx = start + i;

    red_data[tid] = 0;

    if(particle_idx < end) {
        real_t pos_x = pairs_cuda_interface::get_position(position, particle_idx, 0, particle_capacity);
        real_t pos_y = pairs_cuda_interface::get_position(position, particle_idx, 1, particle_capacity);
        real_t pos_z = pairs_cuda_interface::get_position(position, particle_idx, 2, particle_capacity);

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

int cuda_compute_boundary_weights(
    real_t *position, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {

    const int nblocks = (end - start + (REDUCE_BLOCK_SIZE - 1)) / REDUCE_BLOCK_SIZE;
    int *h_weights = (int *) malloc(nblocks * sizeof(int));
    int *d_weights = (int *) device_alloc(nblocks * sizeof(int));
    int red = 0;

    CUDA_ASSERT(cudaMemset(d_weights, 0, nblocks * sizeof(int)));

    reduceBoundaryWeights<<<nblocks, REDUCE_BLOCK_SIZE>>>(
            position, start, end, particle_capacity,
            xmin, xmax, ymin, ymax, zmin, zmax, d_weights);

    CUDA_ASSERT(cudaPeekAtLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaMemcpy(h_weights, d_weights, nblocks * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < nblocks; i++) {
        red += h_weights[i];
    }

    return red;
}
#endif

namespace pairs {

void compute_boundary_weights(
    PairsRuntime *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    long unsigned int *comp_weight, long unsigned int *comm_weight) {

    const int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
    const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
    const int nghost = ps->getTrackedVariableAsInteger("nghost");
    auto position_prop = ps->getPropertyByName("position");

    #ifndef PAIRS_TARGET_CUDA
    real_t *position_ptr = static_cast<real_t *>(position_prop.getHostPointer());

    *comp_weight = 0;
    *comm_weight = 0;

    for(int i = 0; i < nlocal; i++) {
        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                (*comp_weight)++;
        }
    }

    for(int i = nlocal; i < nlocal + nghost; i++) {
        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                (*comm_weight)++;
        }
    }
    // std::cout << "comp_weight = " << (*comp_weight) << ", comm_weight = " << (*comm_weight) << std::endl;
    #else
    real_t *position_ptr = static_cast<real_t *>(position_prop.getDevicePointer());

    ps->copyPropertyToDevice(position_prop, ReadOnly);

    *comp_weight = cuda_compute_boundary_weights(
        position_ptr, 0, nlocal, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);

    *comm_weight = cuda_compute_boundary_weights(
        position_ptr, nlocal, nlocal + nghost, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);
    #endif
}

}
