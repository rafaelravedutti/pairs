#include "boundary_weights.hpp"
// #include "devices/device.hpp"

// Always include last generated interfaces
#include "last_generated.hpp"
#define CUDA_ASSERT(a) { pairs::cuda_assert((a), __FILE__, __LINE__); }

namespace pairs {

#define REDUCE_BLOCK_SIZE 64

__global__ void reduceBoundaryWeights( real_t *position, int *flags, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax, int *d_weights) {

    __shared__ int red_data[REDUCE_BLOCK_SIZE];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int particle_idx = start + i;

    red_data[tid] = 0;

    if(particle_idx < end) {
        if (!(pairs_cuda_interface::get_flags(flags, i) & (pairs::flags::INFINITE | pairs::flags::GLOBAL))) {

            real_t pos_x = pairs_cuda_interface::get_position(position, particle_idx, 0, particle_capacity);
            real_t pos_y = pairs_cuda_interface::get_position(position, particle_idx, 1, particle_capacity);
            real_t pos_z = pairs_cuda_interface::get_position(position, particle_idx, 2, particle_capacity);

            if( pos_x >= xmin && pos_x < xmax &&
                pos_y >= ymin && pos_y < ymax &&
                pos_z >= zmin && pos_z < zmax) {
                    red_data[tid] = 1;
            }
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
    real_t *position, int *flags, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
    
    if (start==end) return 0;
    const int nblocks = (end - start + (REDUCE_BLOCK_SIZE - 1)) / REDUCE_BLOCK_SIZE;

    int *h_weights = (int *) malloc(nblocks * sizeof(int));
    int *d_weights = (int *) device_alloc(nblocks * sizeof(int));
    int red = 0;

    CUDA_ASSERT(cudaMemset(d_weights, 0, nblocks * sizeof(int)));
    reduceBoundaryWeights<<<nblocks, REDUCE_BLOCK_SIZE>>>(
            position, flags, start, end, particle_capacity,
            xmin, xmax, ymin, ymax, zmin, zmax, d_weights);

    CUDA_ASSERT(cudaPeekAtLastError());
    CUDA_ASSERT(cudaDeviceSynchronize());
    CUDA_ASSERT(cudaMemcpy(h_weights, d_weights, nblocks * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < nblocks; i++) {
        red += h_weights[i];
    }

    return red;
}

void compute_boundary_weights(
    PairsRuntime *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    long unsigned int *comp_weight, long unsigned int *comm_weight) {

    const int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
    const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
    const int nghost = ps->getTrackedVariableAsInteger("nghost");
    auto position_prop = ps->getPropertyByName("position");
    auto flags_prop = ps->getPropertyByName("flags");


    real_t *position_ptr = static_cast<real_t *>(position_prop.getDevicePointer());
    int *flags_ptr = static_cast<int *>(flags_prop.getDevicePointer());

    ps->copyPropertyToDevice(position_prop.getId(), ReadOnly);
    ps->copyPropertyToDevice(flags_prop.getId(), ReadOnly);

    *comp_weight = cuda_compute_boundary_weights(
        position_ptr, flags_ptr, 0, nlocal, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);

    // TODO
    // *comm_weight = cuda_compute_boundary_weights(
    //     position_ptr, nlocal, nlocal + nghost, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);
    *comm_weight = 0;
}

}
