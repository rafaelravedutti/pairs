//---
#include "pairs.hpp"
#include "domain_partitioning.hpp"

#pragma once

namespace pairs {

template <int ndims>
class NoPartitioning : DimensionRanges<ndims> {
public:
    void initialize(int *argc, const char **argv) {
        for(int d = 0; d < ndims; d++) {
            this->nranks[d] = 1;
            this->prev[d] = 0;
            this->next[d] = 0;
            this->pbc_prev[d] = 1;
            this->pbc_next[d] = -1;
            this->subdom_min[d] = this->grid_min[d];
            this->subdom_max[d] = this->grid_max[d];
        }
    }

    void finalize() {}

    void fillArrays(int neighbor_ranks[], int pbc[], real_t subdom[]) {
        for(int d = 0; d < ndims; d++) {
            neighbor_ranks[d * 2 + 0] = this->prev[d];
            neighbor_ranks[d * 2 + 1] = this->next[d];
            pbc[d * 2 + 0] = this->pbc_prev[d];
            pbc[d * 2 + 1] = this->pbc_next[d];
            subdom[d * 2 + 0] = this->subdom_min[d];
            subdom[d * 2 + 1] = this->subdom_max[d];
        }
    }

    void communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
        recv_sizes[dim * 2 + 0] = send_sizes[dim * 2 + 0];
        recv_sizes[dim * 2 + 1] = send_sizes[dim * 2 + 1];
    }

    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

        for(int i = 0; i < nsend[dim * 2 + 0] * elem_size; i++) {
            recv_buf[recv_offsets[dim * 2 + 0] + i] = send_buf[send_offsets[dim * 2 + 0] + i];
        }

        for(int i = 0; i < nsend[dim * 2 + 1] * elem_size; i++) {
            recv_buf[recv_offsets[dim * 2 + 1] + i] = send_buf[send_offsets[dim * 2 + 1] + i];
        }
    }
};

}
