//---
#include "pairs.hpp"
#include "domain_partitioning.hpp"

#pragma once

namespace pairs {

template <int ndims>
class NoPartitioning : DimensionRanges<ndims> {
public:
    void initialize(int *argc, const char **argv) {
        this->world_size = 1;
        this->rank = 0;

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
};

}
