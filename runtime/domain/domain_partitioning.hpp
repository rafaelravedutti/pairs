//---
#include "pairs.hpp"

#pragma once

namespace pairs {

template<int ndims>
class DomainPartitioning {
protected:
    int world_size, rank;
    real_t grid_min[ndims];
    real_t grid_max[ndims];
    
public:
    DomainPartitioning(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
        static_assert(ndims == 3, "Constructor called from DomainPartitioning class mismatches number of dimensions!");
        PAIRS_ASSERT(xmax > xmin);
        PAIRS_ASSERT(ymax > ymin);
        PAIRS_ASSERT(zmax > zmin);

        grid_min[0] = xmin;
        grid_max[0] = xmax;
        grid_min[1] = ymin;
        grid_max[1] = ymax;
        grid_min[2] = zmin;
        grid_max[2] = zmax;
    }

    virtual void initialize();
    virtual void finalize();
};

template<int ndims>
class DimensionRanges : DomainPartitioning<ndims> {
protected:
    int nranks[ndims];
    int prev[ndims];
    int next[ndims];
    int pbc_prev[ndims];
    int pbc_next[ndims];
    real_t subdom_min[ndims];
    real_t subdom_max[ndims];

public:
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
};

class ListOfBoxes : DomainPartitioning {};

}
