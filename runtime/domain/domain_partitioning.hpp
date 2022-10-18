#include <mpi.h>
//---
#include "../pairs_common.hpp"

#pragma once

namespace pairs {

class Regular6DStencil;

class DomainPartitioner {
    friend class Regular6DStencil;

protected:
    real_t *grid_min;
    real_t *grid_max;
    int ndims;

public:
    DomainPartitioner(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
        PAIRS_ASSERT(xmax > xmin);
        PAIRS_ASSERT(ymax > ymin);
        PAIRS_ASSERT(zmax > zmin);

        ndims = 3;
        grid_min = new real_t[ndims];
        grid_max = new real_t[ndims];
        grid_min[0] = xmin;
        grid_min[1] = ymin;
        grid_min[2] = zmin;
        grid_max[0] = xmax;
        grid_max[1] = ymax;
        grid_max[2] = zmax;
    }

    ~DomainPartitioner() {
        delete[] grid_min;
        delete[] grid_max;
    }

    virtual void initialize(int *argc, char ***argv) = 0;
    virtual void fillArrays(int *neighbor_ranks, int *pbc, real_t *subdom) = 0;
    virtual void communicateSizes(int dim, const int *nsend, int *nrecv) = 0;
    virtual void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv) = 0;
    virtual void finalize() = 0;
};

}
