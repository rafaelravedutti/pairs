#include <mpi.h>

#pragma once

typedef double real_t;

namespace pairs {

template<int ndims> class Regular6DStencil;

template<int ndims> class DomainPartitioner {
    friend class Regular6DStencil<ndims>;

protected:
    real_t grid_min[ndims];
    real_t grid_max[ndims];
    
public:
    DomainPartitioner(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
        static_assert(ndims == 3, "DomainPartitioner(): number of dimensions mismatching!");
        //PAIRS_ASSERT(xmax > xmin);
        //PAIRS_ASSERT(ymax > ymin);
        //PAIRS_ASSERT(zmax > zmin);

        grid_min[0] = xmin;
        grid_max[0] = xmax;
        grid_min[1] = ymin;
        grid_max[1] = ymax;
        grid_min[2] = zmin;
        grid_max[2] = zmax;
    }

    virtual void initialize(int *argc, char ***argv) = 0;
    virtual void fillArrays(int neighbor_ranks[], int pbc[], real_t subdom[]) = 0;
    virtual void communicateSizes(int dim, const int *nsend, int *nrecv) = 0;
    virtual void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv) = 0;
    virtual void finalize() = 0;
};

//class DomainPartitioner<3>;

}
