#include <mpi.h>

#pragma once

typedef double real_t;

namespace pairs {

template<int ndims>
class DomainPartitioner {
protected:
    int world_size, rank;
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

    virtual void initialize();
    virtual void fillArrays(int neighbor_ranks[], int pbc[], real_t subdom[]);
    virtual void communicateSizes(int dim, const int *nsend, int *nrecv);
    virtual void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv);
    virtual void finalize();
    inline int getRank() { return rank; }
};

template<int ndims>
class DimensionRanges : DomainPartitioner<ndims> {
protected:
    int nranks[ndims];
    int prev[ndims];
    int next[ndims];
    int pbc_prev[ndims];
    int pbc_next[ndims];
    real_t subdom_min[ndims];
    real_t subdom_max[ndims];

public:
    DimensionRanges(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) :
        DomainPartitioner<ndims>(xmin, xmax, ymin, ymax, zmin, zmax) {}

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
        if(prev[dim] != this->getRank()) {
            MPI_Send(&send_sizes[dim * 2 + 0], 1, MPI_INT, prev[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[dim * 2 + 0], 1, MPI_INT, next[dim], 0, MPI_COMM_WORLD);
        } else {
            recv_sizes[dim * 2 + 0] = send_sizes[dim * 2 + 0];
        }

        if(next[dim] != this->getRank()) {
            MPI_Send(&send_sizes[dim * 2 + 1], 1, MPI_INT, next[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[dim * 2 + 1], 1, MPI_INT, prev[dim], 0, MPI_COMM_WORLD);
        } else {
            recv_sizes[dim * 2 + 1] = send_sizes[dim * 2 + 1];
        }
    }

    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

        if(prev[dim] != this->getRank()) {
            MPI_Send(&send_buf[send_offsets[dim * 2 + 0]], nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_buf[recv_offsets[dim * 2 + 0]], nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD);
        } else {
            for(int i = 0; i < nsend[dim * 2 + 0] * elem_size; i++) {
                recv_buf[recv_offsets[dim * 2 + 0] + i] = send_buf[send_offsets[dim * 2 + 0] + i];
            }
        }

        if(next[dim] != this->getRank()) {
            MPI_Send(&send_buf[send_offsets[dim * 2 + 1]], nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_buf[recv_offsets[dim * 2 + 1]], nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD);
        } else {
            for(int i = 0; i < nsend[dim * 2 + 1] * elem_size; i++) {
                recv_buf[recv_offsets[dim * 2 + 1] + i] = send_buf[send_offsets[dim * 2 + 1] + i];
            }
        }
    }
};

template<int ndims>
class ListOfBoxes : DomainPartitioner<ndims> {};

}
