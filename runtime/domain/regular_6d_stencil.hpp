#include "../pairs_common.hpp"
#include "domain_partitioning.hpp"

#pragma once

#define SMALL 0.00001

namespace pairs {

class Regular6DStencil : public DomainPartitioner {
private:
    int world_size, rank;
    int *partition_flags;
    int *nranks;
    int *prev;
    int *next;
    int *pbc_prev;
    int *pbc_next;
    real_t *subdom_min;
    real_t *subdom_max;

public:
    Regular6DStencil(
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax, const int part[]) :
        DomainPartitioner(xmin, xmax, ymin, ymax, zmin, zmax) {

        nranks = new int[ndims];
        prev = new int[ndims];
        next = new int[ndims];
        pbc_prev = new int[ndims];
        pbc_next = new int[ndims];
        subdom_min = new real_t[ndims];
        subdom_max = new real_t[ndims];
        partition_flags = new int[ndims];

        for(int d = 0; d < ndims; d++) {
            partition_flags[d] = part[d];
        }
    }

    ~Regular6DStencil() {
        delete[] nranks;
        delete[] prev;
        delete[] next;
        delete[] pbc_prev;
        delete[] pbc_next;
        delete[] subdom_min;
        delete[] subdom_max;
    }

    void setConfig();
    void setBoundingBox();
    void initialize(int *argc, char ***argv);
    void finalize();

    int getWorldSize() const { return world_size; }
    int getRank() const { return rank; }
    int getNumberOfNeighborRanks() { return 6; }
    int getNumberOfNeighborAABBs() { return 6; }

    int isWithinSubdomain(real_t x, real_t y, real_t z);
    void fillArrays(int *neighbor_ranks, int *pbc, real_t *subdom);
    void communicateSizes(int dim, const int *send_sizes, int *recv_sizes);
    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv);

    void communicateAllData(
        int ndims, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv);
};

}
