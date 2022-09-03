#include "pairs.hpp"

#pragma once

namespace pairs {

template<int ndims>
void initDomain(PairsSimulation<ndims> *pairs, int *argc, char ***argv, real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
    pairs->initDomain(argc, argv, xmin, xmax, ymin, ymax, zmin, zmax);
}

template<int ndims>
void communicateSizes(PairsSimulation<ndims> *pairs, int dim, const int *send_sizes, int *recv_sizes) {
    pairs->getDomainPartitioner()->communicateSizes(dim, send_sizes, recv_sizes);
}

template<int ndims>
void communicateData(
    PairsSimulation<ndims> *pairs, int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    pairs->getDomainPartitioner()->communicateData(dim, elem_size, send_buf, send_offsets, nsend, recv_buf, recv_offsets, nrecv);
}

template<int ndims>
void fillCommunicationArrays(PairsSimulation<ndims> *pairs, int neighbor_ranks[], int pbc[], real_t subdom[]) {
    pairs->getDomainPartitioner()->fillArrays(neighbor_ranks, pbc, subdom);
}

}
