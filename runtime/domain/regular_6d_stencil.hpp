#include "domain_partitioning.hpp"

#pragma once

typedef double real_t;

namespace pairs {

template <int ndims>
class Regular6DStencil : public DomainPartitioner<ndims> {
private:
    int world_size, rank;
    int nranks[ndims];
    int prev[ndims];
    int next[ndims];
    int pbc_prev[ndims];
    int pbc_next[ndims];
    real_t subdom_min[ndims];
    real_t subdom_max[ndims];

public:
    Regular6DStencil(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) :
        DomainPartitioner<ndims>(xmin, xmax, ymin, ymax, zmin, zmax) {}

    void setConfig() {
        static_assert(ndims == 3, "setConfig() only implemented for three dimensions!");
        real_t area[ndims];
        real_t best_surf = 0.0;

        for(int d = 0; d < ndims; d++) {
            this->nranks[d] = 1;
            area[d] = (this->grid_max[d] - this->grid_min[d]) * (this->grid_max[d] - this->grid_min[d]);
            best_surf += 2.0 * area[d];
        }

        for(int i = 1; i < this->world_size; i++) {
            if(this->world_size % i == 0) {
                const int rem_yz = this->world_size / i;
                for(int j = 0; j < rem_yz; j++) {
                    if(rem_yz % j == 0) {
                        const int k = rem_yz / j;
                        const real_t surf = (area[0] / i / j) + (area[1] / i / k) + (area[2] / j / k);
                        if(surf < best_surf) {
                            this->nranks[0] = i;
                            this->nranks[1] = j;
                            this->nranks[2] = k;
                            best_surf = surf;
                        }
                    }
                }
            }
        }
    }

    void setBoundingBox() {
        MPI_Comm cartesian;
        int myloc[ndims];
        int periods[ndims];
        real_t rank_length[ndims];
        int reorder = 0;

        for(int d = 0; d < ndims; d++) {
            periods[d] = 1;
            rank_length[d] = (this->grid_max[d] - this->grid_min[d]) / (real_t)this->nranks[d];
        }

        MPI_Cart_create(MPI_COMM_WORLD, ndims, this->nranks, periods, reorder, &cartesian);
        MPI_Cart_get(cartesian, ndims, this->nranks, periods, myloc);
        for(int d = 0; d < ndims; d++) {
            MPI_Cart_shift(cartesian, d, 1, &(this->prev[d]), &(this->next[d]));
            this->pbc_prev[d] = (myloc[d] == 0) ? 1 : 0;
            this->pbc_next[d] = (myloc[d] == this->nranks[d] - 1) ? -1 : 0;
            this->subdom_min[d] = this->grid_min[d] + rank_length[d] * (real_t)myloc[d];
            this->subdom_max[d] = this->subdom_min[d] + rank_length[d];
        }

        MPI_Comm_free(&cartesian);
    }

    void initialize(int *argc, char ***argv) {
        MPI_Init(argc, argv);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        this->setConfig();
        this->setBoundingBox();
    }

    void finalize() {
        MPI_Finalize();
    }

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
        if(prev[dim] != rank) {
            MPI_Send(&send_sizes[dim * 2 + 0], 1, MPI_INT, prev[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[dim * 2 + 0], 1, MPI_INT, next[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            recv_sizes[dim * 2 + 0] = send_sizes[dim * 2 + 0];
        }

        if(next[dim] != rank) {
            MPI_Send(&send_sizes[dim * 2 + 1], 1, MPI_INT, next[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(&recv_sizes[dim * 2 + 1], 1, MPI_INT, prev[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            recv_sizes[dim * 2 + 1] = send_sizes[dim * 2 + 1];
        }
    }

    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

        const real_t *send_prev = &send_buf[send_offsets[dim * 2 + 0] * elem_size];
        const real_t *send_next = &send_buf[send_offsets[dim * 2 + 1] * elem_size];
        real_t *recv_prev = &recv_buf[recv_offsets[dim * 2 + 0] * elem_size];
        real_t *recv_next = &recv_buf[recv_offsets[dim * 2 + 1] * elem_size];

        if(prev[dim] != rank) {
            MPI_Send(send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            for(int i = 0; i < nsend[dim * 2 + 0] * elem_size; i++) {
                recv_prev[i] = send_prev[i];
            }
        }

        if(next[dim] != rank) {
            MPI_Send(send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD);
            MPI_Recv(recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            for(int i = 0; i < nsend[dim * 2 + 1] * elem_size; i++) {
                recv_next[i] = send_next[i];
            }
        }
    }
};

}
