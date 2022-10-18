#include "domain_partitioning.hpp"

#pragma once

#define SMALL   0.00001

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
        int d = 0;

        for(int d1 = 0; d1 < ndims; d1++) {
            nranks[d1] = 1;
            for(int d2 = d1 + 1; d2 < ndims; d2++) {
                area[d] = (this->grid_max[d1] - this->grid_min[d1]) * (this->grid_max[d2] - this->grid_min[d2]);
                best_surf += 2.0 * area[d];
                d++;
            }
        }

        for(int i = 1; i < world_size; i++) {
            if(world_size % i == 0) {
                const int rem_yz = world_size / i;
                for(int j = 1; j < rem_yz; j++) {
                    if(rem_yz % j == 0) {
                        const int k = rem_yz / j;
                        const real_t surf = (area[0] / i / j) + (area[1] / i / k) + (area[2] / j / k);
                        if(surf < best_surf) {
                            nranks[0] = i;
                            nranks[1] = j;
                            nranks[2] = k;
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
            rank_length[d] = (this->grid_max[d] - this->grid_min[d]) / (real_t) nranks[d];
        }

        MPI_Cart_create(MPI_COMM_WORLD, ndims, nranks, periods, reorder, &cartesian);
        MPI_Cart_get(cartesian, ndims, nranks, periods, myloc);
        for(int d = 0; d < ndims; d++) {
            MPI_Cart_shift(cartesian, d, 1, &(prev[d]), &(next[d]));
            pbc_prev[d] = (myloc[d] == 0) ? 1 : 0;
            pbc_next[d] = (myloc[d] == nranks[d] - 1) ? -1 : 0;
            subdom_min[d] = this->grid_min[d] + rank_length[d] * (real_t)myloc[d];
            subdom_max[d] = subdom_min[d] + rank_length[d];
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

    int getWorldSize() const { return world_size; }
    int getRank() const { return rank; }
    int isWithinSubdomain(real_t x, real_t y, real_t z) {
        return x >= subdom_min[0] && x < subdom_max[0] - SMALL &&
               y >= subdom_min[1] && y < subdom_max[1] - SMALL &&
               z >= subdom_min[2] && z < subdom_max[2] - SMALL;
    }

    void fillArrays(int neighbor_ranks[], int pbc[], real_t subdom[]) {
        for(int d = 0; d < ndims; d++) {
            neighbor_ranks[d * 2 + 0] = prev[d];
            neighbor_ranks[d * 2 + 1] = next[d];
            pbc[d * 2 + 0] = pbc_prev[d];
            pbc[d * 2 + 1] = pbc_next[d];
            subdom[d * 2 + 0] = subdom_min[d];
            subdom[d * 2 + 1] = subdom_max[d];
        }
    }

    void communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
        std::cout << "communicateSizes" << std::endl;

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
        std::cout << "communicateData" << std::endl;

        const real_t *send_prev = &send_buf[send_offsets[dim * 2 + 0] * elem_size];
        const real_t *send_next = &send_buf[send_offsets[dim * 2 + 1] * elem_size];
        real_t *recv_prev = &recv_buf[recv_offsets[dim * 2 + 0] * elem_size];
        real_t *recv_next = &recv_buf[recv_offsets[dim * 2 + 1] * elem_size];

        if(prev[dim] != rank) {
            std::cout << rank << ": send " << nsend[dim * 2 + 0] << " elems to " << prev[dim] << ", recv " << nrecv[dim * 2 + 0] << " elems from " << next[dim] << std::endl;
            //MPI_Send(send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD);
            //MPI_Recv(recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(
                send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0,
                recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } else {
            for(int i = 0; i < nsend[dim * 2 + 0] * elem_size; i++) {
                recv_prev[i] = send_prev[i];
            }
        }

        if(next[dim] != rank) {
            std::cout << rank << ": send " << nsend[dim * 2 + 1] << " elems to " << next[dim] << ", recv " << nrecv[dim * 2 + 1] << " elems from " << prev[dim] << std::endl;
            //MPI_Send(send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0, MPI_COMM_WORLD);
            //MPI_Recv(recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(
                send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0,
                recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } else {
            for(int i = 0; i < nsend[dim * 2 + 1] * elem_size; i++) {
                recv_next[i] = send_next[i];
            }
        }
    }
};

}
