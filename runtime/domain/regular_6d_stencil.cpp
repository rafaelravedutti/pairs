#include <mpi.h>
#include <vector>
//---
#include "../pairs_common.hpp"
#include "../devices/device.hpp"
#include "regular_6d_stencil.hpp"

namespace pairs {

void Regular6DStencil::setConfig() {
    PAIRS_ASSERT(ndims == 3);
    real_t area[3];
    real_t best_surf = 0.0;
    int d = 0;

    for(int d1 = 0; d1 < ndims; d1++) {
        nranks[d1] = 1;

        for(int d2 = d1 + 1; d2 < ndims; d2++) {
            area[d] = (this->grid_max[d1] - this->grid_min[d1]) *
                      (this->grid_max[d2] - this->grid_min[d2]);

            best_surf += 2.0 * area[d];
            d++;
        }
    }

    for (int i = 1; i <= world_size; i++) {
        if (world_size % i == 0) {
            const int rem_yz = world_size / i;

            for (int j = 1; j <= rem_yz; j++) {
                if (rem_yz % j == 0) {
                    const int k = rem_yz / j;

                    // Check flags for each dimension
                    if((partition_flags[0] || i == 1) &&
                       (partition_flags[1] || j == 1) &&
                       (partition_flags[2] || k == 1)) {

                        const real_t surf = (area[0] / i / j) + (area[1] / i / k) + (area[2] / j / k);

                        if (surf < best_surf) {
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
}

void Regular6DStencil::setBoundingBox() {
    MPI_Comm cartesian;
    int *myloc = new int[ndims];
    int *periods = new int[ndims];
    real_t *rank_length = new real_t[ndims];
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
        subdom_min[d] = this->grid_min[d] + rank_length[d] * (real_t) myloc[d];
        subdom_max[d] = subdom_min[d] + rank_length[d];
    }

    delete[] myloc;
    delete[] periods;
    delete[] rank_length;
    MPI_Comm_free(&cartesian);
}

void Regular6DStencil::initialize(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    this->setConfig();
    this->setBoundingBox();
}

void Regular6DStencil::finalize() {
    MPI_Finalize();
}

int Regular6DStencil::isWithinSubdomain(real_t x, real_t y, real_t z) {
    return x >= subdom_min[0] && x < subdom_max[0] - SMALL &&
           y >= subdom_min[1] && y < subdom_max[1] - SMALL &&
           z >= subdom_min[2] && z < subdom_max[2] - SMALL;
}

void Regular6DStencil::copyRuntimeArray(const std::string& name, void *dest, const int size) {
    for(int d = 0; d < ndims; d++) {
        if(name.compare('neighbor_ranks')) {
            int *neighbor_ranks = static_cast<int *>(dest);
            neighbor_ranks[d * 2 + 0] = prev[d];
            neighbor_ranks[d * 2 + 1] = next[d];
        }

        if(name.compare('pbc')) {
            int *pbc = static_cast<int *>(dest);
            pbc[d * 2 + 0] = pbc_prev[d];
            pbc[d * 2 + 1] = pbc_next[d];
        }

        if(name.compare('subdom')) {
            real_t *subdom = static_cast<real_t *>(dest);
            subdom[d * 2 + 0] = subdom_min[d];
            subdom[d * 2 + 1] = subdom_max[d];
        }
    }
}

void Regular6DStencil::communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
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

void Regular6DStencil::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    const real_t *send_prev = &send_buf[send_offsets[dim * 2 + 0] * elem_size];
    const real_t *send_next = &send_buf[send_offsets[dim * 2 + 1] * elem_size];
    real_t *recv_prev = &recv_buf[recv_offsets[dim * 2 + 0] * elem_size];
    real_t *recv_next = &recv_buf[recv_offsets[dim * 2 + 1] * elem_size];

    if(prev[dim] != rank) {
        MPI_Sendrecv(
            send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0,
            recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        pairs::copy_in_device(recv_prev, send_prev, nsend[dim * 2 + 0] * elem_size * sizeof(real_t));
    }

    if(next[dim] != rank) {
        MPI_Sendrecv(
            send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0,
            recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        pairs::copy_in_device(recv_next, send_next, nsend[dim * 2 + 1] * elem_size * sizeof(real_t));
    }
}

void Regular6DStencil::communicateAllData(
    int ndims, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    for (int d = 0; d < ndims; d++) {
        const real_t *send_prev = &send_buf[send_offsets[d * 2 + 0] * elem_size];
        const real_t *send_next = &send_buf[send_offsets[d * 2 + 1] * elem_size];
        real_t *recv_prev = &recv_buf[recv_offsets[d * 2 + 0] * elem_size];
        real_t *recv_next = &recv_buf[recv_offsets[d * 2 + 1] * elem_size];

        if (prev[d] != rank) {
            MPI_Sendrecv(
                send_prev, nsend[d * 2 + 0] * elem_size, MPI_DOUBLE, prev[d], 0,
                recv_prev, nrecv[d * 2 + 0] * elem_size, MPI_DOUBLE, next[d], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            pairs::copy_in_device(recv_prev, send_prev, nsend[d * 2 + 0] * elem_size * sizeof(real_t));
        }

        if (next[d] != rank) {
            MPI_Sendrecv(
                send_next, nsend[d * 2 + 1] * elem_size, MPI_DOUBLE, next[d], 0,
                recv_next, nrecv[d * 2 + 1] * elem_size, MPI_DOUBLE, prev[d], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {
            pairs::copy_in_device(recv_next, send_next, nsend[d * 2 + 1] * elem_size * sizeof(real_t));
        }
    }
}

}
