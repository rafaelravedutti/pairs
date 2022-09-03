#include "domain_partitioning.hpp"

#pragma once

typedef double real_t;

namespace pairs {

template <int ndims>
class Regular6DStencil : public DimensionRanges<ndims> {
public:
    Regular6DStencil(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) :
        DimensionRanges<ndims>(xmin, xmax, ymin, ymax, zmin, zmax) {}

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
        int rank_length[ndims];
        int reorder = 0;

        for(int d = 0; d < ndims; d++) {
            periods[d] = 1;
            rank_length[d] = (this->grid_max[d] - this->grid_min[d]) / this->nranks[d];
        }

        MPI_Cart_create(MPI_COMM_WORLD, ndims, this->nranks, periods, reorder, &cartesian);
        MPI_Cart_get(cartesian, ndims, this->nranks, periods, myloc);
        for(int d = 0; d < ndims; d++) {
            MPI_Cart_shift(cartesian, d, 1, &(this->prev[d]), &(this->next[d]));
            this->pbc_prev[d] = (myloc[d] == 0) ? 1 : 0;
            this->pbc_next[d] = (myloc[d] == this->nranks[d]) ? -1 : 0;
            this->subdom_min[d] = this->grid_min[d] + rank_length[d] * myloc[d];
            this->subdom_max[d] = this->subdom_min[d] + rank_length[d];
        }

        MPI_Comm_free(&cartesian);
    }

    void initialize(int *argc, char ***argv) {
        MPI_Init(argc, argv);
        MPI_Comm_size(MPI_COMM_WORLD, &(this->world_size));
        MPI_Comm_rank(MPI_COMM_WORLD, &(this->rank));
        this->setConfig();
        this->setBoundingBox();
    }

    void finalize() {
        MPI_Finalize();
    }
};

}
