#include <map>
//---
#include "../pairs_common.hpp"
#include "domain_partitioning.hpp"

#pragma once

#define SMALL 0.00001

namespace pairs {

class BlockForest : public DomainPartitioner {
private:
    std::shared_ptr<BlockForest> forest;
    std::map<uint_t, std::vector<math::AABB>> neighborhood;
    std::map<uint_t, std::vector<BlockID>> blocks_pushed;
    std::vector<int> ranks;
    std::vector<int> naabbs;
    std::vector<double> aabbs;
    real_t *subdom_min;
    real_t *subdom_max;
    int world_size, rank, total_aabbs;

public:
    BlockForest(
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) :
        DomainPartitioner(xmin, xmax, ymin, ymax, zmin, zmax) {

        subdom_min = new real_t[ndims];
        subdom_max = new real_t[ndims];
    }

    ~BlockForest() {
        delete[] subdom_min;
        delete[] subdom_max;
    }

    void setConfig();
    void setBoundingBox();
    void initialize(int *argc, char ***argv);
    void finalize();
    int getWorldSize() const { return world_size; }
    int getRank() const { return rank; }
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
