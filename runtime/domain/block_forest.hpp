#include <map>
//---
#include <blockforest/BlockForest.h>
#include <blockforest/Initialization.h>
#include <blockforest/loadbalancing/DynamicCurve.h>
#include <blockforest/loadbalancing/DynamicDiffusive.h>
#include <blockforest/loadbalancing/DynamicParMetis.h>
#include <blockforest/loadbalancing/InfoCollection.h>
#include <blockforest/loadbalancing/PODPhantomData.h>
#include <blockforest/loadbalancing/level_determination/MinMaxLevelDetermination.h>
#include <blockforest/loadbalancing/weight_assignment/MetisAssignmentFunctor.h>
#include <blockforest/loadbalancing/weight_assignment/WeightAssignmentFunctor.h>
//---
#include "../pairs_common.hpp"
#include "domain_partitioning.hpp"

#pragma once

#define SMALL 0.00001

namespace pairs {

class BlockForest : public DomainPartitioner {
private:
    std::shared_ptr<BlockForest> forest;
    walberla::blockforest::InfoCollection info;
    std::map<int, std::vector<walberla::math::AABB>> neighborhood;
    std::map<int, std::vector<walberla::BlockID>> blocks_pushed;
    std::vector<int> ranks;
    std::vector<int> naabbs;
    std::vector<int> aabb_offsetss;
    std::vector<double> aabbs;
    real_t *subdom;
    int world_size, rank, nranks, total_aabbs;
    bool balance_workload;

public:
    BlockForest(
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) :
        DomainPartitioner(xmin, xmax, ymin, ymax, zmin, zmax) {

        subdom = new real_t[ndims * 2];
    }

    ~BlockForest() {
        delete[] subdom;
    }

    void initialize(int *argc, char ***argv);
    void finalize();
    int getWorldSize() const { return world_size; }
    int getRank() const { return rank; }
    int getNumberOfNeighborRanks() { return this->nranks; }
    int getNumberOfNeighborAABBs() { return this->total_aabbs; }

    void initializeWorkloadBalancer();
    void updateNeighborhood();
    void updateWeights(real_t *position, int nparticles);
    walberla::Vector3<int> getBlockConfig(int num_processes, int nx, int ny, int nz);
    int getInitialRefinementLevel(int num_processes);
    void setBoundingBox();
    void rebalance();

    int isWithinSubdomain(real_t x, real_t y, real_t z);
    void copyRuntimeArray(const std::string& name, void *dest, const int size);
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
