#include <memory>
#include <map>

#include "../pairs_common.hpp"
#include "domain_partitioning.hpp"

#pragma once

#define SMALL 0.00001

namespace walberla {
    namespace blockforest{
        class BlockForest;
        class BlockID;
        class BlockInfo;
        using InfoCollection = std::map<BlockID, BlockInfo>;
    }

    namespace mpi {
        class MPIManager;
    }

    namespace math{
        template<typename T> 
        class Vector3;
    }
}
namespace pairs {

class PairsRuntime;

class BlockForest : public DomainPartitioner {
private:
    std::shared_ptr<walberla::mpi::MPIManager> mpiManager;
    std::shared_ptr<walberla::blockforest::BlockForest> forest;
    std::shared_ptr<walberla::blockforest::InfoCollection> info;
    std::vector<int> ranks;
    std::vector<int> naabbs;
    std::vector<int> aabb_offsets;
    std::vector<double> aabbs;
    PairsRuntime *ps;
    real_t *subdom;
    const bool globalPBC[3];
    int world_size, rank, nranks, total_aabbs;
    bool balance_workload;

public:
    BlockForest(
        PairsRuntime *ps_,
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax, bool pbcx, bool pbcy, bool pbcz);

    BlockForest(PairsRuntime *ps_, const std::shared_ptr<walberla::blockforest::BlockForest> &bf);

    ~BlockForest() {
        delete[] subdom;
    }

    void initialize(int *argc, char ***argv);
    void update();
    void finalize();
    int getWorldSize() const { return world_size; }
    int getRank() const { return rank; }
    int getNumberOfNeighborRanks() { return this->nranks; }
    int getNumberOfNeighborAABBs() { return this->total_aabbs; }

    void initializeWorkloadBalancer();
    void updateNeighborhood();
    void updateWeights();
    walberla::math::Vector3<int> getBlockConfig(int num_processes, int nx, int ny, int nz);
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
