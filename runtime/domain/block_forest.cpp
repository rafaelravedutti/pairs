#include <mpi.h>
#include <vector>
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
#include "../boundary_weights.hpp"
#include "../pairs_common.hpp"
#include "../devices/device.hpp"
#include "regular_6d_stencil.hpp"
#include "ParticleDataHandling.hpp"

namespace pairs {

void BlockForest::updateNeighborhood() {
    auto me = walberla::mpi::MPIManager::instance()->rank();
    this->nranks = 0;
    this->total_aabbs = 0;

    ranks.clear();
    naabbs.clear();
    aabb_offsets.clear();
    aabbs.clear();

    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        auto& block_info = info[block->getId()];

        if(block_info.computationalWeight > 0) {
            for(uint neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
                auto neighbor_rank = walberla::int_c(block->getNeighborProcess(neigh));

                if(neighbor_rank != me) {
                    const walberla::BlockID& neighbor_block = block->getNeighborId(neigh);
                    walberla::math::AABB neighbor_aabb = block->getNeighborAABB(neigh);
                    auto neighbor_info = info[neighbor_block];
                    auto begin = blocks_pushed[neighbor_rank].begin();
                    auto end = blocks_pushed[neighbor_rank].end();

                    if(neighbor_info.computationalWeight > 0 &&
                       find_if(begin, end, [neighbor_block](const auto &nbh) {
                            return nbh == neighbor_block; }) == end) {

                        neighborhood[neighbor_rank].push_back(neighbor_aabb);
                        blocks_pushed[neighbor_rank].push_back(neighbor_block);
                    }
                }
            }
        }
    }

    for(auto& nbh: neighborhood) {
        auto rank = nbh.first;
        auto aabb_list = nbh.second;
        ranks.push_back((int) rank);
        aabb_offsets.push_back(this->total_aabbs);
        naabbs.push_back((int) aabb_list.size());

        for(auto &aabb: aabb_list) {
            aabbs.push_back(aabb.xMin());
            aabbs.push_back(aabb.xMax());
            aabbs.push_back(aabb.yMin());
            aabbs.push_back(aabb.yMax());
            aabbs.push_back(aabb.zMin());
            aabbs.push_back(aabb.zMax());
            this->total_aabbs++;
        }

        this->nranks++;
    }
}

void BlockForest::copyRuntimeArray(const std::string& name, void *dest, const int size) {
    void *src = name.compare("ranks")           ? static_cast<void *>(ranks.data()) :
                name.compare("naabbs")          ? static_cast<void *>(naabbs.data()) :
                name.compare("aabb_offsets")    ? static_cast<void *>(aabb_offsets.data()) :
                name.compare("aabbs")           ? static_cast<void *>(aabbs.data()) :
                name.compare("subdom")          ? static_cast<void *>(subdom) : nullptr;

    PAIRS_ASSERT(src != nullptr);
    bool is_real = name.compare("aabbs") || name.compare("subdom");
    int tsize = is_real ? sizeof(real_t) : sizeof(int);
    std::memcpy(dest, src, size * tsize);
}

void BlockForest::updateWeights() {
    walberla::mpi::BufferSystem bs(walberla::mpi::MPIManager::instance()->comm(), 756);

    info.clear();
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        auto aabb = block->getAABB();
        auto& block_info = info[block->getId()];

        pairs::compute_boundary_weights(
            this->ps,
            aabb.xMin(), aabb.xMax(), aabb.yMin(), aabb.yMax(), aabb.zMin(), aabb.zMax(),
            &(block_info.computationalWeight), &(block_info.communicationWeight));

        for(int branch = 0; branch < 8; ++branch) {
            const auto b_id = walberla::BlockID(block->getId(), branch);
            const auto b_aabb = forest->getAABBFromBlockId(b_id);
            auto& b_info = info[b_id];

            pairs::compute_boundary_weights(
                this->ps,
                b_aabb.xMin(), b_aabb.xMax(), b_aabb.yMin(), b_aabb.yMax(), b_aabb.zMin(), b_aabb.zMax(),
                &(b_info.computationalWeight), &(b_info.communicationWeight));
        }
    }

    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        auto& block_info = info[block->getId()];

        for(int neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
            bs.sendBuffer(block->getNeighborProcess(neigh)) <<
                walberla::blockforest::InfoCollection::value_type(block->getId(), block_info);
        }

        for(int branch = 0; branch < 8; ++branch) {
            const auto b_id = walberla::BlockID(block->getId(), branch);
            auto& b_info = info[b_id];

            for(int neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
                bs.sendBuffer(block->getNeighborProcess(neigh)) <<
                    walberla::blockforest::InfoCollection::value_type(b_id, b_info);
            }
        }
    }

    bs.setReceiverInfoFromSendBufferState(false, true);
    bs.sendAll();

    for(auto recv = bs.begin(); recv != bs.end(); ++recv) {
        while(!recv.buffer().isEmpty()) {
            walberla::blockforest::InfoCollectionPair val;
            recv.buffer() >> val;
            info.insert(val);
        }
    }
}

walberla::Vector3<int> BlockForest::getBlockConfig(int num_processes, int nx, int ny, int nz) {
    const int bx_factor = 1;
    const int by_factor = 1;
    const int bz_factor = 1;
    const int ax = nx * ny;
    const int ay = nx * nz;
    const int az = ny * nz;

    int bestsurf = 2 * (ax + ay + az);
    int x = 1;
    int y = 1;
    int z = 1;

    for(int i = 1; i < num_processes; ++i) {
        if(num_processes % i == 0) {
            const int rem_yz = num_processes / i;

            for(int j = 1; j < rem_yz; ++j) {
                if(rem_yz % j == 0) {
                    const int k = rem_yz / j;
                    const int surf = (ax / i / j) + (ay / i / k) + (az / j / k);

                    if(surf < bestsurf) {
                        x = i, y = j, z = k;
                        bestsurf = surf;
                    }
                }
            }
        }
    }

    return walberla::Vector3<int>(x * bx_factor, y * by_factor, z * bz_factor);
}

int BlockForest::getInitialRefinementLevel(int num_processes) {
    int splitFactor = 8;
    int blocks = splitFactor;
    int refinementLevel = 1;

    while(blocks < num_processes) {
        refinementLevel++;
        blocks *= splitFactor;
    }

    return refinementLevel;
}

void BlockForest::setBoundingBox() {
    auto aabb_union = forest->begin()->getAABB();
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        aabb_union.merge(block->getAABB());
    }

    subdom[0] = aabb_union.xMin();
    subdom[1] = aabb_union.xMax();
    subdom[2] = aabb_union.yMin();
    subdom[3] = aabb_union.yMax();
    subdom[4] = aabb_union.zMin();
    subdom[5] = aabb_union.zMax();
}

void BlockForest::rebalance() {
    if(balance_workload) {
        this->updateWeights();
        forest->refresh();
    }

    this->setBoundingBox();
    this->updateWeights();
    this->updateNeighborhood();
}

void BlockForest::initialize(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto mpiManager = walberla::mpi::MPIManager::instance();
    mpiManager->initializeMPI(argc, argv);
    mpiManager->useWorldComm();

    walberla::math::AABB domain(
        grid_min[0], grid_min[1], grid_min[2], grid_max[0], grid_max[1], grid_max[2]);

    int gridsize[3] = {32, 32, 32};
    auto procs = mpiManager->numProcesses();
    auto block_config = balance_workload ? walberla::Vector3<int>(1, 1, 1) :
                                           getBlockConfig(procs, gridsize[0], gridsize[1], gridsize[2]);
    auto ref_level = balance_workload ? getInitialRefinementLevel(procs) : 0;

    forest = walberla::blockforest::createBlockForest(
        domain, block_config, walberla::Vector3<bool>(true, true, true), procs, ref_level);

    this->info = make_shared<walberla::blockforest::InfoCollection>();
    this->setBoundingBox();

    if(balance_workload) {
        this->initializeWorkloadBalancer();
    }
}

void BlockForest::initializeWorkloadBalancer() {
    const std::string algorithm = "morton";
    real_t baseWeight = 1.0;
    real_t metisipc2redist = 1.0;
    size_t regridMin = 10;
    size_t regridMax = 100;
    int maxBlocksPerProcess = 100;
    string metisAlgorithm = "none";
    string metisWeightsToUse = "none";
    string metisEdgeSource = "none";

    forest->recalculateBlockLevelsInRefresh(true);
    forest->alwaysRebalanceInRefresh(true);
    forest->reevaluateMinTargetLevelsAfterForcedRefinement(true);
    forest->allowRefreshChangingDepth(true);

    forest->allowMultipleRefreshCycles(false);
    forest->checkForEarlyOutInRefresh(false);
    forest->checkForLateOutInRefresh(false);
    forest->setRefreshMinTargetLevelDeterminationFunction(
        walberla::blockforest::MinMaxLevelDetermination(info, regridMin, regridMax));

    for_each(algorithm.begin(), algorithm.end(), [](char& c) { c = (char) ::tolower(c); });

    if(algorithm == "morton") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicCurveBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(false, true, false);

        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "hilbert") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicCurveBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(true, true, false);

        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "metis") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(walberla::blockforest::MetisAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(walberla::blockforest::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(walberla::blockforest::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto alg = walberla::blockforest::DynamicParMetis::stringToAlgorithm(metisAlgorithm);
        auto vWeight = walberla::blockforest::DynamicParMetis::stringToWeightsToUse(metisWeightsToUse);
        auto eWeight = walberla::blockforest::DynamicParMetis::stringToEdgeSource(metisEdgeSource);
        auto prepFunc = walberla::blockforest::DynamicParMetis(alg, vWeight, eWeight);

        prepFunc.setipc2redist(metisipc2redist);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "diffusive") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicDiffusionBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(1, 1, false);

        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    }

    forest->addBlockData(make_shared<walberla::ParticleDataHandling>(ps), "Interface");
}

void BlockForest::finalize() {
    MPI_Finalize();
}

int BlockForest::isWithinSubdomain(real_t x, real_t y, real_t z) {
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);

        if(block->getAABB().contains(x, y, z)) {
            return true;
        }
    }

    return false;
}

void BlockForest::communicateSizes(int dim, const int *nsend, int *nrecv) {
    std::vector<MPI_Request> send_requests(ranks.size(), MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_requests(ranks.size(), MPI_REQUEST_NULL);
    size_t nranks = 0;

    for(auto neigh_rank: ranks) {
        MPI_Irecv(&nrecv[nranks], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &recv_requests[nranks]);
        MPI_Isend(&nsend[nranks], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &send_requests[nranks]);
        nranks++;
    }

    MPI_Waitall(nranks, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(nranks, recv_requests.data(), MPI_STATUSES_IGNORE);
}

void BlockForest::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    std::vector<MPI_Request> send_requests(ranks.size(), MPI_REQUEST_NULL);
    std::vector<MPI_Request> recv_requests(ranks.size(), MPI_REQUEST_NULL);
    size_t nranks = 0;

    for(auto neigh_rank: ranks) {
        const real_t *send_ptr = &send_buf[send_offsets[nranks] * elem_size];
        real_t *recv_ptr = &recv_buf[recv_offsets[nranks] * elem_size];

        if(neigh_rank != rank) {
            MPI_Irecv(
                recv_ptr, nrecv[nranks] * elem_size, MPI_DOUBLE, neigh_rank, 0,
                MPI_COMM_WORLD, &recv_requests[0]);

            MPI_Isend(
                send_ptr, nsend[nranks] * elem_size, MPI_DOUBLE, neigh_rank, 0,
                MPI_COMM_WORLD, &send_requests[0]);

            nranks++;
        } else {
            pairs::copy_in_device(recv_ptr, send_ptr, nsend[nranks] * elem_size * sizeof(real_t));
        }

        nranks++;
    }

    MPI_Waitall(nranks, send_requests.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(nranks, recv_requests.data(), MPI_STATUSES_IGNORE);
}

void BlockForest::communicateAllData(
    int ndims, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    this->communicateData(0, elem_size, send_buf, send_offsets, nsend, recv_buf, recv_offsets, nrecv);
}

}
