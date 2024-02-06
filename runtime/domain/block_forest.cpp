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
#include <pe/amr/level_determination/MinMaxLevelDetermination.h>
#include <pe/amr/weight_assignment/MetisAssignmentFunctor.h>
#include <pe/amr/weight_assignment/WeightAssignmentFunctor.h>
//---
#include "../boundary_weights.hpp"
#include "../pairs_common.hpp"
#include "../devices/device.hpp"
#include "regular_6d_stencil.hpp"
#include "ParticleDataHandling.h"

namespace pairs {

void BlockForest::updateNeighborhood() {
    auto me = mpi::MPIManager::instance()->rank();
    this->nranks = 0;
    this->total_aabbs = 0;

    ranks.clear();
    naabbs.clear();
    aabbs.clear();

    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
        auto& block_info = info[block->getId()];

        if(block_info.computationalWeight > 0) {
            for(uint neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
                auto neighbor_rank = int_c(block->getNeighborProcess(neigh));

                if(neighbor_rank != me) {
                    const BlockID& neighbor_block = block->getNeighborId(neigh);
                    math::AABB neighbor_aabb = block->getNeighborAABB(neigh);
                    auto neighbor_info = info[neighbor_block];
                    auto begin = blocks_pushed[neighbor_rank].begin();
                    auto end = blocks_pushed[neighbor_rank].end();

                    if(neighbor_info.computationalWeight > 0 &&
                       find_if(begin, end, [nb](const auto &nbh) { return nbh == nb; }) == end) {

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
    void *src = name.compare('ranks') ? ranks.data() :
                name.compare('naabbs') ? vec_naabbs.data() :
                name.compare('rank_offsets') ? offsets :
                name.compare('aabbs') ? vec_aabbs.data() :
                name.compare('subdom') ? subdom;

    bool is_real = name.compare('aabbs') || name.compare('subdom');
    int tsize = is_real ? sizeof(real_t) : sizeof(int);
    std::memcpy(dest, src, size * tsize);
}

void BlockForest::updateWeights(PairsSimulation *ps, int nparticles) {
    mpi::BufferSystem bs(mpi::MPIManager::instance()->comm(), 756);

    info.clear();
    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
        auto aabb = block->getAABB();
        auto& block_info = info[block->getId()];

        pairs::compute_boundary_weights(
            ps,
            aabb.xMin(), aabb.xMax(), aabb.yMin(), aabb.yMax(), aabb.zMin(), aabb.zMax(),
            &(block_info.computationalWeight), &(block_info.communicationWeight));

        for(uint_t branch = 0; branch < 8; ++branch) {
            const auto b_id = BlockID(block->getId(), branch);
            const auto b_aabb = forest->getAABBFromBlockId(b_id);
            auto& b_info = info[b_id];

            pairs::compute_boundary_weights(
                ps,
                b_aabb.xMin(), b_aabb.xMax(), b_aabb.yMin(), b_aabb.yMax(), b_aabb.zMin(), b_aabb.zMax(),
                &(b_info.computationalWeight), &(b_info.communicationWeight));
        }
    }

    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
        auto& block_info = info[block->getId()];

        for(uint_t neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
            bs.sendBuffer(block->getNeighborProcess(neigh)) <<
                blockforest::InfoCollection::value_type(block->getId(), block_info);
        }

        for(uint_t branch = 0; branch < 8; ++branch) {
            const auto b_id = BlockID(block->getId(), branch);
            auto& b_info = info[b_id];

            for(uint_t neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
                bs.sendBuffer(block->getNeighborProcess(neigh)) <<
                    blockforest::InfoCollection::value_type(b_id, b_info);
            }
        }
    }

    bs.setReceiverInfoFromSendBufferState(false, true);
    bs.sendAll();

    for(auto recv = bs.begin(); recv != bs.end(); ++recv) {
        while(!recv.buffer().isEmpty()) {
            blockforest::InfoCollectionPair val;
            recv.buffer() >> val;
            info.insert(val);
        }
    }
}

Vector3<uint_t> BlockForest::getBlockConfig(uint_t num_processes, uint_t nx, uint_t ny, uint_t nz) {
    const uint_t bx_factor = 1;
    const uint_t by_factor = 1;
    const uint_t bz_factor = 1;
    const uint_t ax = nx * ny;
    const uint_t ay = nx * nz;
    const uint_t az = ny * nz;

    uint_t bestsurf = 2 * (ax + ay + az);
    uint_t x = 1;
    uint_t y = 1;
    uint_t z = 1;

    for(uint_t i = 1; i < num_processes; ++i) {
        if(num_processes % i == 0) {
            const uint_t rem_yz = num_processes / i;

            for(uint_t j = 1; j < rem_yz; ++j) {
                if(rem_yz % j == 0) {
                    const uint_t k = rem_yz / j;
                    const uint_t surf = (ax / i / j) + (ay / i / k) + (az / j / k);

                    if(surf < bestsurf) {
                        x = i, y = j, z = k;
                        bestsurf = surf;
                    }
                }
            }
        }
    }

    return Vector3<uint_t>(x * bx_factor, y * by_factor, z * bz_factor);
}

uint_t BlockForest::getInitialRefinementLevel(uint_t num_processes) {
    uint_t splitFactor = 8;
    uint_t blocks = splitFactor;
    uint_t refinementLevel = 1;

    while(blocks < num_processes) {
        refinementLevel++;
        blocks *= splitFactor;
    }

    return refinementLevel;
}

void BlockForest::setBoundingBox() {
    auto aabb_union = forest->begin()->getAABB();
    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
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

    auto mpiManager = mpi::MPIManager::instance();
    mpiManager->initializeMPI(&argc, &argv);
    mpiManager->useWorldComm();

    math::AABB domain(xmin, ymin, zmin, xmax, ymax, zmax);
    int gridsize[3] = {32, 32, 32};
    auto procs = mpiManager->numProcesses();
    auto block_config = use_load_balancing ? Vector3<uint_t>(1, 1, 1) : getBlockConfig(procs, gridsize[0], gridsize[1], gridsize[2]);
    auto ref_level = use_load_balancing ? getInitialRefinementLevel(procs) : 0;

    forest = blockforest::createBlockForest(
        domain, block_config, Vector3<bool>(true, true, true), procs, ref_level);

    info = make_shared<blockforest::InfoCollection>();
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
    forest->setRefreshMinTargetLevelDeterminationFunction(pe::amr::MinMaxLevelDetermination(info, regridMin, regridMax));

    for_each(algorithm.begin(), algorithm.end(), [](char& c) { c = (char) ::tolower(c); });

    if(algorithm == "morton") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(pe::amr::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = blockforest::DynamicCurveBalance<pe::amr::WeightAssignmentFunctor::PhantomBlockWeight>(false, true, false);

        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "hilbert") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(pe::amr::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = blockforest::DynamicCurveBalance<pe::amr::WeightAssignmentFunctor::PhantomBlockWeight>(true, true, false);

        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "metis") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(pe::amr::MetisAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(pe::amr::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(pe::amr::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto alg = blockforest::DynamicParMetis::stringToAlgorithm(metisAlgorithm);
        auto vWeight = blockforest::DynamicParMetis::stringToWeightsToUse(metisWeightsToUse);
        auto eWeight = blockforest::DynamicParMetis::stringToEdgeSource(metisEdgeSource);
        auto prepFunc = blockforest::DynamicParMetis(alg, vWeight, eWeight);

        prepFunc.setipc2redist(metisipc2redist);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    } else if(algorithm == "diffusive") {
        forest->setRefreshPhantomBlockDataAssignmentFunction(pe::amr::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(pe::amr::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = blockforest::DynamicDiffusionBalance<pe::amr::WeightAssignmentFunctor::PhantomBlockWeight>(1, 1, false);

        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    }

    forest->addBlockData(make_shared<ParticleDataHandling>(), "Interface");
}

void BlockForest::finalize() {
    MPI_Finalize();
}

bool BlockForest::isWithinSubdomain(real_t x, real_t y, real_t z) {
    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);

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
        MPI_Irecv(&recv_sizes[i], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &nrecv[i], &recv_requests[i]);
        MPI_Isend(&send_sizes[i], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &nsend[i], &send_requests[i]);
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
