#include <map>
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
#include "boundary_weights.hpp"
#include "../pairs_common.hpp"
#include "../devices/device.hpp"
#include "regular_6d_stencil.hpp"
#include "ParticleDataHandling.hpp"
#include "../unique_id.hpp"

namespace pairs {

BlockForest::BlockForest(
        PairsRuntime *ps_,
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax, bool balance_workload_) :
        DomainPartitioner(xmin, xmax, ymin, ymax, zmin, zmax), ps(ps_), balance_workload(balance_workload_) {

        subdom = new real_t[ndims * 2];
}

BlockForest::BlockForest(PairsRuntime *ps_, const std::shared_ptr<walberla::blockforest::BlockForest> &bf) :
        forest(bf),
        DomainPartitioner(bf->getDomain().xMin(), bf->getDomain().xMax(),
                        bf->getDomain().yMin(), bf->getDomain().yMax(),
                        bf->getDomain().zMin(), bf->getDomain().zMax()), 
        ps(ps_) {
            subdom = new real_t[ndims * 2];
            mpiManager = walberla::mpi::MPIManager::instance();
            world_size = mpiManager->numProcesses();
            rank = mpiManager->rank();
            this->info = make_shared<walberla::blockforest::InfoCollection>();
}

void BlockForest::updateNeighborhood() {
    std::map<int, std::vector<walberla::math::AABB>> neighborhood;
    std::map<int, std::vector<walberla::BlockID>> blocks_pushed;
    auto me = mpiManager->rank();
    this->nranks = 0;
    this->total_aabbs = 0;

    ranks.clear();
    naabbs.clear();
    aabb_offsets.clear();
    aabbs.clear();
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        for(uint neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
            auto neighbor_rank = walberla::int_c(block->getNeighborProcess(neigh));

            // TODO: Make PBCs work with runtime load balancing
            // if(neighbor_rank != me) {
                const walberla::BlockID& neighbor_id = block->getNeighborId(neigh);
                walberla::math::AABB neighbor_aabb = block->getNeighborAABB(neigh);
                auto begin = blocks_pushed[neighbor_rank].begin();
                auto end = blocks_pushed[neighbor_rank].end();
                
                if(find_if(begin, end, [neighbor_id](const auto &bp) { return bp == neighbor_id; }) == end) {
                    neighborhood[neighbor_rank].push_back(neighbor_aabb);
                    blocks_pushed[neighbor_rank].push_back(neighbor_id);
                }
            // }
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
    void *src = name.compare("ranks") == 0          ? static_cast<void *>(ranks.data()) :
                name.compare("naabbs") == 0         ? static_cast<void *>(naabbs.data()) :
                name.compare("aabb_offsets") == 0   ? static_cast<void *>(aabb_offsets.data()) :
                name.compare("aabbs") == 0          ? static_cast<void *>(aabbs.data()) :
                name.compare("subdom") == 0         ? static_cast<void *>(subdom) : nullptr;

    PAIRS_ASSERT(src != nullptr);
    bool is_real = (name.compare("aabbs") == 0) || (name.compare("subdom") == 0);
    int tsize = is_real ? sizeof(real_t) : sizeof(int);
    std::memcpy(dest, src, size * tsize);
}

void BlockForest::updateWeights() {
    walberla::mpi::BufferSystem bs(mpiManager->comm(), 756);

    info->clear();

    int sum_block_locals = 0;
    // Compute the weights for my blocks and their children
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        auto aabb = block->getAABB();
        auto& block_info = (*info)[block->getId()];

        pairs::compute_boundary_weights(
            this->ps,
            aabb.xMin(), aabb.xMax(), aabb.yMin(), aabb.yMax(), aabb.zMin(), aabb.zMax(),
            &(block_info.computationalWeight), &(block_info.communicationWeight));
        
        sum_block_locals += block_info.computationalWeight;

        for(int branch = 0; branch < 8; ++branch) {
            const auto b_id = walberla::BlockID(block->getId(), branch);
            const auto b_aabb = forest->getAABBFromBlockId(b_id);
            auto& b_info = (*info)[b_id];

            pairs::compute_boundary_weights(
                this->ps,
                b_aabb.xMin(), b_aabb.xMax(), b_aabb.yMin(), b_aabb.yMax(), b_aabb.zMin(), b_aabb.zMax(),
                &(b_info.computationalWeight), &(b_info.communicationWeight));
        }
    }
    
    int non_globals = ps->getTrackedVariableAsInteger("nlocal") - UniqueID::getNumGlobals();
    
    if(sum_block_locals!=non_globals){
        std::cout << "Warning: " << non_globals - sum_block_locals << " particles in rank " << rank << 
        " may get lost in the next rebalancing." << std::endl;
    }

    // Send the weights of my blocks and their children to the neighbors of my blocks
    for(auto& iblock: *forest) {
        auto block = static_cast<walberla::blockforest::Block *>(&iblock);
        auto& block_info = (*info)[block->getId()];

        for(int neigh = 0; neigh < block->getNeighborhoodSize(); ++neigh) {
            bs.sendBuffer(block->getNeighborProcess(neigh)) <<
                walberla::blockforest::InfoCollection::value_type(block->getId(), block_info);
        }

        for(int branch = 0; branch < 8; ++branch) {
            const auto b_id = walberla::BlockID(block->getId(), branch);
            auto& b_info = (*info)[b_id];

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
            info->insert(val);
        }
    }
}

walberla::Vector3<int> BlockForest::getBlockConfig() {
    real_t area[3];
    real_t best_surf = 0.0;
    int ndims = 3;
    int d = 0;
    int nranks[3] = {1, 1, 1};

    for(int d1 = 0; d1 < ndims; d1++) {
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
    return walberla::Vector3<int>(nranks[0], nranks[1], nranks[2]);
}

int BlockForest::getInitialRefinementLevel(int num_processes) {
    int splitFactor = 8;
    int blocks = 1;
    int refinementLevel = 0;

    while(blocks < num_processes) {
        refinementLevel++;
        blocks *= splitFactor;
    }

    return refinementLevel;
}

void BlockForest::setBoundingBox() {
    for (int i=0; i<6; ++i) subdom[i] = 0.0;
    if (forest->empty()) return;

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

void BlockForest::initialize(int *argc, char ***argv) {
    mpiManager = walberla::mpi::MPIManager::instance();
    mpiManager->initializeMPI(argc, argv);
    mpiManager->useWorldComm();
    world_size = mpiManager->numProcesses();
    rank = mpiManager->rank();
    
    auto block_config = balance_workload ? walberla::Vector3<int>(1, 1, 1) : getBlockConfig();
    auto ref_level = balance_workload ? getInitialRefinementLevel(world_size) : 0;
    
    // PBC's are forced to true here and sperately handled when determining ghosts 
    walberla::Vector3<bool> pbc(true, true, true);
    walberla::math::AABB domain(grid_min[0], grid_min[1], grid_min[2], grid_max[0], grid_max[1], grid_max[2]);
    forest = walberla::blockforest::createBlockForest(domain, block_config, pbc, world_size, ref_level);

    this->info = make_shared<walberla::blockforest::InfoCollection>();

    if (rank==0) {
        std::cout << "Domain: " << domain << std::endl;
        std::cout << "PBC: " << pbc << std::endl;
        std::cout << "Block config: " << block_config  << std::endl;
        std::cout << "Initial refinement level: " << ref_level << std::endl;
        std::cout << "Dynamic load balancing: " << (balance_workload ? "True" : "False") << std::endl;
    }
}

void BlockForest::update() {
    if(balance_workload) {
        if(!forest->loadBalancingFunctionRegistered()){
            std::cerr << "Workload balancer is not initialized." << std::endl;
            exit(-1);
        }

        this->updateWeights();
        const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
        for(auto &prop: ps->getProperties()) {
            if(!prop.isVolatile()) {
                const int ptypesize = get_proptype_size(prop.getType());
                ps->copyPropertyToHost(prop, pairs::WriteAfterRead, nlocal*ptypesize);
            }
        }
        
        // PAIRS_DEBUG("Rebalance\n");
        if (rank==0) std::cout << "Rebalance" << std::endl;
        forest->refresh(); 
    }

    this->updateNeighborhood();
    this->setBoundingBox();
}

void BlockForest::initWorkloadBalancer(LoadBalancingAlgorithms algorithm, size_t regridMin, size_t regridMax) {
    if (rank==0) {
        std::cout << "Load balancing algorithm: " << getAlgorithmName(algorithm) << std::endl;
        std::cout << "regridMin = " << regridMin << ", regirdMax = " << regridMax << std::endl;
    }
    this->balance_workload = true;  // balance_workload is set to true in case the forest has been initialized externally
    real_t baseWeight = 1.0;
    int maxBlocksPerProcess = 100;

    // Metis-specific params
    real_t metisipc2redist = 1.0;
    string metisAlgorithm = "PART_GEOM_KWAY";
    string metisWeightsToUse = "BOTH_WEIGHTS";
    string metisEdgeSource = "EDGES_FROM_EDGE_WEIGHTS";

    forest->recalculateBlockLevelsInRefresh(true);
    forest->alwaysRebalanceInRefresh(true);
    forest->reevaluateMinTargetLevelsAfterForcedRefinement(true);
    forest->allowRefreshChangingDepth(true);

    forest->allowMultipleRefreshCycles(false);
    forest->checkForEarlyOutInRefresh(false);
    forest->checkForLateOutInRefresh(false);

    // TODO: Define another functor that makes use of communicationWeight as well
    forest->setRefreshMinTargetLevelDeterminationFunction(
        walberla::blockforest::MinMaxLevelDetermination(info, regridMin, regridMax));

    if(algorithm == Morton) {
        forest->setRefreshPhantomBlockDataAssignmentFunction(
            walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicCurveBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(false, true, false);
        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);

    } else if(algorithm == Hilbert) {
        forest->setRefreshPhantomBlockDataAssignmentFunction(
            walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicCurveBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(true, true, false);
        prepFunc.setMaxBlocksPerProcess(maxBlocksPerProcess);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);

    } else if(algorithm == Metis) {
        forest->setRefreshPhantomBlockDataAssignmentFunction(
            walberla::blockforest::MetisAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(
            walberla::blockforest::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(
            walberla::blockforest::MetisAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto alg = walberla::blockforest::DynamicParMetis::stringToAlgorithm(metisAlgorithm);
        auto vWeight = walberla::blockforest::DynamicParMetis::stringToWeightsToUse(metisWeightsToUse);
        auto eWeight = walberla::blockforest::DynamicParMetis::stringToEdgeSource(metisEdgeSource);
        auto prepFunc = walberla::blockforest::DynamicParMetis(alg, vWeight, eWeight);

        prepFunc.setipc2redist(metisipc2redist);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);

    } else if(algorithm == Diffusive) {
        forest->setRefreshPhantomBlockDataAssignmentFunction(
            walberla::blockforest::WeightAssignmentFunctor(info, baseWeight));
        forest->setRefreshPhantomBlockDataPackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());
        forest->setRefreshPhantomBlockDataUnpackFunction(
            walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeightPackUnpackFunctor());

        auto prepFunc = walberla::blockforest::DynamicDiffusionBalance<walberla::blockforest::WeightAssignmentFunctor::PhantomBlockWeight>(1, 1, false);
        forest->setRefreshPhantomBlockMigrationPreparationFunction(prepFunc);
    }
    else {
        std::cerr << "Invalid load balancing algorithm." << std::endl;
        exit(-1);
    }

    forest->addBlockData(make_shared<walberla::ParticleDataHandling>(ps), "Interface");
}

void BlockForest::finalize() {
    mpiManager->finalizeMPI();
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
    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    size_t nranks = 0;

    for(auto neigh_rank: ranks) {
        if(neigh_rank != rank) {
            MPI_Request send_req, recv_req;
            MPI_Irecv(&nrecv[nranks], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Isend(&nsend[nranks], 1, MPI_INT, neigh_rank, 0, MPI_COMM_WORLD, &send_req);
            send_requests.push_back(send_req);
            recv_requests.push_back(recv_req);
        } else {
            nrecv[nranks] = nsend[nranks];
        }
        nranks++;
    }

    if(!send_requests.empty()) {
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    }
    if(!recv_requests.empty()) {
        MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
    }
}

void BlockForest::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    std::vector<MPI_Request> send_requests;
    std::vector<MPI_Request> recv_requests;
    size_t nranks = 0;

    for(auto neigh_rank: ranks) {
        const real_t *send_ptr = &send_buf[send_offsets[nranks] * elem_size];
        real_t *recv_ptr = &recv_buf[recv_offsets[nranks] * elem_size];

        if(neigh_rank != rank) {
            MPI_Request send_req, recv_req;

            MPI_Irecv(recv_ptr, nrecv[nranks] * elem_size, MPI_DOUBLE, neigh_rank, 0, MPI_COMM_WORLD, &recv_req);
            MPI_Isend(send_ptr, nsend[nranks] * elem_size, MPI_DOUBLE, neigh_rank, 0, MPI_COMM_WORLD, &send_req);

            send_requests.push_back(send_req);
            recv_requests.push_back(recv_req);
        } else {
            pairs::copy_in_device(recv_ptr, send_ptr, nsend[nranks] * elem_size * sizeof(real_t));
        }

        nranks++;
    }

    if(!send_requests.empty()) {
        MPI_Waitall(send_requests.size(), send_requests.data(), MPI_STATUSES_IGNORE);
    }

    if(!recv_requests.empty()) {
        MPI_Waitall(recv_requests.size(), recv_requests.data(), MPI_STATUSES_IGNORE);
    }
}

void BlockForest::communicateDataReverse(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

        this->communicateData(dim, elem_size,send_buf, send_offsets, nsend, recv_buf, recv_offsets, nrecv);
}

void BlockForest::communicateAllData(
    int ndims, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    this->communicateData(0, elem_size, send_buf, send_offsets, nsend, recv_buf, recv_offsets, nrecv);
}

}
