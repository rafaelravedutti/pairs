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
#include "../pairs_common.hpp"
#include "../devices/device.hpp"
#include "regular_6d_stencil.hpp"
#include "MDDataHandling.h"

namespace pairs {

void BlockForest::updateNeighborhood(
    std::shared_ptr<BlockForest> forest,
    blockforest::InfoCollection& info,
    real_t *ranks,
    real_t *naabbs,
    real_t *aabbs) {

    auto me = mpi::MPIManager::instance()->rank();
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

    vec_ranks.clear();
    vec_naabbs.clear();
    vec_aabbs.clear();
    total_aabbs = 0;

    for(auto& nbh: neighborhood) {
        auto rank = nbh.first;
        auto aabb_list = nbh.second;
        vec_ranks.push_back((int) rank);
        vec_naabbs.push_back((int) aabb_list.size());

        for(auto &aabb: aabb_list) {
            vec_aabbs.push_back(aabb.xMin());
            vec_aabbs.push_back(aabb.xMax());
            vec_aabbs.push_back(aabb.yMin());
            vec_aabbs.push_back(aabb.yMax());
            vec_aabbs.push_back(aabb.zMin());
            vec_aabbs.push_back(aabb.zMax());
            total_aabbs++;
        }
    }

    *nranks = nranks;
    if(nranks > *rank_capacity) {
        // reallocateArray?
        const int new_capacity = nranks + 10;
        delete[] ranks;
        delete[] naabbs;
        delete[] offsets;
        ranks = new int[new_capacity];
        naabbs = new int[new_capacity];
        offsets = new int[new_capacity];
        *rank_capacity = new_capacity;
    }    

    if(total_aabbs > *aabb_capacity) {
        const int new_capacity = total_aabbs + 10;
        aabbs = new real_t[new_capacity * 6];
        *aabb_capacity = new_capacity;
    }

    int offset = 0;
    for(int i = 0; i < nranks; i++) {
        ranks[i] = vec_ranks.data()[i];
        naabbs[i] = vec_naabbs.data()[i];
        offsets[i] = offset;
        offset += naabbs[i];
    }

    for(int i = 0; i < total_aabbs * 6; i++) {
        aabbs[i] = vec_aabbs.data()[i];
    }

    ps->copyToDevice(aabbs);
}

/*
  extern fn md_compute_boundary_weights(
    xmin: real_t, xmax: real_t, ymin: real_t, ymax: real_t, zmin: real_t, zmax: real_t,
    computational_weight: &mut u32, communication_weight: &mut u32) -> () {

    let grid = grid_;
    let particle = make_particle(grid, array_dev, ParticleDataLayout(), null_layout());
    let sum = @|a: i32, b: i32| { a + b };
    let aabb = AABB {
        xmin: xmin,
        xmax: xmax,
        ymin: ymin,
        ymax: ymax,
        zmin: zmin,
        zmax: zmax
    };

    *computational_weight = reduce_i32(grid.nparticles, 0, sum, |i| {
        select(is_within_domain(particle.get_position(i), aabb), 1, 0)
    }) as u32;

    *communication_weight = reduce_i32(grid.nghost, 0, sum, |i| {
        select(is_within_domain(particle.get_position(grid.nparticles + i), aabb), 1, 0)
 }) as u32;
 */

void BlockForest::updateWeights(
    shared_ptr<BlockForest> forest,
    blockforest::InfoCollection& info,
    real_t *position,
    int nparticles) {

    mpi::BufferSystem bs(mpi::MPIManager::instance()->comm(), 756);

    info.clear();
    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
        auto aabb = block->getAABB();
        auto& block_info = info[block->getId()];

        pairs->callModule(
            "computeBoundaryWeights",
            aabb.xMin(), aabb.xMax(), aabb.yMin(), aabb.yMax(), aabb.zMin(), aabb.zMax(),
            &(block_info.computationalWeight), &(block_info.communicationWeight));

        for(uint_t branch = 0; branch < 8; ++branch) {
            const auto b_id = BlockID(block->getId(), branch);
            const auto b_aabb = forest->getAABBFromBlockId(b_id);
            auto& b_info = info[b_id];

            pairs->callModule(
                "computeBoundaryWeights",
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

void BlockForest::getBlockForestAABB(double (&rank_aabb)[6]) {
    auto aabb_union = forest->begin()->getAABB();

    for(auto& iblock: *forest) {
        auto block = static_cast<blockforest::Block *>(&iblock);
        aabb_union.merge(block->getAABB());
    }

    rank_aabb[0] = aabb_union.xMin();
    rank_aabb[1] = aabb_union.xMax();
    rank_aabb[2] = aabb_union.yMin();
    rank_aabb[3] = aabb_union.yMax();
    rank_aabb[4] = aabb_union.zMin();
    rank_aabb[5] = aabb_union.zMax();
}

void BlockForest::setConfig() {
    auto mpiManager = mpi::MPIManager::instance();
    mpiManager->initializeMPI(&argc, &argv);
    mpiManager->useWorldComm();

    math::AABB domain(xmin, ymin, zmin, xmax, ymax, zmax);
    int gridsize[3] = {32, 32, 32};
    auto procs = mpiManager->numProcesses();
    auto block_config = use_load_balancing ? Vector3<uint_t>(1, 1, 1) : getBlockConfig(procs, gridsize[0], gridsize[1], gridsize[2]);
    auto ref_level = use_load_balancing ? getInitialRefinementLevel(procs) : 0;

    auto forest = blockforest::createBlockForest(
        domain, block_config, Vector3<bool>(true, true, true), procs, ref_level);

    auto is_within_domain = bind(isWithinBlockForest, _1, _2, _3, forest);
    auto info = make_shared<blockforest::InfoCollection>();
    getBlockForestAABB(forest, rank_aabb);
}

void BlockForest::setBoundingBox() {
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
        subdom_min[d] = this->grid_min[d] + rank_length[d] * (real_t)myloc[d];
        subdom_max[d] = subdom_min[d] + rank_length[d];
    }

    delete[] myloc;
    delete[] periods;
    delete[] rank_length;
    MPI_Comm_free(&cartesian);
}

void BlockForest::initialize(int *argc, char ***argv) {
    MPI_Init(argc, argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    this->setConfig();
    this->setBoundingBox();
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

void BlockForest::fillArrays(int *neighbor_ranks, int *pbc, real_t *subdom) {
    for(int d = 0; d < ndims; d++) {
        neighbor_ranks[d * 2 + 0] = prev[d];
        neighbor_ranks[d * 2 + 1] = next[d];
        pbc[d * 2 + 0] = pbc_prev[d];
        pbc[d * 2 + 1] = pbc_next[d];
        subdom[d * 2 + 0] = subdom_min[d];
        subdom[d * 2 + 1] = subdom_max[d];
    }
}

void BlockForest::communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
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

void BlockForest::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    //MPI_Request recv_requests[2];
    //MPI_Request send_requests[2];
    const real_t *send_prev = &send_buf[send_offsets[dim * 2 + 0] * elem_size];
    const real_t *send_next = &send_buf[send_offsets[dim * 2 + 1] * elem_size];
    real_t *recv_prev = &recv_buf[recv_offsets[dim * 2 + 0] * elem_size];
    real_t *recv_next = &recv_buf[recv_offsets[dim * 2 + 1] * elem_size];

    if(prev[dim] != rank) {
        MPI_Sendrecv(
            send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0,
            recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, next[dim], 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /*
        MPI_Irecv(
            recv_prev, nrecv[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0,
            MPI_COMM_WORLD, &recv_requests[0]);

        MPI_Isend(
            send_prev, nsend[dim * 2 + 0] * elem_size, MPI_DOUBLE, prev[dim], 0,
            MPI_COMM_WORLD, &send_requests[0]);
        */
    } else {
        pairs::copy_in_device(recv_prev, send_prev, nsend[dim * 2 + 0] * elem_size * sizeof(real_t));
    }

    if(next[dim] != rank) {
        MPI_Sendrecv(
            send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0,
            recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, prev[dim], 0,
            MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /*
        MPI_Irecv(
            recv_next, nrecv[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0,
            MPI_COMM_WORLD, &recv_requests[1]);

        MPI_Isend(
            send_next, nsend[dim * 2 + 1] * elem_size, MPI_DOUBLE, next[dim], 0,
            MPI_COMM_WORLD, &send_requests[1]);
        */
    } else {
        pairs::copy_in_device(recv_next, send_next, nsend[dim * 2 + 1] * elem_size * sizeof(real_t));
    }

    //MPI_Waitall(2, recv_requests, MPI_STATUSES_IGNORE);
    //MPI_Waitall(2, send_requests, MPI_STATUSES_IGNORE);
}

void BlockForest::communicateAllData(
    int ndims, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    //std::vector<MPI_Request> send_requests(ndims * 2, MPI_REQUEST_NULL);
    //std::vector<MPI_Request> recv_requests(ndims * 2, MPI_REQUEST_NULL);

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

            /*
            MPI_Isend(
                send_prev, nsend[d * 2 + 0] * elem_size, MPI_DOUBLE, prev[d], 0,
                MPI_COMM_WORLD, &send_requests[d * 2 + 0]);

            MPI_Irecv(
                recv_prev, nrecv[d * 2 + 0] * elem_size, MPI_DOUBLE, prev[d], 0,
                MPI_COMM_WORLD, &recv_requests[d * 2 + 0]);
            */
        } else {
            pairs::copy_in_device(recv_prev, send_prev, nsend[d * 2 + 0] * elem_size * sizeof(real_t));
        }

        if (next[d] != rank) {
            MPI_Sendrecv(
                send_next, nsend[d * 2 + 1] * elem_size, MPI_DOUBLE, next[d], 0,
                recv_next, nrecv[d * 2 + 1] * elem_size, MPI_DOUBLE, prev[d], 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            /*
            MPI_Isend(
                send_next, nsend[d * 2 + 1] * elem_size, MPI_DOUBLE, next[d], 0,
                MPI_COMM_WORLD, &send_requests[d * 2 + 1]);

            MPI_Irecv(
                recv_next, nrecv[d * 2 + 1] * elem_size, MPI_DOUBLE, next[d], 0,
                MPI_COMM_WORLD, &recv_requests[d * 2 + 1]);
            */
        } else {
            pairs::copy_in_device(recv_next, send_next, nsend[d * 2 + 1] * elem_size * sizeof(real_t));
        }
    }

    //MPI_Waitall(ndims * 2, send_requests.data(), MPI_STATUSES_IGNORE);
    //MPI_Waitall(ndims * 2, recv_requests.data(), MPI_STATUSES_IGNORE);
}

}
