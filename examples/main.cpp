#include <iostream>
//---
#include "dem_sd.hpp"
#include "core/mpi/Broadcast.h"

int main(int argc, char **argv) {
    auto pairs_sim = std::make_shared<PairsSimulation>();
    auto pairs_acc = std::make_shared<PairsAccessor>(pairs_sim);

    // Create forest (make sure to use_domain(forest)) ----------------------------------------------
    walberla::math::AABB domain(0, 0, 0, 0.1, 0.1, 0.1);
    std::shared_ptr<walberla::mpi::MPIManager> mpiManager = walberla::mpi::MPIManager::instance();
    mpiManager->initializeMPI(&argc, &argv);
    mpiManager->useWorldComm();
    auto rank = mpiManager->rank();
    auto procs = mpiManager->numProcesses();
    auto block_config = walberla::Vector3<int>(2, 2, 1);
    auto ref_level = 0;
    std::shared_ptr<walberla::BlockForest> forest = walberla::blockforest::createBlockForest(
            domain, block_config, walberla::Vector3<bool>(false, false, false), procs, ref_level);
    //-----------------------------------------------------------------------------------------------

    // initialize pairs data structures ----------------------------------------------
    pairs_sim->initialize();

    // either create new domain or use an existing one ----------------------------------------
    // pairs_sim->create_domain(argc, argv);
    pairs_sim->use_domain(forest);

    // create planes and particles ------------------------------------------------------------
    pairs_sim->create_halfspace(0,     0,      0,      1, 0, 0,        0, 13);
    pairs_sim->create_halfspace(0,     0,      0,      0, 1, 0,        0, 13);
    pairs_sim->create_halfspace(0,     0,      0,      0, 0, 1,        0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    -1, 0, 0,       0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    0, -1, 0,       0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    0, 0, -1,       0, 13);

    pairs::id_t pUid = pairs_sim->create_sphere(0.03,   0.03,   0.08,   0.5, 0.5, 0 ,   1000, 0.0045, 0, 0);
    pairs_sim->create_sphere(0.07,   0.07,   0.08,   -0.5, -0.5, 0 , 1000, 0.0045, 0, 0);

    // Tracking a particle ------------------------------------------------------------------------
    if (pUid != pairs_acc->getInvalidUid()){
        std::cout<< "Particle " << pUid << " is created in rank " << rank << std::endl;
    }
    
    walberla::mpi::allReduceInplace(pUid, walberla::mpi::SUM);

    if (pUid != pairs_acc->getInvalidUid()){
        std::cout<< "Particle " << pUid << " will be tracked by rank " << rank << std::endl;
    }

    auto pIsInMyRank = [&](pairs::id_t uid){return pairs_acc->uidToIdx(uid) != pairs_acc->getInvalidIdx();};


    // TODO: make sure linkedCellWidth is larger than max diameter in the system
    // setup particles, setup functions, and the cell list stencil-------------------------------
    pairs_sim->setup_sim();

    for(int i=0; i<pairs_acc->size(); ++i){
        if(pairs_acc->getShape(i) == 0){
            std::cout<< "rank: " << rank << " sphere " <<  pairs_acc->getUid(i) << "  " << pairs_acc->getPosition(i) << std::endl; 
        }
        else if(pairs_acc->getShape(i) == 1){
            std::cout<< "rank: " << rank << " halfspace " <<  pairs_acc->getUid(i) << "  " << pairs_acc->getPosition(i) << pairs_acc->getNormal(i) << std::endl; 
        }
    }

    if (pIsInMyRank(pUid)){
        int idx = pairs_acc->uidToIdx(pUid);
        pairs_acc->setPosition(idx, walberla::Vector3<double>(0.01, 0.01, 0.08));
        pairs_acc->setLinearVelocity(idx, walberla::Vector3<double>(0.5, 0.5, 0.5));
    }

    for (int i=0; i<10000; ++i){
        if ((i%200==0) && (pIsInMyRank(pUid))){
            int idx = pairs_acc->uidToIdx(pUid);
            std::cout<< "Tracked particle is now in rank " << rank << " --- " << pairs_acc->getPosition(idx)<< std::endl;
        }

        pairs_sim->do_timestep(i);
    }

    pairs_sim->end();

    return 0;
}
