#include <iostream>
//---
#include "dem_sd.hpp"

int main(int argc, char **argv) {
    PairsSimulation *ps = new PairsSimulation();

    // Create forest (make sure to use_domain(forest)) ----------------------------------------------
    walberla::math::AABB domain(0, 0, 0, 0.1, 0.1, 0.1);
    std::shared_ptr<walberla::mpi::MPIManager> mpiManager = walberla::mpi::MPIManager::instance();
    mpiManager->initializeMPI(&argc, &argv);
    mpiManager->useWorldComm();
    auto procs = mpiManager->numProcesses();
    auto block_config = walberla::Vector3<int>(2, 2, 1);
    auto ref_level = 0;
    std::shared_ptr<walberla::BlockForest> forest = walberla::blockforest::createBlockForest(
            domain, block_config, walberla::Vector3<bool>(true, true, false), procs, ref_level);
    //-----------------------------------------------------------------------------------------------

    // initialize pairs data structures ----------------------------------------------
    ps->initialize();

    // either create new domain or use an existing one ----------------------------------------
    // ps->create_domain(argc, argv);
    ps->use_domain(forest);

    // setup particles, setup functions, and the cell list stencil-------------------------------
    ps->setup_sim();

    for (int i=0; i<10000; ++i){
        ps->do_timestep(i);
    }

    ps->end();

    return 0;
}
