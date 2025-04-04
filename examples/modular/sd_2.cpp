#include <iostream>
#include <memory>

#include <blockforest/BlockForest.h>
#include <blockforest/Initialization.h>

#include "spring_dashpot.hpp"

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    // Create forest
    // -------------------------------------------------------------------------------
    walberla::math::AABB domain(0, 0, 0, 1, 1, 1);
    std::shared_ptr<walberla::mpi::MPIManager> mpiManager = walberla::mpi::MPIManager::instance();
    mpiManager->initializeMPI(&argc, &argv);
    mpiManager->useWorldComm();
    auto procs = mpiManager->numProcesses();

    walberla::Vector3<int> block_config;
    if (procs==1)        block_config = walberla::Vector3<int>(1, 1, 1);
    else if (procs==4)   block_config = walberla::Vector3<int>(2, 2, 1);
    else { std::cout << "Error: Check block_config" << std::endl; exit(-1);} 

    auto ref_level = 0;
    std::shared_ptr<walberla::BlockForest> forest = walberla::blockforest::createBlockForest(
            domain, block_config, walberla::Vector3<bool>(false, false, false), procs, ref_level);

    // Pass forest to P4IRS
    // -------------------------------------------------------------------------------
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    pairs_runtime->useDomain(forest);

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  -1, 0, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, -1, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, 0, -1,    0, 13);
    pairs::create_sphere(pairs_runtime, 0.6, 0.6, 0.7,      -2, -2, 0,  1000, 0.05, 0, 0);
    pairs::create_sphere(pairs_runtime, 0.4, 0.4, 0.68,    2, 2, 0,    1000, 0.05, 0, 0);

    pairs_sim->update_mass_and_inertia();

    pairs_sim->setCellWidth(0.1, 0.1, 0.1);
    pairs_sim->setInteractionRadius(0.1);
    pairs_sim->updateDomain();

    int num_timesteps = 2000;
    int vtk_freq = 20;
    double dt = 1e-3;

    for (int t=0; t<num_timesteps; ++t){
        if ((t%500==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;

        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 
        pairs_sim->euler(dt); 
        pairs_sim->reneighbor();

        pairs::vtk_write_data(pairs_runtime, "output/sd_2_local", 0, pairs_sim->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/sd_2_ghost", pairs_sim->nlocal(), pairs_sim->size(), t, vtk_freq);
    }

    pairs_sim->end();
}
