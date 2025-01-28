#include <iostream>
//---
#include "dem_sd.hpp"
#include <blockforest/BlockForest.h>
#include <blockforest/Initialization.h>

int main(int argc, char **argv) {
    double dt = 5e-5;
    auto pairs_sim = std::make_shared<PairsSimulation>();
    auto pairs_acc = std::make_shared<PairsAccessor>(pairs_sim);

    // Create forest (make sure to use_domain(forest)) ----------------------------------------------
    // walberla::math::AABB domain(0, 0, 0, 0.1, 0.1, 0.1);
    // std::shared_ptr<walberla::mpi::MPIManager> mpiManager = walberla::mpi::MPIManager::instance();
    // mpiManager->initializeMPI(&argc, &argv);
    // mpiManager->useWorldComm();
    // auto procs = mpiManager->numProcesses();
    // auto block_config = walberla::Vector3<int>(2, 2, 2);
    // auto ref_level = 0;
    // std::shared_ptr<walberla::BlockForest> forest = walberla::blockforest::createBlockForest(
    //         domain, block_config, walberla::Vector3<bool>(false, false, false), procs, ref_level);
    //-----------------------------------------------------------------------------------------------

    // initialize pairs data structures ----------------------------------------------
    pairs_sim->initialize();

    // either create new domain or use an existing one ----------------------------------------
    pairs_sim->create_domain(argc, argv);

    // pairs_sim->use_domain(forest);

    // create planes and particles ------------------------------------------------------------
    pairs_sim->create_halfspace(0,     0,      0,      1, 0, 0,        0, 13);
    pairs_sim->create_halfspace(0,     0,      0,      0, 1, 0,        0, 13);
    pairs_sim->create_halfspace(0,     0,      0,      0, 0, 1,        0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    -1, 0, 0,       0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    0, -1, 0,       0, 13);
    pairs_sim->create_halfspace(0.1,   0.1,    0.1,    0, 0, -1,       0, 13);

    // pairs::id_t pUid = pairs_sim->create_sphere(0.0499,   0.0499,   0.07,   0.5, 0.5, 0 ,   1000, 0.0045, 0, 0);
    // pairs::id_t pUid2 = pairs_sim->create_sphere(0.0499,   0.0499,   0.0499,   0.5, 0.5, 0 ,   1000, 0.0045, 0, 0);

    // Tracking a particle ------------------------------------------------------------------------
    // if (pUid != pairs_acc->getInvalidUid()){
    //     std::cout<< "Particle " << pUid << " is created in rank " << rank << std::endl;
    // }
    
    // walberla::mpi::allReduceInplace(pUid, walberla::mpi::SUM);
    // walberla::mpi::allReduceInplace(pUid2, walberla::mpi::SUM);
    // MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    // MPI_Allreduce(MPI_IN_PLACE, &pUid2, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);


    // if (pUid != pairs_acc->getInvalidUid()){
    //     std::cout<< "Particle " << pUid << " will be tracked by rank " << rank << std::endl;
    // }

    // auto pIsLocalInMyRank = [&](pairs::id_t uid){return pairs_acc->uidToIdx(uid) != pairs_acc->getInvalidIdx();};
    // auto pIsGhostInMyRank = [&](pairs::id_t uid){return pairs_acc->uidToIdxGhost(uid) != pairs_acc->getInvalidIdx();};


    // TODO: make sure linkedCellWidth is larger than max diameter in the system
    // setup particles, setup functions, and the cell list stencil-------------------------------
    pairs_sim->setup_sim();

    // for(int i=0; i<pairs_acc->size(); ++i){
    //     if(pairs_acc->getShape(i) == 0){
    //         std::cout<< "rank: " << rank << " sphere " <<  pairs_acc->getUid(i) << "  " << pairs_acc->getPosition(i) << std::endl; 
    //     }
    //     else if(pairs_acc->getShape(i) == 1){
    //         std::cout<< "rank: " << rank << " halfspace " <<  pairs_acc->getUid(i) << "  " << pairs_acc->getPosition(i) << pairs_acc->getNormal(i) << std::endl; 
    //     }
    // }

    // if (pIsLocalInMyRank(pUid)){
    //     int idx = pairs_acc->uidToIdx(pUid);
    //     pairs_acc->setPosition(idx, walberla::Vector3<double>(0.01, 0.01, 0.08));
    //     pairs_acc->setLinearVelocity(idx, walberla::Vector3<double>(0.5, 0.5, 0.5));
    // }

    for (int t=0; t<5000; ++t){
        // if ((t%200==0) && (pIsLocalInMyRank(pUid))){
        //     int idx = pairs_acc->uidToIdx(pUid);
        //     std::cout<< "Tracked particle is now in rank " << rank << " --- " << pairs_acc->getPosition(idx)<< std::endl;
        // }

        pairs_sim->communicate(t);

        // if(pairs_sim->rank() == 0){
        //     pairs_sim->pairs_runtime->copyPropertyToHost(0, ReadOnly, (((pairs_sim->pobj->nlocal + pairs_sim->pobj->nghost) * 1) * sizeof(unsigned long long int))); // uid
        //     for(int i=0; i<pairs_sim->pobj->nlocal; ++i){
        //         std::cout << "rank ##0## local uid[" << i << "] = " << pairs_sim->pobj->uid[i] << std::endl;
        //     }
        // }
        // if(pairs_sim->rank() == 2){
        //     pairs_sim->pairs_runtime->copyPropertyToHost(0, ReadOnly, (((pairs_sim->pobj->nlocal + pairs_sim->pobj->nghost) * 1) * sizeof(unsigned long long int))); // uid
        //     for(int i=0; i<pairs_sim->pobj->nlocal; ++i){
        //         std::cout << "rank 2 local uid[" << i << "] = " << pairs_sim->pobj->uid[i] << std::endl;
        //     }
        //     for(int i=pairs_sim->pobj->nlocal; i<pairs_sim->pobj->nlocal + pairs_sim->pobj->nghost; ++i){
        //         std::cout << "rank 2 ghost uid[" << i << "] = " << pairs_sim->pobj->uid[i] << std::endl;
        //     }
        // }

        // std::cout << pairs_sim->rank() << " --------- nlocal = " << pairs_acc->nlocal() << " nghost = " << pairs_acc->nghost() << std::endl;

        // if (pairs_sim->rank() == 2) {
        //     for(int i=0; i<pairs_acc->nlocal(); ++i){
        //         std::cout << pairs_sim->rank() << " Local  " << i << " -- uid= " << pairs_acc->getUid(i) << std::endl; 
        //     }
        //     for(int i=pairs_acc->nlocal(); i<pairs_acc->size(); ++i){
        //         std::cout << pairs_sim->rank() << " Ghost -- " << i << " -- uid= " << pairs_acc->getUid(i) << std::endl; 
        //     }
        // }



        // if (pIsLocalInMyRank(pUid)){
        //     int idx = pairs_acc->uidToIdx(pUid);
        //     pairs_acc->setHydrodynamicForce(idx, walberla::Vector3<double>(1, 1, 1));
        //     pairs_acc->setHydrodynamicTorque(idx, walberla::Vector3<double>(2, 2, 2));
        // }
        // if (pIsLocalInMyRank(pUid2)){
        //     int idx = pairs_acc->uidToIdx(pUid2);
        //     pairs_acc->setHydrodynamicForce(idx, walberla::Vector3<double>(10, 10, 10));
        //     pairs_acc->setHydrodynamicTorque(idx, walberla::Vector3<double>(20, 20, 20));
        // }
        
        pairs_sim->update_cells(); 
        pairs_sim->gravity(); 
        pairs_sim->linear_spring_dashpot(); 
        pairs_sim->euler(dt); 

        // std::cout << "reset_volatiles" << std::endl;
        pairs_sim->reset_volatiles(); 

        pairs_sim->vtk_write("output/dem_sd_local", 0, pairs_acc->nlocal(), t, 200);
        pairs_sim->vtk_write("output/dem_sd_ghost", pairs_acc->nlocal(), pairs_acc->size(), t, 200);

        // if (pIsGhostInMyRank(pUid)){
        //     std::cout << pairs_sim->rank() << " - ghost 1: " << pUid << std::endl;
        //     int idx = pairs_acc->uidToIdxGhost(pUid);
        //     pairs_acc->setHydrodynamicForce(idx, walberla::Vector3<double>(1, 1, 1));
        //     pairs_acc->setHydrodynamicTorque(idx, walberla::Vector3<double>(3, 3, 3));
        // }
        // if (pIsGhostInMyRank(pUid2)){
        //     std::cout << pairs_sim->rank() << " - ghost 2: " << pUid2 << std::endl;
        //     int idx = pairs_acc->uidToIdxGhost(pUid2);
        //     pairs_acc->setHydrodynamicForce(idx, walberla::Vector3<double>(2, 2, 2));
        //     pairs_acc->setHydrodynamicTorque(idx, walberla::Vector3<double>(4, 4, 4));
        // }

        // std::cout << "reverse comm and reduce" << std::endl;
        // pairs_sim->reverse_comm();

        // if (pIsLocalInMyRank(pUid)){
        //     int idx = pairs_acc->uidToIdx(pUid);
        //     auto forceSum = pairs_acc->getHydrodynamicForce(idx);
        //     auto torqueSum = pairs_acc->getHydrodynamicTorque(idx);
        //     std::cout << pUid << " @@@@ reduced force = " << forceSum << std::endl;
        //     std::cout << pUid << " @@@@ reduced torque = " << torqueSum << std::endl;
        // }
        // if (pIsLocalInMyRank(pUid2)){
        //     int idx = pairs_acc->uidToIdx(pUid2);
        //     auto forceSum = pairs_acc->getHydrodynamicForce(idx);
        //     auto torqueSum = pairs_acc->getHydrodynamicTorque(idx);
        //     std::cout << pUid2 << " @@@@ reduced force = " << forceSum << std::endl;
        //     std::cout << pUid2 << " @@@@ reduced torque = " << torqueSum << std::endl;
        // }
    }

    pairs_sim->end();

    return 0;
}
