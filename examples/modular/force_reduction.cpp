#include <iostream>
//---
#include "force_reduction.hpp"

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();    
    pairs_sim->initialize();
    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());
    
    // Set domain
    pairs_sim->set_domain(argc, argv, 0, 0, 0, 0.1, 0.1, 0.1);

    // Create bodies
    pairs::id_t pUid = pairs_sim->create_sphere(0.0499,   0.0499,   0.07,   0.5, 0.5, 0 ,   1000, 0.0045, 0, 0);
    
    // setup_sim after creating all bodies
    pairs_sim->setup_sim();
    pairs_sim->update_mass_and_inertia();

    // Track particle
    //-------------------------------------------------------------------------------------------
    if (pUid != ac->getInvalidUid()){
        std::cout<< "Particle " << pUid << " is created in rank " << pairs_sim->rank() << std::endl;
    }

    MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    if (pUid != ac->getInvalidUid()){
        std::cout<< "Particle " << pUid << " will be tracked by rank " << pairs_sim->rank() << std::endl;
    }

    // Communicate particles (exchange/ghost)
    //-------------------------------------------------------------------------------------------
    pairs_sim->communicate(0);
    ac->update();
        
    // Helper lambdas for demo
    //-------------------------------------------------------------------------------------------
    auto pIsLocalInMyRank = [&](pairs::id_t uid){return ac->uidToIdxLocal(uid) != ac->getInvalidIdx();};
    auto pIsGhostInMyRank = [&](pairs::id_t uid){return ac->uidToIdxGhost(uid) != ac->getInvalidIdx();};

    // Check which rank owns the particle, and which ranks have it as a ghost
    //-------------------------------------------------------------------------------------------
    ac->syncUid(PairsAccessor::Host);
    if (pIsLocalInMyRank(pUid)){
        std::cout<< "Particle " << pUid << " is local in rank " << pairs_sim->rank() << std::endl;
    }
    if (pIsGhostInMyRank(pUid)){
        std::cout<< "Particle " << pUid << " is ghost in rank " << pairs_sim->rank() << std::endl;
    }

    // Start timestep loop
    //-------------------------------------------------------------------------------------------
    int num_timesteps = 1;
    for (int t=0; t<num_timesteps; ++t){
        ac->syncUid(PairsAccessor::Host);

        // Add local contribution
        //-------------------------------------------------------------------------------------------
        if (pIsLocalInMyRank(pUid)){
            int idx = ac->uidToIdxLocal(pUid);
            pairs::Vector3<double> local_force(0.1, 0.1, 0.1);
            pairs::Vector3<double> local_torque(0.2, 0.2, 0.2);

            std::cout << "Force on particle " << pUid << " from local rank [" << pairs_sim->rank() << "] : (" 
                        << local_force[0] << ", " << local_force[1] << ", " << local_force[2] << ")" <<  std::endl;

            ac->setHydrodynamicForce(idx, local_force);
            ac->setHydrodynamicTorque(idx, local_torque);
            ac->syncHydrodynamicForce(PairsAccessor::Host, true);
            ac->syncHydrodynamicTorque(PairsAccessor::Host, true);
        }

        // Add neighbor contributions
        //-------------------------------------------------------------------------------------------
        if (pIsGhostInMyRank(pUid)){
            int idx = ac->uidToIdxGhost(pUid);
            pairs::Vector3<double> ghost_force(pairs_sim->rank()*10, 1, 1);
            pairs::Vector3<double> ghost_torque(pairs_sim->rank()*20, 2, 2);

            std::cout << "Force on particle " << pUid << " from neighbor rank [" << pairs_sim->rank() << "] : (" 
                        << ghost_force[0] << ", " << ghost_force[1] << ", " << ghost_force[2] << ")" <<  std::endl;

            ac->setHydrodynamicForce(idx, ghost_force);
            ac->setHydrodynamicTorque(idx, ghost_torque);
            ac->syncHydrodynamicForce(PairsAccessor::Host, true);
            ac->syncHydrodynamicTorque(PairsAccessor::Host, true);
        }
        
        // Do computations
        //-------------------------------------------------------------------------------------------
        pairs_sim->update_cells(t); 
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot();
        pairs_sim->euler(5e-5); 
        pairs_sim->reset_volatiles();         
        //-------------------------------------------------------------------------------------------

        std::cout << "---- reverse_comm and reduce ----" << std::endl;
        // reverse_comm() communicates data from ghost particles back to their owner ranks using
        // information from the previous time that communicate() was called 
        pairs_sim->reverse_comm();  

        // Get the reduced force on the owner rank
        //-------------------------------------------------------------------------------------------
        if (pIsLocalInMyRank(pUid)){
            int idx = ac->uidToIdxLocal(pUid);
            ac->syncHydrodynamicForce(PairsAccessor::Host);
            ac->syncHydrodynamicTorque(PairsAccessor::Host);
            auto force_sum = ac->getHydrodynamicForce(idx);
            auto torque_sum = ac->getHydrodynamicTorque(idx);

            std::cout << "Reduced force on particle " << pUid << " in local rank [" << pairs_sim->rank() << "] : (" 
                        << force_sum[0] << ", " << force_sum[1] << ", " << force_sum[2] << ")" <<  std::endl;
        }
        
        // Usual communication 
        //-------------------------------------------------------------------------------------------
        pairs_sim->communicate(t);
        ac->update();
    }

    pairs_sim->end();
}
