#include <iostream>
#include "reduction_example.hpp"

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    ParticleAccessor ac(pairs_sim.get());
    
    // Set domain
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 0.1, 0.1, 0.1);

    // Create bodies
    pairs::id_t pUid = pairs::create_sphere(pairs_runtime, 0.0499,   0.0499,   0.07,   0.5, 0.5, 0 ,   1000, 0.0045, 0, 0);
    
    pairs_sim->update_mass_and_inertia();
    
    // updateDomain after creating all bodies
    pairs_sim->setCellWidth(0.01, 0.01, 0.01);
    pairs_sim->setInteractionRadius(0.01);
    pairs_sim->updateDomain();
    ac.update();

    // Track particle
    //-------------------------------------------------------------------------------------------
    if (pUid != ac.getInvalidUid()){
        std::cout<< "Particle " << pUid << " is created in rank " << pairs_sim->rank() << std::endl;
    }

    MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    if (pUid != ac.getInvalidUid()){
        std::cout<< "Particle " << pUid << " will be tracked by rank " << pairs_sim->rank() << std::endl;
    }
        
    // Helper lambdas for demo
    //-------------------------------------------------------------------------------------------
    auto pIsLocalInMyRank = [&](pairs::id_t uid){return ac.uidToIdxLocal(uid) != ac.getInvalidIdx();};
    auto pIsGhostInMyRank = [&](pairs::id_t uid){return ac.uidToIdxGhost(uid) != ac.getInvalidIdx();};

    // Check which rank owns the particle, and which ranks have it as a ghost
    //-------------------------------------------------------------------------------------------
    ac.syncUid(ParticleAccessor::Host);
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
        ac.syncUid(ParticleAccessor::Host);

        // Add local contribution
        //-------------------------------------------------------------------------------------------
        if (pIsLocalInMyRank(pUid)){
            int idx = ac.uidToIdxLocal(pUid);
            pairs::Vector3<double> local_force(0.1, 0.1, 0.1);
            pairs::Vector3<double> local_torque(0.2, 0.2, 0.2);

            std::cout << "Force on particle " << pUid << " from local rank [" << pairs_sim->rank() << "] : (" 
                        << local_force[0] << ", " << local_force[1] << ", " << local_force[2] << ")" <<  std::endl;

            ac.setHydrodynamicForce(idx, local_force);
            ac.setHydrodynamicTorque(idx, local_torque);
            ac.syncHydrodynamicForce(ParticleAccessor::Host, true);
            ac.syncHydrodynamicTorque(ParticleAccessor::Host, true);
        }

        // Add neighbor contributions
        //-------------------------------------------------------------------------------------------
        if (pIsGhostInMyRank(pUid)){
            int idx = ac.uidToIdxGhost(pUid);
            pairs::Vector3<double> ghost_force(pairs_sim->rank()*10, 1, 1);
            pairs::Vector3<double> ghost_torque(pairs_sim->rank()*20, 2, 2);

            std::cout << "Force on particle " << pUid << " from neighbor rank [" << pairs_sim->rank() << "] : (" 
                        << ghost_force[0] << ", " << ghost_force[1] << ", " << ghost_force[2] << ")" <<  std::endl;

            ac.setHydrodynamicForce(idx, ghost_force);
            ac.setHydrodynamicTorque(idx, ghost_torque);
            ac.syncHydrodynamicForce(ParticleAccessor::Host, true);
            ac.syncHydrodynamicTorque(ParticleAccessor::Host, true);
        }
        
        // Do computations
        //-------------------------------------------------------------------------------------------
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot();
        pairs_sim->euler(5e-5);        
        //-------------------------------------------------------------------------------------------

        std::cout << "Reverse communicate and reduce." << std::endl;
        // Communicate ghost particle data back to their owner ranks and reduce
        pairs_sim->reduceGhosts();  

        // Get the reduced force on the owner rank
        //-------------------------------------------------------------------------------------------
        if (pIsLocalInMyRank(pUid)){
            int idx = ac.uidToIdxLocal(pUid);
            ac.syncHydrodynamicForce(ParticleAccessor::Host);
            ac.syncHydrodynamicTorque(ParticleAccessor::Host);
            auto force_sum = ac.getHydrodynamicForce(idx);
            // auto torque_sum = ac.getHydrodynamicTorque(idx);

            std::cout << "Reduced force on particle " << pUid << " in local rank [" << pairs_sim->rank() << "] : (" 
                        << force_sum[0] << ", " << force_sum[1] << ", " << force_sum[2] << ")" <<  std::endl;
        }
        
        // Forward communication 
        //-------------------------------------------------------------------------------------------
        pairs_sim->reneighbor();
        ac.update();
    }

    ac.end();
}
