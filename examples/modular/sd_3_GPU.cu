#include <iostream>
#include <memory>
#include <cuda_runtime.h>

#include "spring_dashpot.hpp"

void checkCudaError(cudaError_t err, const char* func) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", func, cudaGetErrorString(err));
        exit(err);
    }
}

__global__ void print_position(PairsAccessor ac, int idx){
    printf("Position [from device] = (%f, %f, %f) \n", ac.getPosition(idx)[0], ac.getPosition(idx)[1], ac.getPosition(idx)[2]);
}

__global__ void change_gravitational_force(PairsAccessor ac, int idx){
    printf("Force [from device] before setting = (%f, %f, %f) \n", ac.getForce(idx)[0], ac.getForce(idx)[1], ac.getForce(idx)[2]);

    pairs::Vector3<double> upward_gravity(0.0, 0.0, 2 * ac.getMass(idx) * 9.81); 
    ac.setForce(idx, ac.getForce(idx) + upward_gravity);

    printf("Force [from device] after setting = (%f, %f, %f) \n", ac.getForce(idx)[0], ac.getForce(idx)[1], ac.getForce(idx)[2]);
}

void set_feature_properties(std::shared_ptr<PairsAccessor> &ac){
    ac->setTypeStiffness(0,0, 0);
    ac->setTypeStiffness(0,1, 1000);
    ac->setTypeStiffness(1,0, 1000);
    ac->setTypeStiffness(1,1, 3000);
    ac->syncTypeStiffness();

    ac->setTypeDampingNorm(0,0, 0);
    ac->setTypeDampingNorm(0,1, 20);
    ac->setTypeDampingNorm(1,0, 20);
    ac->setTypeDampingNorm(1,1, 10);
    ac->syncTypeDampingNorm();
}

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    // Create PairsAccessor after PairsSimulation is initialized
    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());

    auto pairs_runtime = pairs_sim->getPairsRuntime();
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 1, 1, 1);

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  -1, 0, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, -1, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, 0, -1,    0, 13);

    pairs::id_t pUid = pairs::create_sphere(pairs_runtime, 0.6, 0.6, 0.7,      0, 0, 0,  1000, 0.05, 1, 0);
    pairs::create_sphere(pairs_runtime, 0.4, 0.4, 0.76,    2, 2, 0,    1000, 0.05, 1, 0);

    set_feature_properties(ac);

    MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    auto pIsLocalInMyRank = [&](pairs::id_t uid){return ac->uidToIdxLocal(uid) != ac->getInvalidIdx();};
    
    pairs_sim->update_mass_and_inertia();

    pairs_sim->setCellWidth(0.1, 0.1, 0.1);
    pairs_sim->setInteractionRadius(0.1);
    pairs_sim->updateDomain();

    // PairsAccessor requires an update when particles are communicated 
    ac->update();

    int num_timesteps = 2000;
    int vtk_freq = 20;
    double dt = 1e-3;

    for (int t=0; t<num_timesteps; ++t){
        // Up-to-date uids might be on host or device. So sync uid in Host before accessing them from host
        ac->syncUid(PairsAccessor::Host);

        // Print position of particle pUid
        //-------------------------------------------------------------------------------------------
        if(pIsLocalInMyRank(pUid)){
            std::cout << "Timestep (" << t << "): Particle " << pUid << " is in rank " << pairs_sim->rank() << std::endl;
            int idx = ac->uidToIdxLocal(pUid);

            // Up-to-date position might be on host or device. 
            // Sync position on Host before reading it from host:
            ac->syncPosition(PairsAccessor::Host); 
            std::cout << "Position [from host] = (" 
                    << ac->getPosition(idx)[0] << ", "
                    << ac->getPosition(idx)[1] << ", " 
                    << ac->getPosition(idx)[2] << ")" << std::endl;
            
            // Sync position on Device before reading it from device:
            ac->syncPosition(PairsAccessor::Device); 
            print_position<<<1,1>>>(*ac, idx);
            checkCudaError(cudaDeviceSynchronize(), "print_position");
            
            // There's no need to sync position here to continue the simulation, since position wasn't modified.
        }

        // Calculate forces
        //-------------------------------------------------------------------------------------------
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 

        // Change gravitational force on particle pUid
        //-------------------------------------------------------------------------------------------
        ac->syncUid(PairsAccessor::Host);

        if(pIsLocalInMyRank(pUid)){
            std::cout << "Force Timestep (" << t << "): Particle " << pUid << " is in rank " << pairs_sim->rank() << std::endl;
            int idx = ac->uidToIdxLocal(pUid);

            // Up-to-date force and mass might be on host or device. 
            // So sync them in Device before accessing them on device. (No data will be transfered if they are already on device)
            ac->syncForce(PairsAccessor::Device);
            ac->syncMass(PairsAccessor::Device);

            // Modify force from device:
            change_gravitational_force<<<1,1>>>(*ac, idx);
            checkCudaError(cudaDeviceSynchronize(), "change_gravitational_force");

            // Force on device was modified.
            // So sync force before continuing the simulation.
            ac->syncForce(PairsAccessor::Host);
            std::cout << "Force [from host] after changing = (" 
                    << ac->getForce(idx)[0] << ", "
                    << ac->getForce(idx)[1] << ", " 
                    << ac->getForce(idx)[2] << ")" << std::endl;
        }

        // Euler
        //-------------------------------------------------------------------------------------------
        pairs_sim->euler(dt);

        // Reneighbor
        //-------------------------------------------------------------------------------------------
        pairs_sim->reneighbor();
        // PairsAccessor requires an update when particles are reneighbored
        ac->update();

        pairs::vtk_write_data(pairs_runtime, "output/dem_sd_local", 0, ac->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/dem_sd_ghost", ac->nlocal(), ac->size(), t, vtk_freq);
    }

    pairs_sim->end();
}