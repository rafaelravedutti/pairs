#include <iostream>
#include <memory>

#include "spring_dashpot.hpp"

void change_gravitational_force(std::shared_ptr<PairsAccessor> &ac, int idx){
    pairs::Vector3<double> upward_gravity(0.0, 0.0, 2 * ac->getMass(idx) * 9.81); 
    ac->setForce(idx, ac->getForce(idx) + upward_gravity);
}

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());
    
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 1, 1, 1);

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  -1, 0, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, -1, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, 0, -1,    0, 13);

    pairs::id_t pUid = pairs::create_sphere(pairs_runtime ,0.6, 0.6, 0.7,      0, 0, 0,  1000, 0.05, 0, 0);
    pairs::create_sphere(pairs_runtime, 0.4, 0.4, 0.76,    2, 2, 0,    1000, 0.05, 0, 0);

    MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    auto pIsLocalInMyRank = [&](pairs::id_t uid){return ac->uidToIdxLocal(uid) != ac->getInvalidIdx();};

    pairs_sim->setup_sim(0.1, 0.1, 0.1, 0.1);
    pairs_sim->update_mass_and_inertia();

    pairs_sim->communicate(0);

    int num_timesteps = 2000;
    int vtk_freq = 20;
    double dt = 1e-3;

    for (int t=0; t<num_timesteps; ++t){

        // Print position of particle pUid
        //-------------------------------------------------------------------------------------------
        if(pIsLocalInMyRank(pUid)){
            std::cout << "Timestep (" << t << "): Particle " << pUid << " is in rank " << pairs_sim->rank() << std::endl;
            int idx = ac->uidToIdxLocal(pUid);
            std::cout << "Position = (" 
                    << ac->getPosition(idx)[0] << ", "
                    << ac->getPosition(idx)[1] << ", " 
                    << ac->getPosition(idx)[2] << ")" << std::endl;

        }

        // Calculate forces
        //-------------------------------------------------------------------------------------------
        pairs_sim->update_cells(t);
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 

        // Change gravitational force on particle pUid
        //-------------------------------------------------------------------------------------------
        if(pIsLocalInMyRank(pUid)){
            int idx = ac->uidToIdxLocal(pUid);

            std::cout << "Force before changing = (" 
                    << ac->getForce(idx)[0] << ", "
                    << ac->getForce(idx)[1] << ", " 
                    << ac->getForce(idx)[2] << ")" << std::endl;

            change_gravitational_force(ac, idx);

            std::cout << "Force after changing = (" 
                    << ac->getForce(idx)[0] << ", "
                    << ac->getForce(idx)[1] << ", " 
                    << ac->getForce(idx)[2] << ")" << std::endl;
        }

        // Euler
        //-------------------------------------------------------------------------------------------
        pairs_sim->euler(dt);

        // Communicate
        //-------------------------------------------------------------------------------------------
        pairs_sim->communicate(t);

        pairs::vtk_write_data(pairs_runtime, "output/sd_3_CPU_local", 0, ac->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/sd_3_CPU_ghost", ac->nlocal(), ac->size(), t, vtk_freq);
    }

    pairs_sim->end();
}