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

    pairs_sim->set_domain(argc, argv, 0, 0, 0, 1, 1, 1);

    pairs_sim->create_halfspace(0,0,0,  1, 0, 0,     0, 13);
    pairs_sim->create_halfspace(0,0,0,  0, 1, 0,     0, 13);
    pairs_sim->create_halfspace(0,0,0,  0, 0, 1,     0, 13);
    pairs_sim->create_halfspace(1,1,1,  -1, 0, 0,    0, 13);
    pairs_sim->create_halfspace(1,1,1,  0, -1, 0,    0, 13);
    pairs_sim->create_halfspace(1,1,1,  0, 0, -1,    0, 13);

    pairs::id_t pUid = pairs_sim->create_sphere(0.6, 0.6, 0.7,      0, 0, 0,  1000, 0.05, 0, 0);
    pairs::id_t pUid2 = pairs_sim->create_sphere(0.4, 0.4, 0.76,    2, 2, 0,    1000, 0.05, 0, 0);

    MPI_Allreduce(MPI_IN_PLACE, &pUid, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &pUid2, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

    auto pIsLocalInMyRank = [&](pairs::id_t uid){return ac->uidToIdxLocal(uid) != ac->getInvalidIdx();};

    pairs_sim->setup_sim();

    pairs_sim->communicate();

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
        pairs_sim->update_cells();
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
        pairs_sim->reset_volatiles(); 

        // Communicate
        //-------------------------------------------------------------------------------------------
        pairs_sim->communicate(t);

        pairs_sim->vtk_write("output/dem_sd_local", 0, ac->nlocal(), t, vtk_freq);
        pairs_sim->vtk_write("output/dem_sd_ghost", ac->nlocal(), ac->size(), t, vtk_freq);
    }

    pairs_sim->end();
}