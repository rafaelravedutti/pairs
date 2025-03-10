#include <iostream>
#include <memory>

#include "spring_dashpot.hpp"

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    auto pairs_runtime = pairs_sim->getPairsRuntime();

    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 1, 1, 1); 

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  -1, 0, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, -1, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 1,1,1,  0, 0, -1,    0, 13);
    pairs::create_sphere(pairs_runtime, 0.6, 0.6, 0.7,      -2, -2, 0,  1000, 0.05, 0, 0);
    pairs::create_sphere(pairs_runtime, 0.4, 0.4, 0.68,    2, 2, 0,    1000, 0.05, 0, 0);

    pairs_sim->setup_cells(0.1, 0.1, 0.1, 0.1);
    pairs_sim->update_mass_and_inertia();

    int num_timesteps = 2000;
    int vtk_freq = 20;
    double dt = 1e-3;
    
    for (int t=0; t<num_timesteps; ++t){
        if ((t%500==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;

        pairs_sim->communicate(t);
        
        pairs_sim->update_cells(t); 

        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 
        pairs_sim->euler(dt); 

        pairs::vtk_write_data(pairs_runtime, "output/sd_1_local", 0, pairs_sim->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/sd_1_ghost", pairs_sim->nlocal(), pairs_sim->size(), t, vtk_freq);
    }

    pairs_sim->end();
}
