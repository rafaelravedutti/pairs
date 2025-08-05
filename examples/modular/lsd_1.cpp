#include <iostream>
#include <memory>

#include "linear_spring_dashpot.hpp"
// cmake -DINPUT_SCRIPT=../examples/modular/linear_spring_dashpot.py -DCOMPILE_CUDA=ON -DUSE_WALBERLA=OFF -DCMAKE_BUILD_TYPE=Release -DUSER_SOURCE_FILES=../examples/modular/lsd_1.cpp ..

int main(int argc, char **argv) {

    auto pairs_sim = std::make_shared<PairsSimulation>();
    auto pairs_runtime = pairs_sim->getPairsRuntime();

    double dom_size = 0.1;
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, dom_size, dom_size, dom_size); 

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, pairs::flags::INFINITE | pairs::flags::FIXED);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, pairs::flags::INFINITE | pairs::flags::FIXED);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, pairs::flags::INFINITE | pairs::flags::FIXED);
    pairs::create_halfspace(pairs_runtime, dom_size, dom_size, dom_size,  -1, 0, 0,    0, pairs::flags::INFINITE | pairs::flags::FIXED);
    pairs::create_halfspace(pairs_runtime, dom_size, dom_size, dom_size,  0, -1, 0,    0, pairs::flags::INFINITE | pairs::flags::FIXED);
    pairs::create_halfspace(pairs_runtime, dom_size, dom_size, dom_size,  0, 0, -1,    0, pairs::flags::INFINITE | pairs::flags::FIXED);
    
    double density = 2550;
    double diameter = 0.0029; 
    double sphere_spacing = diameter * 1.2;
    double init_vel = 0.2;

    pairs::dem_sc_grid(pairs_runtime, dom_size/2.0, dom_size/2.0, dom_size/1.5,  
                        sphere_spacing, 
                        diameter, diameter, diameter, init_vel, density, 1);
    
    // pairs::create_sphere(pairs_runtime, 0.03, 0.03, 0.02,      -0.2, -0.2, 0,  density, diameter/2.0, 0, 0);
    // pairs::create_sphere(pairs_runtime, 0.02, 0.02, 0.02,    0.2, 0.2, 0,    density, diameter/2.0, 0, 0);

    pairs_sim->update_mass_and_inertia();

    pairs_sim->setCellWidth(diameter, diameter, diameter);
    pairs_sim->setInteractionRadius(diameter);
    pairs_sim->updateDomain();

    int num_timesteps = 20000;
    int vtk_freq = 200;
    double dt = 1e-5;
    double collision_time = dt*10;
    pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", 0);
    
    for (int t=0; t<num_timesteps; ++t){
        if ((t%500==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;

        pairs_sim->gravity(); 
        // pairs_sim->spring_dashpot(collision_time);          // Without contact history
        pairs_sim->linear_spring_dashpot(collision_time, dt);   // With contact history
        pairs_sim->euler(dt); 
        pairs_sim->reneighbor();

        pairs::vtk_write_data(pairs_runtime, "output/sd_1_local", 0, pairs_sim->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/sd_1_ghost", pairs_sim->nlocal(), pairs_sim->size(), t, vtk_freq);
    }

}
