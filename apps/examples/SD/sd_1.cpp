#include <iostream>
#include <memory>
#include <filesystem>

#include "spring_dashpot.hpp"

int main(int argc, char **argv) {
    
    auto pairs_sim = std::make_shared<pairs::gen::PairsSimulation>();
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    pairs::gen::ParticleAccessor ac(pairs_sim.get());

    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 1, 1, 1); 

    // Create six halfspaces around the domain
    int idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {1, 0, 0});
    idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 1, 0});
    idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 0, 1});
    idx = pairs_sim->createObject(1, 1, 1, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {-1, 0, 0});
    idx = pairs_sim->createObject(1, 1, 1, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, -1, 0});
    idx = pairs_sim->createObject(1, 1, 1, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 0, -1});


    // Create a spheres
    int sphere_idx = pairs_sim->createObject( 0.6, 0.6, 0.7, pairs::Shapes::Sphere, 0);
    if(sphere_idx != ac.getInvalidIdx()){
        ac.setLinearVelocity(sphere_idx, {-2, -2, 0});
        ac.setRadius(sphere_idx, 0.05);
        ac.setMass(sphere_idx, ((4.0 / 3.0) * M_PI) * 0.05 * 0.05 * 0.05 * 1000);
    }

    // Create another sphere
    sphere_idx = pairs_sim->createObject( 0.4, 0.4, 0.68, pairs::Shapes::Sphere, 0);
    if(sphere_idx != ac.getInvalidIdx()){
        ac.setLinearVelocity(sphere_idx, {2, 2, 0});
        ac.setRadius(sphere_idx, 0.05);
        ac.setMass(sphere_idx, ((4.0 / 3.0) * M_PI) * 0.05 * 0.05 * 0.05 * 1000);
    }

    pairs_sim->update_mass_and_inertia();

    pairs_sim->setCellWidth(0.1, 0.1, 0.1);
    pairs_sim->setInteractionRadius(0.1);
    pairs_sim->updateDomain();

    int num_timesteps = 2000;
    int vtk_freq = 20;
    double dt = 1e-3;
    std::filesystem::create_directories("output");
    
    for (int t=0; t<num_timesteps; ++t){
        if ((t%500==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;

        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 
        pairs_sim->euler(dt); 
        pairs_sim->reneighbor();

        pairs::vtk_write_data(pairs_runtime, "output/sd_1_local", 0, pairs_sim->nlocal(), t, vtk_freq);
        pairs::vtk_write_data(pairs_runtime, "output/sd_1_ghost", pairs_sim->nlocal(), pairs_sim->size(), t, vtk_freq);
    }

    
}
