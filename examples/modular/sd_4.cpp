#include <iostream>
#include <memory>
#include <iomanip>

#include "spring_dashpot.hpp"

void set_feature_properties(std::shared_ptr<PairsAccessor> &ac){
    ac->setTypeStiffness(0,0, 100000);
    ac->setTypeStiffness(0,1, 100000);
    ac->setTypeStiffness(1,0, 100000);
    ac->setTypeStiffness(1,1, 100000);
    ac->syncTypeStiffness();

    ac->setTypeDampingNorm(0,0, 300);
    ac->setTypeDampingNorm(0,1, 300);
    ac->setTypeDampingNorm(1,0, 300);
    ac->setTypeDampingNorm(1,1, 300);
    ac->syncTypeDampingNorm();

    ac->setTypeFriction(0,0, 0.5);
    ac->setTypeFriction(0,1, 0.5);
    ac->setTypeFriction(1,0, 0.5);
    ac->setTypeFriction(1,1, 0.5);
    ac->syncTypeFriction();

    ac->setTypeDampingTan(0,0, 20);
    ac->setTypeDampingTan(0,1, 20);
    ac->setTypeDampingTan(1,0, 20);
    ac->setTypeDampingTan(1,1, 20);
    ac->syncTypeDampingTan();
}

int main(int argc, char **argv) {
    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());
    set_feature_properties(ac);

    auto pairs_runtime = pairs_sim->getPairsRuntime();

    pairs_runtime->initDomain(&argc, &argv, 
                    0, 0, 0, 20, 20, 20,    // Domain bounds
                    true                    // Enable dynamic load balancing (does initial refinement on a <1,1,1> blockforest)
                ); 

    pairs_runtime->getDomainPartitioner()->initWorkloadBalancer(pairs::Hilbert, 100, 1000);

    pairs::create_halfspace(pairs_runtime, 0,0,0,  1, 0, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 1, 0,     0, 13);
    pairs::create_halfspace(pairs_runtime, 0,0,0,  0, 0, 1,     0, 13);
    pairs::create_halfspace(pairs_runtime, 20,20,20,  -1, 0, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 20,20,20,  0, -1, 0,    0, 13);
    pairs::create_halfspace(pairs_runtime, 20,20,20,  0, 0, -1,    0, 13);

    double diameter_min = 0.3;
    double diameter_max = 0.3;
    double sphere_spacing = 0.4;
    pairs::dem_sc_grid(pairs_runtime, 10, 10, 15,  sphere_spacing, diameter_min, diameter_min, diameter_max,    2,      100,    2);
    
    double cell_width = diameter_max;
    double interaction_radius = diameter_max;

    pairs_sim->update_mass_and_inertia();

    pairs_sim->setCellWidth(cell_width, cell_width, cell_width);
    pairs_sim->setInteractionRadius(interaction_radius);
    pairs_sim->updateDomain();

    int num_timesteps = 4000;
    int vtk_freq = 20;
    int rebalance_freq = 200;
    double dt = 1e-3;

    pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", 0);
    
    for (int t=0; t<num_timesteps; ++t){
        if ((t % vtk_freq==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;
        
        
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 
        pairs_sim->euler(dt); 
        
        if (t % rebalance_freq == 0){ 
            pairs_sim->updateDomain();
        }
        else {
            pairs_sim->reneighbor();
        }

        if (t % vtk_freq==0){
            pairs::vtk_write_subdom(pairs_runtime, "output/subdom", t);
            pairs::vtk_write_data(pairs_runtime, "output/sd_4_local", 0, pairs_sim->nlocal(), t);
            pairs::vtk_write_data(pairs_runtime, "output/sd_4_ghost", pairs_sim->nlocal(), pairs_sim->size(), t);
        }
    }

    pairs_sim->end();
}