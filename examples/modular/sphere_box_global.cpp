#include <iostream>
#include <memory>
#include <iomanip>

#include "sphere_box_global.hpp"

void set_feature_properties(ParticleAccessor ac){
    ac.setTypeStiffness(0,0, 1e6);
    ac.setTypeStiffness(0,1, 1e6);
    ac.setTypeStiffness(1,0, 1e6);
    ac.setTypeStiffness(1,1, 1e6);
    ac.syncTypeStiffness();

    ac.setTypeDampingNorm(0,0, 300);
    ac.setTypeDampingNorm(0,1, 300);
    ac.setTypeDampingNorm(1,0, 300);
    ac.setTypeDampingNorm(1,1, 300);
    ac.syncTypeDampingNorm();

    ac.setTypeFriction(0,0, 1.2);
    ac.setTypeFriction(0,1, 1.2);
    ac.setTypeFriction(1,0, 1.2);
    ac.setTypeFriction(1,1, 1.2);
    ac.syncTypeFriction();

    ac.setTypeDampingTan(0,0, 300);
    ac.setTypeDampingTan(0,1, 300);
    ac.setTypeDampingTan(1,0, 300);
    ac.setTypeDampingTan(1,1, 300);
    ac.syncTypeDampingTan();
}

int main(int argc, char **argv) {
    auto pairs_sim = std::make_shared<PairsSimulation>();
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    ParticleAccessor ac(pairs_sim.get());

    set_feature_properties(ac);

    // Create a domain of size 30*30*30 and enable dynamic load balancing
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, 30, 30, 30, true); 
    // Set up the redistribution algorithm
    pairs_runtime->getDomainPartitioner()->initWorkloadBalancer(pairs::Hilbert, 100, 800);

    // Create six halfspaces around the domain
    int idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {1, 0, 0});
    idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 1, 0});
    idx = pairs_sim->createObject(0, 0, 0, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 0, 1});
    idx = pairs_sim->createObject(30, 30, 30, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {-1, 0, 0});
    idx = pairs_sim->createObject(30, 30, 30, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, -1, 0});
    idx = pairs_sim->createObject(30, 30, 30, pairs::Shapes::Halfspace, pairs::flags::INFINITE | pairs::flags::FIXED);
    ac.setNormal(idx, {0, 0, -1});

    // radius of small particles
    double radius = 0.5;

    // Create a bed of small particles
    pairs::dem_sc_grid(pairs_runtime, 30, 20, 5,  radius*2 , radius*2 , radius*2, radius*2,    2,      250,    2);

    // Create a global box
    int box_idx = pairs_sim->createObject(12, 12, 13.5, pairs::Shapes::Box, pairs::flags::GLOBAL);
    auto box_uid = ac.getUid(box_idx);
    ac.setEdgeLength(box_idx, {15, 2, 13});
    ac.setMass(box_idx, 15*2*13*20.0);

    // Create a global sphere
    int sp1_idx = pairs_sim->createObject(15, 20, 15, pairs::Shapes::Sphere, pairs::flags::GLOBAL);
    ac.setLinearVelocity(sp1_idx, {0, 4, 0});
    ac.setRadius(sp1_idx, 4);
    ac.setMass(sp1_idx, ((4.0 / 3.0) * M_PI) * 64 * 50);
 
    // Create a global fixed sphere
    int sp2_idx = pairs_sim->createObject(15, 25, 4, pairs::Shapes::Sphere, pairs::flags::GLOBAL | pairs::flags::FIXED);
    ac.setRadius(sp2_idx, 4);
    ac.setMass(sp2_idx, ((4.0 / 3.0) * M_PI) * 64 * 50);

    // Create 1 extra local sphere
    int sp3_idx = pairs_sim->createObject(25, 25, 25, pairs::Shapes::Sphere, 0);
    if(sp3_idx != ac.getInvalidIdx()){
        ac.setLinearVelocity(sp3_idx, {-5, -7, 0});
        ac.setRadius(sp3_idx, radius);
        ac.setMass(sp3_idx, ((4.0 / 3.0) * M_PI) * radius * radius * radius * 50);
    }

    pairs_sim->update_mass_and_inertia();
    
    // Use the diameter of small particles to set up the cell list
    double cell_width = radius * 2;
    pairs_sim->setCellWidth(cell_width, cell_width, cell_width);
    pairs_sim->setInteractionRadius(cell_width);

    // Update the domain once all bodies have been created
    pairs_sim->updateDomain();

    int num_timesteps = 20000; 
    int vtk_freq = 100;  
    int rebalance_freq = 2000;
    double dt = 0.001;

    pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", 0);
    
    for (int t=0; t<num_timesteps; ++t){
        if ((t % vtk_freq==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;
        
        pairs_sim->gravity(); 
        
        // All global and local interactions are contained within the 'spring_dashpot' module
        // You have the option to call spring_dashpot before or after 'gravity' or any other force-update module
        pairs_sim->spring_dashpot();     

        pairs_sim->euler(dt);

        if (t % rebalance_freq == 0){ 
            pairs_sim->updateDomain();  // Rebalance the domain + reneighbor
        }
        else {
            pairs_sim->reneighbor();    // Exchange ghosts + migrating + cell-list update
        }
        
        if (t % vtk_freq==0){
            pairs::vtk_with_rotation(pairs_runtime, pairs::Shapes::Box, "output/local_boxes", 0, pairs_sim->nlocal(), t);
            pairs::vtk_with_rotation(pairs_runtime, pairs::Shapes::Sphere, "output/local_spheres", 0, pairs_sim->nlocal(), t);
        }
    }

    ac.end();
}