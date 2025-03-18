#include <iostream>
#include <memory>

#include "spring_dashpot.hpp"
#include <chrono>

void set_feature_properties(std::shared_ptr<PairsAccessor> &ac){
    ac->setTypeStiffness(0,0, 1e6);
    ac->syncTypeStiffness();

    ac->setTypeDampingNorm(0,0, 300);
    ac->syncTypeDampingNorm();

    ac->setTypeFriction(0,0, 1.2);
    ac->syncTypeFriction();

    ac->setTypeDampingTan(0,0, 300);
    ac->syncTypeDampingTan();
}

int main(int argc, char **argv) {
    if(argc!=5){
        std::cerr << "4 args are required: Domain size (i.e. number of particles) in x,y,z and #timesteps." << std::endl;
        exit(-1);
    }

    double domain_size[3] = {std::stod(argv[1]), std::stod(argv[2]), std::stod(argv[3])};
    uint64_t num_timesteps = std::stoull(argv[4]);    

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());
    set_feature_properties(ac);     // Arbitray properties (all forces cancel out due to PBC in all dims)

    auto pairs_runtime = pairs_sim->getPairsRuntime();

    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, domain_size[0], domain_size[1], domain_size[2], false);

    double particle_spacing = 1.0;

    // Particle overlap is required for force calculation
    double peneration_depth = 0.01;  
    double diameter = particle_spacing + peneration_depth;
    
    double initial_velocity = 0.0;  // Stationary 
    double density = 1000;          // Arbitrary
    
    pairs::dem_sc_grid(pairs_runtime,   domain_size[0], domain_size[1], domain_size[2],
                                        particle_spacing, 
                                        diameter, diameter, diameter,
                                        initial_velocity, density, 1);
    
    
    // Cell width is here smaller than sphere diameter only for convenience to have everything aligned on a grid, but this 
    // doesn't affect the interactions computed. All spheres are on cell centers and are in contact with 6 neighbors. 
    double cell_width = particle_spacing;
    pairs_sim->setup_cells(cell_width, cell_width, cell_width, cell_width);
    
    // Inertia update is required for euler updates to be valid (but particles remain stationary)
    pairs_sim->update_mass_and_inertia(); 
    double dt = 0.001;  // Arbitrary
    
    int rank = pairs_sim->rank();
    if(rank==0) std::cout << "NUM_PROC: " << pairs_runtime->getDomainPartitioner()->getWorldSize() << std::endl;
    if(rank==0) std::cout << "NUM_NEIGH_AABBS: " << pairs_runtime->getDomainPartitioner()->getNumberOfNeighborAABBs() << std::endl;
    if(rank==0) std::cout << "NUM_TIMESTEPS: " << num_timesteps << std::endl;
    int print_interval = (num_timesteps >= 5) ? (num_timesteps / 5) : 1;

    // pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", pairs_runtime->getDomainPartitioner()->getWorldSize());
    

    // ------------------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t=0; t<num_timesteps; ++t){
        if ((t%print_interval==0) && rank==0) std::cout << "Timestep: " << t << std::endl;
        pairs_sim->communicate(t);
        pairs_sim->update_cells(t);
        pairs_sim->spring_dashpot();
        pairs_sim->euler(dt);
    }

    auto end = std::chrono::high_resolution_clock::now();
    // ------------------------------------------------------------------------------
    
    
    uint64_t nlocal = pairs_sim->nlocal();
    uint64_t global_nparticles;
    MPI_Reduce(&nlocal, &global_nparticles, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank==0) {
        auto duration = std::chrono::duration<double>(end - start);
        double total_runtime = duration.count(); // seconds
        std::cout << "TOTAL_RUNTIME: " << total_runtime << std::endl;
        std::cout << "GLOBAL_NPARTICLES: " << global_nparticles << std::endl;
        
        double pups = global_nparticles * num_timesteps / total_runtime;    // particle updates per second
        std::cout << "PUPS: " << pups << std::endl;
    }
    
    // if (rank==1) pairs::vtk_write_data(pairs_runtime, "output/ghost_spheres", pairs_sim->nlocal(), pairs_sim->size(), 0);
    // if (rank==2) pairs::vtk_write_data(pairs_runtime, "output/ghost_spheres", pairs_sim->nlocal(), pairs_sim->size(), 0);
    
    pairs_sim->end();
}