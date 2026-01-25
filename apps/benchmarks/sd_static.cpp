#include <iostream>
#include <memory>
#include <chrono>
#include <filesystem>

#include "print_stats.hpp"
#include "sd.hpp"

using Accessor_T = pairs::gen::ParticleAccessor;

void randomaize_indices(Accessor_T ac){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, ac.nlocal() - 1);

    ac.syncPosition(Accessor_T::Host);
    for (int i=0; i<ac.nlocal(); ++i){
        int j = dist(gen);
        auto tmp = ac.getPosition(i);
        ac.setPosition(i, ac.getPosition(j));
        ac.setPosition(j, tmp);
    }
    ac.syncPosition(Accessor_T::Host);
}

double run_sim(std::shared_ptr<pairs::gen::PairsSimulation> &pairs_sim, double dt, uint64_t num_timesteps, bool profile=false){
    MPI_Barrier(MPI_COMM_WORLD);
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    uint64_t print_interval = (num_timesteps >= 5) ? (num_timesteps / 5) : 1;
    int rank = pairs_sim->rank();

    if(profile){
        if(rank==0) std::cout << "Running sim with profiling enabled..." << std::endl;
        pairs_runtime->getTimers()->enable();
        pairs_runtime->getTimers()->enableMPIBarrier();
    }
    else{
        if(rank==0) std::cout << "Running sim without profiling..." << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t t=0; t<num_timesteps; ++t){
        if ((t%print_interval==0) && rank==0) std::cout << "Timestep: " << t << std::endl;
        pairs_sim->spring_dashpot();
        pairs_sim->euler(dt);
        pairs_sim->reneighbor();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);
    double total_runtime = duration.count(); // seconds
    
    if(profile){
        pairs::log_timers(pairs_runtime);
        pairs_runtime->getTimers()->disable();
        pairs_runtime->getTimers()->disableMPIBarrier();
    }

    return total_runtime;
}

int main(int argc, char **argv) {
    if(argc!=6){
        std::cerr << "5 args are required: profile(bool), Domain size (i.e. number of particles) in x,y,z and #timesteps." << std::endl;
        exit(-1);
    }

    bool profile = (std::stoi(argv[1]) != 0);
    double domain_size[3] = {std::stod(argv[2]), std::stod(argv[3]), std::stod(argv[4])};
    uint64_t num_timesteps = std::stoull(argv[5]);    

    auto pairs_sim = std::make_shared<pairs::gen::PairsSimulation>();
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    Accessor_T ac(pairs_sim.get());

    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, domain_size[0], domain_size[1], domain_size[2], false);

    double particle_spacing = 1.0;

    // Particle overlap is required for force calculation (but forces remain zero since stiffnesses are zero by default)
    double peneration_depth = 0.01;  
    double diameter = particle_spacing + peneration_depth;
    
    double initial_velocity = 0.0;  // Stationary 
    double density = 1000;          // Arbitrary
    
    pairs::dem_sc_grid(pairs_runtime,   domain_size[0], domain_size[1], domain_size[2],
                                        particle_spacing, 
                                        diameter, diameter, diameter,
                                        initial_velocity, density, 1);
    
    pairs_sim->update_mass_and_inertia(); 
    
    // Cell width is here smaller than sphere diameter only for convenience to have everything aligned on a grid, but this 
    // doesn't affect the interactions computed. All spheres are on cell centers and are in contact with 6 neighbors. 
    double cell_width = particle_spacing;

    pairs_sim->setCellWidth(cell_width, cell_width, cell_width);
    pairs_sim->setInteractionRadius(cell_width);
    pairs_sim->updateDomain();
    
    randomaize_indices(ac);
    pairs_sim->reneighbor();

    // Inertia update is required for euler updates to be valid (but particles remain stationary)
    double dt = 0.001;  // Arbitrary
    
    // Stats
    // ------------------------------------------------------------------------------
    int rank = pairs_sim->rank();
    int world_size = pairs_runtime->getDomainPartitioner()->getWorldSize();

    int num_local_aabbs = pairs_runtime->getDomainPartitioner()->getNumberOfLocalAABBs();
    int num_neigh_aabbs = pairs_runtime->getDomainPartitioner()->getNumberOfNeighborAABBs();
    int num_neigh_ranks = pairs_runtime->getDomainPartitioner()->getNumberOfNeighborRanks();
    uint64_t nlocal = pairs_sim->nlocal();
    uint64_t nghost = pairs_sim->nghost();

    std::cout << "rank (" << rank << "): \t nlocal = " << nlocal << " nghost = " << nghost << 
         " local_aabbs = " << num_local_aabbs << 
         " neigh_aabbs = " << num_neigh_aabbs << 
         " neigh_ranks = " << num_neigh_ranks << std::endl;

    print_global_stats("NLOCAL", nlocal, MPI_UINT64_T, MPI_COMM_WORLD);
    print_global_stats("NGHOST", nghost, MPI_UINT64_T, MPI_COMM_WORLD);
    print_global_stats("NUM_LOCAL_AABBS", num_local_aabbs, MPI_INT, MPI_COMM_WORLD);
    print_global_stats("NUM_NEIGH_AABBS", num_neigh_aabbs, MPI_INT, MPI_COMM_WORLD);
    print_global_stats("NUM_NEIGH_RANKS", num_neigh_ranks, MPI_INT, MPI_COMM_WORLD);

    if(rank==0){
        std::cout << "NUM_PROC: " << world_size << std::endl;
        std::cout << "NUM_TIMESTEPS: " << num_timesteps << std::endl;
    }

    // std::filesystem::create_directories("output");
    // pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", pairs_runtime->getDomainPartitioner()->getWorldSize());
    
    // Run simulation without timers (and no MPI barriers) to get the total runtime  
    // ------------------------------------------------------------------------------
    double total_runtime = run_sim(pairs_sim, dt, num_timesteps);
      
    uint64_t global_nparticles;
    MPI_Reduce(&nlocal, &global_nparticles, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank==0) {
        std::cout << "TOTAL_RUNTIME: " << total_runtime << std::endl;
        std::cout << "GLOBAL_NPARTICLES: " << global_nparticles << std::endl;
        
        double pups = double(global_nparticles * num_timesteps) / total_runtime;    // particle updates per second
        std::cout << "PUPS: " << pups << std::endl;
        std::cout << "PUPPS: " << pups / world_size << std::endl;
    }


    // Run simulation with timers (and MPI barriers) to get detailed runtimes for all kernels
    // ------------------------------------------------------------------------------
    if(profile) run_sim(pairs_sim, dt, num_timesteps, true);
    
    // pairs::vtk_write_data(pairs_runtime, "output/local_spheres", 0, pairs_sim->nlocal(), 0);
    // pairs::vtk_write_data(pairs_runtime, "output/ghost_spheres", pairs_sim->nlocal(), pairs_sim->size(), 0);
    
    ac.end();
}
