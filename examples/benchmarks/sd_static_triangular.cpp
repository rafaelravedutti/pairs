#include <iostream>
#include <memory>
#include <chrono>

#include "spring_dashpot_no_pbc.hpp"

void randomaize_indices(PairsAccessor ac){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, ac.nlocal() - 1);

    ac.syncPosition(PairsAccessor::Host);
    for (int i=0; i<ac.nlocal(); ++i){
        int j = dist(gen);
        auto tmp = ac.getPosition(i);
        ac.setPosition(i, ac.getPosition(j));
        ac.setPosition(j, tmp);
    }
    ac.syncPosition(PairsAccessor::Host);
}

template<typename Type>
void print_global_stats(std::string name, Type value, MPI_Datatype mpi_type, MPI_Comm comm){
    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    std::vector<Type> all_values(world_size);
    MPI_Gather(&value, 1, mpi_type, all_values.data(), 1, mpi_type, 0, comm);

    if(rank == 0){
        std::sort(all_values.begin(), all_values.end());
        Type sum = std::accumulate(all_values.begin(), all_values.end(), Type());
        double avg = static_cast<double>(sum)/world_size;

        // Standard deviation ------------------
        double sum_sq = 0.0;
        for(const auto &v : all_values){
            sum_sq += (static_cast<double>(v) - avg) * (static_cast<double>(v) - avg);
        }
        double std_dev = std::sqrt(sum_sq / world_size);

        // Median ------------------------------
        double median = 0.0;
        if(world_size%2 == 0){
            median = static_cast<double>(all_values[world_size/2 - 1] + all_values[world_size/2]) / 2.0;   
        } else{
            median = static_cast<double>(all_values[world_size/2]);
        }

        std::cout << "-----------------------------------" << std::endl;
        std::cout << name + "_MIN: " << all_values[0] << std::endl;
        std::cout << name + "_MAX: " << all_values[world_size - 1] << std::endl;
        std::cout << name + "_SUM: " << sum << std::endl;
        std::cout << name + "_AVG: " << avg << std::endl;
        std::cout << name + "_MED: " << median << std::endl;
        std::cout << name + "_STDDEV: " << std_dev << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }
}

void print_usage(){
    std::cerr << "Invalid args." << std::endl;
    std::cerr << "Required args: X Y Z #Timesteps LoadBalanced(bool)" << std::endl;
    std::cerr << "If LoadBalanced, 3 addition args are required: Algorithm(hilbert/morton) RegridMin RegridMax." << std::endl;
    exit(-1);
}

int main(int argc, char **argv) {
    if(argc<6) print_usage();

    double domain_size[3] = {std::stod(argv[1]), std::stod(argv[2]), std::stod(argv[3])};
    uint64_t num_timesteps = std::stoull(argv[4]);

    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();

    PairsAccessor ac(pairs_sim.get());
    
    auto pairs_runtime = pairs_sim->getPairsRuntime();
    
    bool load_balanced = (std::stoi(argv[5]) != 0);
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, domain_size[0], domain_size[1], domain_size[2], load_balanced);

    if(load_balanced){
        if(argc!=9) print_usage();

        std::string alg_str = argv[6];
        LoadBalancingAlgorithms alg;

        if(alg_str == "hilbert") alg = pairs::Hilbert;
        else if (alg_str == "morton") alg = pairs::Morton;
        else {std::cerr << "Invalid rebalancing algorithm." << std::endl; print_usage();}

        int regrid_min = std::stoi(argv[7]);
        int regrid_max = std::stoi(argv[8]);
        pairs_runtime->getDomainPartitioner()->initWorkloadBalancer(alg, regrid_min, regrid_max);
    } else {
        if(argc!=6) print_usage();
    }

    double particle_spacing = 1.0;

    // Particle overlap is required for force calculation (but stiffnesses are zero by default, so forces remain zero)
    double peneration_depth = 0.01;  
    double diameter = particle_spacing + peneration_depth;
    
    double initial_velocity = 0.0;  // Stationary 
    double density = 1000;          // Arbitrary
    bool lower_tirangular = true;
    
    pairs::dem_sc_grid(pairs_runtime,   domain_size[0], domain_size[1], domain_size[2],
                                        particle_spacing, 
                                        diameter, diameter, diameter,
                                        initial_velocity, density, 1, lower_tirangular);
    
    // Inertia update is required for euler updates to be valid (but particles remain stationary)
    pairs_sim->update_mass_and_inertia(); 

    double cell_width = diameter;
    pairs_sim->setCellWidth(cell_width, cell_width, cell_width);
    pairs_sim->setInteractionRadius(cell_width);

    double dt = 0.001;  // Arbitrary
    uint64_t print_interval = (num_timesteps >= 5) ? (num_timesteps / 5) : 1;
    
    // Rebalance
    pairs_sim->updateDomain();

    randomaize_indices(ac);
    pairs_sim->reneighbor();

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

    // ------------------------------------------------------------------------------
    // Timestep Loop
    // ------------------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t t=0; t<num_timesteps; ++t){
        if ((t%print_interval==0) && rank==0) std::cout << "Timestep: " << t << std::endl;
        pairs_sim->spring_dashpot();
        pairs_sim->euler(dt);
        pairs_sim->reneighbor();
    }

    auto end = std::chrono::high_resolution_clock::now();
    // ------------------------------------------------------------------------------
    
    uint64_t global_nparticles;
    MPI_Reduce(&nlocal, &global_nparticles, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank==0) {
        auto duration = std::chrono::duration<double>(end - start);
        double total_runtime = duration.count(); // seconds
        std::cout << "TOTAL_RUNTIME: " << total_runtime << std::endl;
        std::cout << "GLOBAL_NPARTICLES: " << global_nparticles << std::endl;
        
        double pups = static_cast<double>(global_nparticles * num_timesteps) / total_runtime;    // particle updates per second
        std::cout << "PUPS: " << pups << std::endl;
    }
    
    // pairs::vtk_write_subdom(pairs_runtime, "output/sd_subdom", 0);
    // pairs::vtk_write_data(pairs_runtime, "output/sd_local", 0, pairs_sim->nlocal(), 0);
    // pairs::vtk_write_data(pairs_runtime, "output/sd_ghost", pairs_sim->nlocal(), pairs_sim->size(), 0);
    pairs::log_timers(pairs_runtime);

    ac.end();
    pairs_sim->end();
}