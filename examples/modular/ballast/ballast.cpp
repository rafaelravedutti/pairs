#include <iostream>
#include <memory>
#include <iomanip>
#include <chrono>

#include "ballast.hpp"

// cmake -DPAIRS_INPUT_SCRIPT=../examples/modular/ballast/ballast.py -DPAIRS_INPUT_SRCS=../examples/modular/ballast/ballast.cpp -DPAIRS_BUILD_WITH_WALBERLA=OFF -DPAIRS_BUILD_WITH_CUDA=OFF ..

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

    ac.setTypeFriction(0,0, 1.0);
    ac.setTypeFriction(0,1, 1.0);
    ac.setTypeFriction(1,0, 1.0);
    ac.setTypeFriction(1,1, 1.0);
    ac.syncTypeFriction();

    ac.setTypeDampingTan(0,0, 300);
    ac.setTypeDampingTan(0,1, 300);
    ac.setTypeDampingTan(1,0, 300);
    ac.setTypeDampingTan(1,1, 300);
    ac.syncTypeDampingTan();
}

void print_usage(){
    std::cerr << "Required args: num_beds_x num_beds_y num_timesteps" << std::endl;
    exit(-1);
}

int main(int argc, char **argv) {
    if (argc<4) print_usage();

    auto pairs_sim = std::make_shared<PairsSimulation>();
    
    ParticleAccessor ac(pairs_sim.get());
    auto pairs_runtime = pairs_sim->getPairsRuntime();

    // ===================================================================================================
    // Domain 
    // ===================================================================================================
    int nbeds[2] = {std::stoi(argv[1]), std::stoi(argv[2])};
    double bed_size[3] = {1.5, 1.0, 0.65};
    double domain_size[3] = {bed_size[0]*nbeds[0], bed_size[1]*nbeds[1], bed_size[2]};
    pairs_runtime->initDomain(&argc, &argv, 0, 0, 0, domain_size[0], domain_size[1], domain_size[2], false);

    // ===================================================================================================
    // Spheres
    // ===================================================================================================
    double spacing = 0.061;
    double diamter_max = 0.06;
    double diamter_min = 0.025;
    double sphere_density = 2700;

    // Initial grid of particles (domain_size = {1.5, 1.0, 1.5};)
    // ------------------------------------------------------------------
    // pairs::dem_sc_grid(pairs_runtime, 
    //     domain_size[0], domain_size[1], domain_size[2],  spacing , 0, diamter_min, diamter_max, 0.01, sphere_density,    2);


    // Settled bed with box in place
    // ------------------------------------------------------------------
    for(int i=0; i<nbeds[0]; ++i){
        for(int j=0; j<nbeds[1]; ++j){
            pairs::read_spheres(pairs_runtime, 
                "../examples/modular/ballast/spheres_bed_with_box.txt", 
                {i*bed_size[0], j*bed_size[1], 0.0});

        }
    }

    // ===================================================================================================
    // Box (sleeper)
    // ===================================================================================================
    // pairs::read_boxes(pairs_runtime, "../examples/modular/ballast/boxes.txt");
    double box_density = 8000;
    double box_size[3] = {0.4, 0.3, 0.15};
    double box_pos[3] = {0.779191, 0.497293, 0.484325};
    int num_boxes = nbeds[0] * nbeds[1];
    pairs::id_t box_uid[num_boxes];
    for(int i=0; i<nbeds[0]; ++i){
        for(int j=0; j<nbeds[1]; ++j){
            box_uid[i*nbeds[1] + j] = pairs::create_box(pairs_runtime, 
                box_pos[0] + i*bed_size[0], 
                box_pos[1] + j*bed_size[1], 
                box_pos[2],
                0, 0, 0, 
                box_size[0], box_size[1], box_size[2],  
                box_density, 1, pairs::flags::GLOBAL); 
        }
    }

    pairs::Vector3<double> external_force(100, 0, 0);
    // pairs::Vector3<double> external_force(0, 0, 0);

    // ===================================================================================================
    // Halfspace (bottom plate)
    // ===================================================================================================
    auto plate_uid = pairs::create_halfspace(pairs_runtime, 0,0,0, 0, 0, 1, 0, pairs::flags::INFINITE | pairs::flags::FIXED);
    bool vibrate = true;
    double vib_freq = 30;   // Hz
    double vib_amp = 0.001;  

    // ===================================================================================================
    // Other setup
    // ===================================================================================================
    double cell_width = diamter_max;
    pairs_sim->setCellWidth(cell_width, cell_width, cell_width);
    pairs_sim->setInteractionRadius(diamter_max);
    pairs_sim->update_mass_and_inertia();
    pairs_sim->updateDomain();

    int num_timesteps = std::stoi(argv[3]); 
    int vtk_freq = 5000;
    double dt = 1e-5;
    set_feature_properties(ac);

    int rank = pairs_sim->rank();
    std::cout << "rank (" << rank << ") nlocal = " << pairs_sim->nlocal() << " - nghost = " << pairs_sim->nghost() << std::endl;
    pairs::vtk_write_subdom(pairs_runtime, "output/subdom_init", 0);
    
    // ===================================================================================================
    // Timestep Loop
    // ===================================================================================================
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int t=0; t<num_timesteps; ++t){
        ac.syncUid(ParticleAccessor::Host);
        
        // Change bottom plate position
        // ------------------------------------------------------------------
        if (vibrate){
            auto plate_idx = ac.uidToIdxLocal(plate_uid);
            double posz = -cos(2.0*M_PI * vib_freq * (t*dt)) * (vib_amp/2.0) + (vib_amp/2.0);
            ac.syncPosition(ParticleAccessor::Host);
            ac.setPosition(plate_idx, {0.0, 0.0, posz});
            ac.syncPosition(ParticleAccessor::Host);
        }

        // Calculate forces
        // ------------------------------------------------------------------
        // MPI_Barrier(MPI_COMM_WORLD);
        pairs_sim->spring_dashpot(); 
        pairs_sim->gravity(); 

        // Apply external force to Box
        // ------------------------------------------------------------------
        for(int i=0; i<num_boxes; ++i){
            auto box_idx = ac.uidToIdxLocal(box_uid[i]);
            ac.syncForce(ParticleAccessor::Host);
            ac.syncTorque(ParticleAccessor::Host);
            ac.setForce(box_idx, ac.getForce(box_idx) + external_force);
            ac.setTorque(box_idx, {0.0, 0.0, 0.0});    // Reset torque to prevent rotation
            ac.syncForce(ParticleAccessor::Host);
            ac.syncTorque(ParticleAccessor::Host);
        }

        // Update positions and reneighbor
        // ------------------------------------------------------------------
        pairs_sim->euler(dt); 
        pairs_sim->reneighbor(); 

        // VTK
        // ------------------------------------------------------------------
        if (t % vtk_freq==0){
            pairs::vtk_write_data(pairs_runtime, "output/local_spheres", 0, pairs_sim->nlocal(), t);

            if(rank==0){
                auto runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
                std::cout << "Timestep: " << t << " - Time = " << dt*t << " - Runtime = " << runtime << std::endl;
                ac.syncPosition(ParticleAccessor::Host);
                for(int i=0; i<num_boxes; ++i){
                    auto box_idx = ac.uidToIdxLocal(box_uid[i]);
                    std::cout << "\t Box (" << i << ") position = " << ac.getPosition(box_idx) << std::endl;
                }
                pairs::vtk_with_rotation(pairs_runtime, pairs::Shapes::Box, "output/local_boxes", 0, pairs_sim->nlocal(), t);
                
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // ===================================================================================================
    // Print stats
    // ===================================================================================================
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

        ac.syncUid(ParticleAccessor::Host);
        for(int i=0; i<num_boxes; ++i){
            auto box_idx = ac.uidToIdxLocal(box_uid[i]);
            std::cout << "box (" << i << ") final position = " << ac.getPosition(box_idx) << std::endl;
        }

        auto plate_idx = ac.uidToIdxLocal(plate_uid);
        std::cout << "plate final position = " << ac.getPosition(plate_idx) << std::endl;
    }
    pairs::log_timers(pairs_runtime);

    ac.end();
    
}