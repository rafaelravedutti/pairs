#include <iostream>
#include <memory>
#include <iomanip>
#include <chrono>

#include "ballast.hpp"

// cmake -DINPUT_SCRIPT=../examples/modular/ballast/ballast.py -DUSER_SOURCE_FILES=../examples/modular/ballast/ballast.cpp -DUSE_WALBERLA=OFF -DCOMPILE_CUDA=OFF ..

void set_feature_properties(std::shared_ptr<PairsAccessor> &ac){
    ac->setTypeStiffness(0,0, 1e6);
    ac->setTypeStiffness(0,1, 1e6);
    ac->setTypeStiffness(1,0, 1e6);
    ac->setTypeStiffness(1,1, 1e6);
    ac->syncTypeStiffness();

    ac->setTypeDampingNorm(0,0, 300);
    ac->setTypeDampingNorm(0,1, 300);
    ac->setTypeDampingNorm(1,0, 300);
    ac->setTypeDampingNorm(1,1, 300);
    ac->syncTypeDampingNorm();

    ac->setTypeFriction(0,0, 1.0);
    ac->setTypeFriction(0,1, 1.0);
    ac->setTypeFriction(1,0, 1.0);
    ac->setTypeFriction(1,1, 1.0);
    ac->syncTypeFriction();

    ac->setTypeDampingTan(0,0, 300);
    ac->setTypeDampingTan(0,1, 300);
    ac->setTypeDampingTan(1,0, 300);
    ac->setTypeDampingTan(1,1, 300);
    ac->syncTypeDampingTan();
}

int main(int argc, char **argv) {
    auto pairs_sim = std::make_shared<PairsSimulation>();
    pairs_sim->initialize();
    auto ac = std::make_shared<PairsAccessor>(pairs_sim.get());
    auto pairs_runtime = pairs_sim->getPairsRuntime();

    // ===================================================================================================
    // Domain
    // ===================================================================================================
    double domain_size[3] = {1.5, 1.0, 0.8};
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
    pairs::read_spheres(pairs_runtime, "../examples/modular/ballast/spheres_bed_with_box.txt");

    // ===================================================================================================
    // Box (sleeper)
    // ===================================================================================================
    // pairs::read_boxes(pairs_runtime, "../examples/modular/ballast/boxes.txt");
    double box_density = 8000;
    double box_size[3] = {0.4, 0.3, 0.15};
    double box_pos[3] = {0.779191, 0.497293, 0.484325};
    auto box_uid = pairs::create_box(pairs_runtime, box_pos[0], box_pos[1], box_pos[2],
                                        0, 0, 0, 
                                        box_size[0], box_size[1], box_size[2],  
                                        box_density, 1, pairs::flags::GLOBAL); 

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

    int num_timesteps = 500000; 
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
        if ((t % vtk_freq==0) && pairs_sim->rank()==0) std::cout << "Timestep: " << t << std::endl;
        ac->syncUid(PairsAccessor::Host);
        
        // Change bottom plate position
        // ------------------------------------------------------------------
        if (vibrate){
            auto plate_idx = ac->uidToIdxLocal(plate_uid);
            double posz = -cos(2.0*M_PI * vib_freq * (t*dt)) * (vib_amp/2.0) + (vib_amp/2.0);
            ac->syncPosition(PairsAccessor::Host);
            ac->setPosition(plate_idx, {0.0, 0.0, posz});
            ac->syncPosition(PairsAccessor::Host);
        }

        // Calculate forces
        // ------------------------------------------------------------------
        pairs_sim->gravity(); 
        pairs_sim->spring_dashpot(); 

        // Apply external force to Box
        // ------------------------------------------------------------------
        auto box_idx = ac->uidToIdxLocal(box_uid);
        ac->syncForce(PairsAccessor::Host);
        ac->syncTorque(PairsAccessor::Host);
        pairs::Vector3<double> external_force(500, 0, 0);
        ac->setForce(box_idx, ac->getForce(box_idx) + external_force);
        ac->setTorque(box_idx, {0.0, 0.0, 0.0});    // Reset torque to prevent rotation
        ac->syncForce(PairsAccessor::Host);
        ac->syncTorque(PairsAccessor::Host);

        // Update positions and reneighbor
        // ------------------------------------------------------------------
        pairs_sim->euler(dt); 
        pairs_sim->reneighbor(); 

        // VTK
        // ------------------------------------------------------------------
        if (t % vtk_freq==0){
            pairs::vtk_write_data(pairs_runtime, "output/local_spheres", 0, pairs_sim->nlocal(), t);

            if(rank==0){
                ac->syncPosition(PairsAccessor::Host);
                std::cout << "Timestep: " << t << " - Time = " << dt*t << " - Box position = " << ac->getPosition(box_idx) << std::endl;
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

        ac->syncUid(PairsAccessor::Host);
        auto plate_idx = ac->uidToIdxLocal(plate_uid);
        auto box_idx = ac->uidToIdxLocal(box_uid);

        std::cout << "box final position = " << ac->getPosition(box_idx) << std::endl;
        std::cout << "plate final position = " << ac->getPosition(plate_idx) << std::endl;
    }

    ac->end();
    pairs_sim->end();
}