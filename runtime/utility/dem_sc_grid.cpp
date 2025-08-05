#include <iostream>
//---
#include "dem_sc_grid.hpp"

namespace pairs {

namespace internal {

static std::mt19937 generator; // static std::mt19937_64 generator;

std::mt19937 & get_generator() {
    // std::mt19937_64
    return generator;
}

}

bool point_within_aabb(double point[], double aabb[]) {
    return point[0] >= aabb[0] && point[0] < aabb[3] &&
           point[1] >= aabb[1] && point[1] < aabb[4] &&
           point[2] >= aabb[2] && point[2] < aabb[5];
}

int dem_sc_grid(PairsRuntime *ps, double xmax, double ymax, double zmax, 
    double spacing, double diameter, double min_diameter, double max_diameter, 
    double initial_velocity, double particle_density, int ntypes, bool lower_triangular) {
        
    auto uids = ps->getAsUInt64Property(ps->getPropertyByName("uid"));
    auto shapes = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
    auto types = ps->getAsIntegerProperty(ps->getPropertyByName("type"));
    auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    int nparticles = ps->getTrackedVariableAsInteger("nlocal");
    int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");

    const double xmin = 0.0;
    const double ymin = 0.0;
    const double zmin = 0.0;

    double gen_domain[] = {xmin, ymin, zmin, xmax, ymax, zmax};
    double ref_point[] = {spacing * 0.5, spacing * 0.5, spacing * 0.5};
    double sc_xmin = xmin - ref_point[0];
    double sc_ymin = ymin - ref_point[1];
    double sc_zmin = zmin - ref_point[2];

    int iret = (int)(ceil(sc_xmin / spacing));
    int jret = (int)(ceil(sc_ymin / spacing));
    int kret = (int)(ceil(sc_zmin / spacing));

    int i = iret;
    int j = jret;
    int k = kret;

    double point[3];
    point[0] = ref_point[0] + i * spacing;
    point[1] = ref_point[1] + j * spacing;
    point[2] = ref_point[2] + k * spacing;


    // EXPERIMENTAL: 3-sphre Clumps
    // ================================================================================================================
    // real_t * local_positions = static_cast<real_t *>((ps->getArrayByName("local_positions")).getHostPointer());
    // real_t * local_radius = static_cast<real_t *>((ps->getArrayByName("local_radius")).getHostPointer());

    // // double pos0[3] = {0, 0, -local_radius[1]};
    // // double pos1[3] = {0, 0, local_radius[0]};
    
    // double r = 0.05;
    // local_radius[0] = r;
    // local_radius[1] = r;
    // local_radius[2] = r;
    // double sq3 = r * sqrt(3.0)/3.0;
    // double pos0[3] = {-r,   -sq3,   0};
    // double pos1[3] = {r,    -sq3,    0};
    // double pos2[3] = {0,    2*sq3,  0};

    // local_positions[3*0 + 0] = pos0[0]; 
    // local_positions[3*0 + 1] = pos0[1]; 
    // local_positions[3*0 + 2] = pos0[2]; 

    // local_positions[3*1 + 0] = pos1[0]; 
    // local_positions[3*1 + 1] = pos1[1]; 
    // local_positions[3*1 + 2] = pos1[2]; 

    // local_positions[3*2 + 0] = pos2[0]; 
    // local_positions[3*2 + 1] = pos2[1]; 
    // local_positions[3*2 + 2] = pos2[2]; 

    // double total_mass = 4.0 / 3.0 * M_PI * local_radius[0] * local_radius[0] * local_radius[0] * particle_density;
    // total_mass += 4.0 / 3.0 * M_PI * local_radius[1] * local_radius[1] * local_radius[1] * particle_density;
    // total_mass += 4.0 / 3.0 * M_PI * local_radius[2] * local_radius[2] * local_radius[2] * particle_density;
    // std::cout << "total mass ============ " << total_mass << std::endl;

    // ================================================================================================================




    while(point_within_aabb(point, gen_domain)) {
        auto pdiam = realRandom<real_t>(min_diameter, max_diameter);

        if(ps->getDomainPartitioner()->isWithinSubdomain(point[0], point[1], point[2]) &&
            
            // If lower_triangular is true, only particles below the diagonal are accepted (in the x-z planes) 
            (lower_triangular ? ((point[2] <= (-zmax/xmax*point[0]+zmax))) : true)) {

            real_t rad = pdiam * 0.5;
            if(nparticles >= particle_capacity) {
                std::cerr << "Number of particles exceeded capacity (" << particle_capacity << ") in rank " << ps->getDomainPartitioner()->getRank() << std::endl;
                // TODO: resize properties, and all arrays that have particle_capacity as a dimension
                exit(-1);
            }
            uids(nparticles) = UniqueID::create(ps);
            radius(nparticles) = rad;
            positions(nparticles, 0) = point[0];
            positions(nparticles, 1) = point[1];
            positions(nparticles, 2) = point[2];
            velocities(nparticles, 0) = 0.1 * realRandom<real_t>(-initial_velocity, initial_velocity);
            velocities(nparticles, 1) = 0.1 * realRandom<real_t>(-initial_velocity, initial_velocity);
            velocities(nparticles, 2) = 0.1 * realRandom<real_t>(-initial_velocity, initial_velocity);
            types(nparticles) = rand() % ntypes;
            flags(nparticles) = 0;
            shapes(nparticles) = Shapes::Sphere;
            masses(nparticles) = ((4.0 / 3.0) * M_PI) * rad * rad * rad * particle_density;
            
            // flags(nparticles) = flags::FIXED;
            // shapes(nparticles) = Shapes::Clump;
            // masses(nparticles) = total_mass;

            /*
            std::cout << uid(nparticles) << "," << types(nparticles) << "," << masses(nparticles) << "," << radius(nparticles) << ","
                      << positions(nparticles, 0) << "," << positions(nparticles, 1) << "," << positions(nparticles, 2) << ","
                      << velocities(nparticles, 0) << "," << velocities(nparticles, 1) << "," << velocities(nparticles, 2) << ","
                      << flags(nparticles) << std::endl;
            */

            nparticles++;
        }

        ++i;
        point[0] = ref_point[0] + i * spacing;
        point[1] = ref_point[1] + j * spacing;
        point[2] = ref_point[2] + k * spacing;

        if(!point_within_aabb(point, gen_domain)) {
            i = iret;
            j++;
            point[0] = ref_point[0] + i * spacing;
            point[1] = ref_point[1] + j * spacing;
            point[2] = ref_point[2] + k * spacing;

            if(!point_within_aabb(point, gen_domain)) {
                j = jret;
                k++;
                point[0] = ref_point[0] + i * spacing;
                point[1] = ref_point[1] + j * spacing;
                point[2] = ref_point[2] + k * spacing;

                if(!point_within_aabb(point, gen_domain)) {
                    break;
                }
            }
        }
    }

    ps->setTrackedVariableAsInteger("nlocal", nparticles);

    int global_nparticles = nparticles;
    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        MPI_Allreduce(&nparticles, &global_nparticles, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    if(ps->getDomainPartitioner()->getRank() == 0) {
        std::cout << "DEM Simple-Cubic Grid" << std::endl;
        std::cout << "Domain size: <" << xmax << ", " << ymax << ", " << zmax << ">" << std::endl;
        std::cout << "Spacing: " << spacing << std::endl;
        std::cout << "Diameter: " << diameter
                  << " (min = " << min_diameter << ", max = " << max_diameter << ")" << std::endl;
        std::cout << "Initial velocity: " << initial_velocity << std::endl;
        std::cout << "Particle density: " << particle_density << std::endl;
        std::cout << "Number of types: " << ntypes << std::endl;
        std::cout << "Number of particles: " << global_nparticles << std::endl;
    }

    return nparticles;
}

}
