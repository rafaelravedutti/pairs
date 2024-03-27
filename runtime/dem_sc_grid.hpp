#include <iostream>
#include <math.h>
#include <random>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

namespace internal {

static std::mt19937 generator; // static std::mt19937_64 generator;

std::mt19937 & get_generator() {
    // std::mt19937_64
    return generator;
}

}

template< typename REAL_TYPE = real_t>
REAL_TYPE realRandom(
    const REAL_TYPE min = REAL_TYPE(0),
    const REAL_TYPE max = REAL_TYPE(1),
    std::mt19937& generator = internal::get_generator()) {

   static_assert(
        std::numeric_limits<REAL_TYPE>::is_specialized &&
        !std::numeric_limits<REAL_TYPE>::is_integer,
        "Floating point type required/expected!" );

   std::uniform_real_distribution<REAL_TYPE> distribution(min, max);

   REAL_TYPE value;
#ifdef _OPENMP
   #pragma omp critical (Random_random)
#endif
   { value = distribution( generator ); }

   return value;
}



template<typename REAL_TYPE> class RealRandom {
public:
   RealRandom(const std::mt19937::result_type& seed = std::mt19937::result_type()) { generator_.seed(seed); }
   REAL_TYPE operator()(const REAL_TYPE min = REAL_TYPE(0), const REAL_TYPE max = REAL_TYPE(1) ) {
      return realRandom(min, max, generator_);
   }
private:
   std::mt19937 generator_;
};

bool point_within_aabb(double point[], double aabb[]) {
    return point[0] >= aabb[0] && point[0] < aabb[3] &&
           point[1] >= aabb[1] && point[1] < aabb[4] &&
           point[2] >= aabb[2] && point[2] < aabb[5];
}

int dem_sc_grid(PairsRuntime *ps, double xmax, double ymax, double zmax, double spacing, double diameter, double min_diameter, double max_diameter, double initial_velocity, double particle_density, int ntypes) {
    auto uid = ps->getAsIntegerProperty(ps->getPropertyByName("uid"));
    auto shape = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
    auto types = ps->getAsIntegerProperty(ps->getPropertyByName("type"));
    auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    int last_uid = 1;
    int nparticles = 0;

    const double xmin = 0.0;
    const double ymin = 0.0;
    const double zmin = diameter;

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

    while(point_within_aabb(point, gen_domain)) {
        int particle_uid = last_uid;
        auto diameter = realRandom<real_t>(min_diameter, max_diameter);

        if(ps->getDomainPartitioner()->isWithinSubdomain(point[0], point[1], point[2])) {
            real_t rad = diameter * 0.5;
            uid(nparticles) = particle_uid;
            radius(nparticles) = rad;
            masses(nparticles) = ((4.0 / 3.0) * M_PI) * rad * rad * rad * particle_density;
            positions(nparticles, 0) = point[0];
            positions(nparticles, 1) = point[1];
            positions(nparticles, 2) = point[2];
            velocities(nparticles, 0) = 0.1 * realRandom<real_t>(-initial_velocity, initial_velocity);
            velocities(nparticles, 1) = 0.1 * realRandom<real_t>(-initial_velocity, initial_velocity);
            velocities(nparticles, 2) = -initial_velocity;
            types(nparticles) = rand() % ntypes;
            flags(nparticles) = 0;
            shape(nparticles) = 0; // sphere

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

        last_uid++;
    }

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
