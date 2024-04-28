#include <math.h>
#include <random>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"

#pragma once

namespace pairs {

namespace internal {

std::mt19937 & get_generator();

}

template<typename REAL_TYPE = real_t>
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

template<typename REAL_TYPE>
class RealRandom {
public:
    RealRandom(const std::mt19937::result_type& seed = std::mt19937::result_type()) {
        generator_.seed(seed);
    }

    REAL_TYPE operator()(const REAL_TYPE min = REAL_TYPE(0), const REAL_TYPE max = REAL_TYPE(1)) {
        return realRandom(min, max, generator_);
    }

private:
   std::mt19937 generator_;
};

bool point_within_aabb(double point[], double aabb[]);

int dem_sc_grid(PairsRuntime *ps, double xmax, double ymax, double zmax, double spacing, double diameter, double min_diameter, double max_diameter, double initial_velocity, double particle_density, int ntypes);

}
