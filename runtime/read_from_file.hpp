#include "pairs.hpp"
#include "pairs_common.hpp"

#pragma once

namespace pairs {

void read_grid_data(PairsRuntime *ps, const char *filename, real_t *grid_buffer);

size_t read_particle_data(
    PairsRuntime *ps, const char *filename, const property_t properties[],
    size_t nprops, int shape_id, int start);

}
