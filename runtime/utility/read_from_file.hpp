#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"
#include "unique_id.hpp"

#pragma once

namespace pairs {

void write_boxes(PairsRuntime *pr, const char *filename);
void read_boxes(PairsRuntime *pr, const char *filename);
void write_spheres(PairsRuntime *pr, const char *filename);
void read_spheres(PairsRuntime *pr, const char *filename, std::array<double, 3> offset = {0.0, 0.0, 0.0});

void read_grid_data(PairsRuntime *ps, const char *filename, real_t *grid_buffer);

size_t read_particle_data(
    PairsRuntime *ps, const char *filename, const property_t properties[],
    int shape_id, int start);

}
