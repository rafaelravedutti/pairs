#pragma once

#include "pairs.hpp"
#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "../pairs.hpp"
#include "../pairs_common.hpp"


namespace pairs {

void compute_boundary_weights(
    PairsRuntime *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    long unsigned int *comp_weight, long unsigned int *comm_weight);

void determine_non_empty_aabbs(PairsRuntime *ps, int num_aabbs, real_t *aabbs, int *non_empty_aabbs);

}
