#include "pairs.hpp"

/*
#define INTERFACE_DIR "interfaces/"
#define INTERFACE_EXT ".hpp"
#define INTERFACE_FILE(a, b, c) a ## b ## c
#define INCLUDE_FILE(filename) #filename
#include INCLUDE_FILE(INTERFACE_FILE(INTERFACE_DIR, APPLICATION_REFERENCE, INTERFACE_EXT))
*/

#pragma once

namespace pairs {

void compute_boundary_weights(
    PairsRuntime *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    long unsigned int *comp_weight, long unsigned int *comm_weight){
        std::cerr<< "TODO: boundary weights should be generated" << std::endl;
        exit(-1);
    };

}
