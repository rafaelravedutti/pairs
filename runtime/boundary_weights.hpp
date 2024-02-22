#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"

/*
#define INTERFACE_DIR "interfaces/"
#define INTERFACE_EXT ".hpp"
#define INTERFACE_FILE(a, b, c) a ## b ## c
#define INCLUDE_FILE(filename) #filename
#include INCLUDE_FILE(INTERFACE_FILE(INTERFACE_DIR, APPLICATION_REFERENCE, INTERFACE_EXT))
*/

// Always include last generated interfaces
#include "interfaces/last_generated.hpp"

#pragma once

#ifdef PAIRS_TARGET_CUDA
int cuda_compute_boundary_weights(
    real_t *position, int start, int end, int particle_capacity,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax);
#endif

namespace pairs {

void compute_boundary_weights(
    PairsSimulation *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    walberla::uint_t *comp_weight, walberla::uint_t *comm_weight) {

    const int particle_capacity = ps->getTrackedVariableAsInteger("particle_capacity");
    const int nlocal = ps->getTrackedVariableAsInteger("nlocal");
    const int nghost = ps->getTrackedVariableAsInteger("nghost");
    auto position_prop = ps->getPropertyByName("position");

    #ifndef PAIRS_TARGET_CUDA
    real_t *position_ptr = static_cast<real_t *>(position_prop.getHostPointer());

    *comp_weight = 0;
    *comm_weight = 0;

    for(int i = 0; i < nlocal; i++) {
        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                *comp_weight++;
        }
    }

    for(int i = nlocal; i < nlocal + nghost; i++) {
        real_t pos_x = pairs_host_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_host_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_host_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                *comm_weight++;
        }
    }
    #else
    real_t *position_ptr = static_cast<real_t *>(position_prop.getDevicePointer());

    ps->copyPropertyToDevice(position_prop, ReadOnly);

    *comp_weight = cuda_compute_boundary_weights(
        position_ptr, 0, nlocal, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);

    *comm_weight = cuda_compute_boundary_weights(
        position_ptr, nlocal, nlocal + nghost, particle_capacity, xmin, xmax, ymin, ymax, zmin, zmax);
    #endif
}

}
