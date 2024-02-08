#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"
#include "gen/interfaces.hpp"

#pragma once

namespace pairs {

void compute_boundary_weights(
    PairsSimulation *ps,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax,
    int *comp_weight, int *comm_weight) {

    const int particle_capacity = ps->getParticleCapacity();
    const int nlocal = ps->getNumberOfLocalParticles();
    const int nghost = ps->getNumberOfGhostParticles();
    auto position_prop = ps->getPropertyByName("position");

    //ps->copyPropertyToDevice(position_prop, Ignore);

    *comp_weight = 0;
    *comm_weight = 0;

    for(int i = 0; i < nlocal; i++) {
        real_t pos_x = pairs_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                *comp_weight++;
        }
    }

    for(int i = nlocal; i < nlocal + nghost; i++) {
        real_t pos_x = pairs_interface::get_position(position_ptr, i, 0, particle_capacity);
        real_t pos_y = pairs_interface::get_position(position_ptr, i, 1, particle_capacity);
        real_t pos_z = pairs_interface::get_position(position_ptr, i, 2, particle_capacity);

        if( pos_x > xmin && pos_x <= xmax &&
            pos_y > ymin && pos_y <= ymax &&
            pos_z > zmin && pos_z <= zmax) {
                *comm_weight++;
        }
    }
}

}