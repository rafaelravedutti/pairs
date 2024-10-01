#include "pairs.hpp"

#pragma once

namespace pairs {

void create_halfspace(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double nx, double ny, double nz, 
                    int type, int flag);

void create_particle(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double vx, double vy, double vz, 
                    double density, double radius, int type, int flag);

}