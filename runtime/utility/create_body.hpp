#include "pairs.hpp"
#include "unique_id.hpp"

#pragma once

namespace pairs {

id_t create_halfspace(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double nx, double ny, double nz, 
                    int type, int flag);

id_t create_sphere(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double vx, double vy, double vz, 
                    double density, double radius, int type, int flag);

id_t create_box(PairsRuntime *pr, 
                    double x, double y, double z, 
                    double vx, double vy, double vz, 
                    double ex, double ey, double ez, 
                    double density, int type, int flag);

id_t create_clump(PairsRuntime *pr, 
    double x, double y, double z, 
    double vx, double vy, double vz, 
    double density, double radius, int type, int flag);

}
