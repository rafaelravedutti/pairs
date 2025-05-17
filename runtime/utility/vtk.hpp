#include "pairs.hpp"

#pragma once

namespace pairs {

void vtk_with_rotation(
    PairsRuntime *ps, Shapes shape, const char *filename, int start, int end, int timestep, int frequency=1);

void vtk_write_halo_cells(PairsRuntime *ps, const char *filename, int timestep, 
    int nhalo_cells, int *halo_cells, int *dim_cells, double *spacing, double *subdom);

void vtk_write_aabb(PairsRuntime *ps, const char *filename, int num,
    double xmin, double xmax, 
    double ymin, double ymax, 
    double zmin, double zmax);

void vtk_write_sphere(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency=1);

void vtk_write_clump(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency=1);

void vtk_write_subdom(PairsRuntime *ps, const char *filename, int timestep, int frequency=1);

void vtk_write_data(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency=1);

}
