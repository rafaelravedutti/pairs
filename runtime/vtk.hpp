#include "pairs.hpp"

#pragma once

namespace pairs {

void vtk_write_data(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency);

}
