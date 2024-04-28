#include "pairs.hpp"

#pragma once

namespace pairs {

double compute_thermo(
    PairsRuntime *ps, int nlocal, double xprd, double yprd, double zprd, int print);

void adjust_thermo(
    PairsRuntime *ps, int nlocal, double xprd, double yprd, double zprd, double temp);

}
