#include "pairs.hpp"
#include "unique_id.hpp"

#pragma once

/* Park/Miller RNG w/out MASKING, so as to be like f90s version */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

namespace pairs {

double myrandom(int* seed);
void random_reset(int *seed, int ibase, double *coord);
double copper_fcc_lattice(
    PairsRuntime *ps, int nx, int ny, int nz, double xprd, double yprd, double zprd,
    double rho, int ntypes);

}
