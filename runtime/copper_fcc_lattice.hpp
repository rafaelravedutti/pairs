#include <iostream>
#include <math.h>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

/* Park/Miller RNG w/out MASKING, so as to be like f90s version */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

double myrandom(int* seed) {
    int k= (*seed) / IQ;
    double ans;

    *seed = IA * (*seed - k * IQ) - IR * k;
    if(*seed < 0) *seed += IM;
    ans = AM * (*seed);
    return ans;
}

void random_reset(int *seed, int ibase, double *coord) {
    int i;
    char *str = (char *) &ibase;
    int n = sizeof(int);
    unsigned int hash = 0;

    for (i = 0; i < n; i++) {
        hash += str[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    str = (char *) coord;
    n = 3 * sizeof(double);
    for (i = 0; i < n; i++) {
        hash += str[i];
        hash += (hash << 10);
        hash ^= (hash >> 6);
    }

    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);

    // keep 31 bits of unsigned int as new seed
    // do not allow seed = 0, since will cause hang in gaussian()

    *seed = hash & 0x7ffffff;
    if (!(*seed)) *seed = 1;

    // warm up the RNG

    for (i = 0; i < 5; i++) myrandom(seed);
    //save = 0;
}

double copper_fcc_lattice(PairsSimulation *ps, int nx, int ny, int nz, double xprd, double yprd, double zprd, double rho, int ntypes) {
    auto shape = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
    auto types = ps->getAsIntegerProperty(ps->getPropertyByName("type"));
    auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    double xlo = 0.0, xhi = xprd;
    double ylo = 0.0, yhi = yprd;
    double zlo = 0.0, zhi = zprd;
    int natoms = 0;
    //int natoms_expected = 4 * nx * ny * nz;

    double alat = pow((4.0 / rho), (1.0 / 3.0));
    int ilo = (int) (xlo / (0.5 * alat) - 1);
    int ihi = (int) (xhi / (0.5 * alat) + 1);
    int jlo = (int) (ylo / (0.5 * alat) - 1);
    int jhi = (int) (yhi / (0.5 * alat) + 1);
    int klo = (int) (zlo / (0.5 * alat) - 1);
    int khi = (int) (zhi / (0.5 * alat) + 1);

    ilo = MAX(ilo, 0);
    ihi = MIN(ihi, 2 * nx - 1);
    jlo = MAX(jlo, 0);
    jhi = MIN(jhi, 2 * ny - 1);
    klo = MAX(klo, 0);
    khi = MIN(khi, 2 * nz - 1);

    double xtmp, ytmp, ztmp, vxtmp, vytmp, vztmp;
    int i, j, k, m, n;
    int sx = 0; int sy = 0; int sz = 0;
    int ox = 0; int oy = 0; int oz = 0;
    int subboxdim = 8;

    while(oz * subboxdim <= khi) {
        k = oz * subboxdim + sz;
        j = oy * subboxdim + sy;
        i = ox * subboxdim + sx;

        if(((i + j + k) % 2 == 0) &&
            (i >= ilo) && (i <= ihi) &&
            (j >= jlo) && (j <= jhi) &&
            (k >= klo) && (k <= khi)) {

            xtmp = 0.5 * alat * i;
            ytmp = 0.5 * alat * j;
            ztmp = 0.5 * alat * k;

            if(ps->getDomainPartitioner()->isWithinSubdomain(xtmp, ytmp, ztmp)) {
                n = k * (2 * ny) * (2 * nx) + j * (2 * nx) + i + 1;
                for(m = 0; m < 5; m++) { myrandom(&n); }
                vxtmp = myrandom(&n);
                for(m = 0; m < 5; m++){ myrandom(&n); }
                vytmp = myrandom(&n);
                for(m = 0; m < 5; m++) { myrandom(&n); }
                vztmp = myrandom(&n);

                masses(natoms) = 1.0;
                positions(natoms, 0) = xtmp;
                positions(natoms, 1) = ytmp;
                positions(natoms, 2) = ztmp;
                velocities(natoms, 0) = vxtmp;
                velocities(natoms, 1) = vytmp;
                velocities(natoms, 2) = vztmp;
                types(natoms) = rand() % ntypes;
                flags(natoms) = 0;
                shape(natoms) = 2; // point mass
                natoms++;
            }
        }

        sx++;

        if(sx == subboxdim) { sx = 0; sy++; }
        if(sy == subboxdim) { sy = 0; sz++; }
        if(sz == subboxdim) { sz = 0; ox++; }
        if(ox * subboxdim > ihi) { ox = 0; oy++; }
        if(oy * subboxdim > jhi) { oy = 0; oz++; }
    }

    return natoms;
}

}
