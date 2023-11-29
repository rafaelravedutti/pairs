#include <iostream>
#include <math.h>
#include <mpi.h>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

double compute_thermo(PairsSimulation *ps, int nlocal, double xprd, double yprd, double zprd, int print) {
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    int natoms = nlocal;

    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        int global_natoms;
        MPI_Allreduce(&natoms, &global_natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        natoms = global_natoms;
    }

    const double mvv2e = 1.0;
    const double dof_boltz = (natoms * 3 - 3);
    const double t_scale = mvv2e / dof_boltz;
    const double p_scale = 1.0 / 3 / xprd / yprd / zprd;
    //const double e_scale = 0.5;
    double t = 0.0, p;

    ps->copyPropertyToHost(masses, false);
    ps->copyPropertyToHost(velocities, false);

    for(int i = 0; i < nlocal; i++) {
        t += masses(i) * (  velocities(i, 0) * velocities(i, 0) +
                            velocities(i, 1) * velocities(i, 1) +
                            velocities(i, 2) * velocities(i, 2)   );
    }

    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        double global_t;
        MPI_Allreduce(&t, &global_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        t = global_t;
    }

    t = t * t_scale;
    if(print == 1 && ps->getDomainPartitioner()->getRank() == 0) {
        p = (t * dof_boltz) * p_scale;
        std::cout << t << "\t" << p << std::endl;
    }

    return t;
}

void adjust_thermo(PairsSimulation *ps, int nlocal, double xprd, double yprd, double zprd, double temp) {
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    double vxtot = 0.0;
    double vytot = 0.0;
    double vztot = 0.0;
    double tmp;
    int natoms = nlocal;

    for(int i = 0; i < nlocal; i++) {
        vxtot += velocities(i, 0);
        vytot += velocities(i, 1);
        vztot += velocities(i, 2);
    }

    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        int global_natoms;
        MPI_Allreduce(&natoms, &global_natoms, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        natoms = global_natoms;
        MPI_Allreduce(&vxtot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vxtot = tmp / natoms;
        MPI_Allreduce(&vytot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vytot = tmp / natoms;
        MPI_Allreduce(&vztot, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vztot = tmp / natoms;
    } else {
        vxtot /= natoms;
        vytot /= natoms;
        vztot /= natoms;
    }

    for(int i = 0; i < nlocal; i++) {
        velocities(i, 0) -= vxtot;
        velocities(i, 1) -= vytot;
        velocities(i, 2) -= vztot;
    }

    double t = pairs::compute_thermo(ps, nlocal, xprd, yprd, zprd, 0);
    double factor = sqrt(temp / t);

    for(int i = 0; i < nlocal; i++) {
        velocities(i, 0) *= factor;
        velocities(i, 1) *= factor;
        velocities(i, 2) *= factor;
    }
}

}
