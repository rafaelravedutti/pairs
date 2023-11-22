#include <iostream>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

void compute_thermo(PairsSimulation *ps, int nlocal) {
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto velocities = ps->getAsVectorProperty(ps->getPropertyByName("linear_velocity"));
    const int natoms = 131072;
    const double xprd = 53.747078;
    const double yprd = 53.747078;
    const double zprd = 53.747078;
    const double mvv2e = 1.0;
    const double dof_boltz = (natoms * 3 - 3);
    const double t_scale = mvv2e / dof_boltz;
    const double p_scale = 1.0 / 3 / xprd / yprd / zprd;
    const double e_scale = 0.5;
    double t = 0.0, p;

    ps->copyPropertyToHost(masses);
    ps->copyPropertyToHost(velocities);

    for(int i = 0; i < nlocal; i++) {
        t += masses(i) * (  velocities(i, 0) * velocities(i, 0) +
                            velocities(i, 1) * velocities(i, 1) +
                            velocities(i, 2) * velocities(i, 2)   );
    }

    t = t * t_scale;
    p = (t * dof_boltz) * p_scale;
    std::cout << t << "\t" << p << std::endl;
}

}
