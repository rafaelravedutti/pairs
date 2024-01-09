#include "pairs.hpp"

#pragma once

using namespace std;

namespace pairs {

void print_stats(PairsSimulation *ps, int nlocal, int nghost) {
    int min_nlocal = nlocal;
    int max_nlocal = nlocal;
    int min_nghost = nghost;
    int max_nghost = nghost;
    int nglobal;

    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        MPI_Allreduce(&nlocal, &nglobal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        min_nlocal = nglobal;
        MPI_Allreduce(&nlocal, &nglobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        max_nlocal = nglobal;
        MPI_Allreduce(&nghost, &nglobal, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        min_nghost = nglobal;
        MPI_Allreduce(&nghost, &nglobal, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        max_nghost = nglobal;
    }

    if(ps->getDomainPartitioner()->getRank() == 0) {
        std::cout << "Number of local particles: " << min_nlocal << " / " << max_nlocal << std::endl;
        std::cout << "Number of ghost particles: " << min_nghost << " / " << max_nghost << std::endl;
    }
}

}
