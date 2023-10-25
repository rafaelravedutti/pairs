#include <iomanip>
#include <iostream>
#include <fstream>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

void vtk_write_data(PairsSimulation *ps, const char *filename, int start, int end, int timestep, int frequency) {
    std::string output_filename(filename);
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    const int n = end - start;
    const int prec = 8;
    std::ostringstream filename_oss;

    if(frequency != 0 && timestep % frequency != 0) {
        return;
    }

    filename_oss << filename << "_";
    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        filename_oss << "r" << ps->getDomainPartitioner()->getRank() << "_";
    }

    filename_oss << timestep << ".vtk";
    std::ofstream out_file(filename_oss.str());

    ps->copyPropertyToHost(masses);
    ps->copyPropertyToHost(positions);

    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Particle data\n";
        out_file << "ASCII\n";
        out_file << "DATASET UNSTRUCTURED_GRID\n";
        out_file << "POINTS " << n << " double\n";

        for(int i = start; i < end; i++) {
            out_file << std::fixed << std::setprecision(prec) << positions(i, 0) << " ";
            out_file << std::fixed << std::setprecision(prec) << positions(i, 1) << " ";
            out_file << std::fixed << std::setprecision(prec) << positions(i, 2) << "\n";
        }

        out_file << "\n\n";
        out_file << "CELLS " << n << " " << (n * 2) << "\n";
        for(int i = start; i < end; i++) {
            out_file << "1 " << (i - start) << "\n";
        }

        out_file << "\n\n";
        out_file << "CELL_TYPES " << n << "\n";
        for(int i = start; i < end; i++) {
            out_file << "1\n";
        }

        out_file << "\n\n";
        out_file << "POINT_DATA " << n << "\n";
        out_file << "SCALARS mass double\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            out_file << std::fixed << std::setprecision(prec) << masses(i) << "\n";
        }

        out_file << "\n\n";
        out_file.close();
    }
}

}
