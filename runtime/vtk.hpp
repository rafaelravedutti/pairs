#include <iomanip>
#include <iostream>
#include <fstream>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

void vtk_write_data(PairsSimulation *ps, const char *filename, int start, int end, int timestep) {
    std::string output_filename(filename);
    std::ostringstream filename_oss;
    filename_oss << filename << "_" << timestep << ".vtk";
    std::ofstream out_file(filename_oss.str());
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    const int n = end - start;

    ps->copyPropertyToHost(masses);
    ps->copyPropertyToHost(positions);

    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Particle data\n";
        out_file << "ASCII\n";
        out_file << "DATASET UNSTRUCTURED_GRID\n";
        out_file << "POINTS " << n << " double\n";

        for(int i = start; i < end; i++) {
            out_file << std::fixed << std::setprecision(4) << positions(i, 0) << " ";
            out_file << std::fixed << std::setprecision(4) << positions(i, 1) << " ";
            out_file << std::fixed << std::setprecision(4) << positions(i, 2) << "\n";
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
            out_file << std::fixed << std::setprecision(4) << masses(i) << "\n";
        }

        out_file << "\n\n";
        out_file.close();
    }
}

}
