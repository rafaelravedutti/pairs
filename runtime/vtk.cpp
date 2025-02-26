#include <iomanip>
#include <iostream>
#include <fstream>
//---
#include "pairs.hpp"

namespace pairs {

void vtk_write_aabb(PairsRuntime *ps, const char *filename, int num,
    double xmin, double xmax, 
    double ymin, double ymax, 
    double zmin, double zmax){

    std::string output_filename(filename);
    const int prec = 8;
    std::ostringstream filename_oss;

    filename_oss << filename << "_" << num;
    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        filename_oss << "r" << ps->getDomainPartitioner()->getRank() ;
    }

    filename_oss <<".vtk";
    std::ofstream out_file(filename_oss.str());

    out_file << std::fixed << std::setprecision(prec);
    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Subdomains\n";
        out_file << "ASCII\n";
        out_file << "DATASET POLYDATA\n";
        out_file << "POINTS 8 double\n";

        out_file << xmin << " " << ymin << " " << zmin << "\n";
        out_file << xmax << " " << ymin << " " << zmin << "\n";
        out_file << xmax << " " << ymax << " " << zmin << "\n";
        out_file << xmin << " " << ymax << " " << zmin << "\n";
        out_file << xmin << " " << ymin << " " << zmax << "\n";
        out_file << xmax << " " << ymin << " " << zmax << "\n";
        out_file << xmax << " " << ymax << " " << zmax << "\n";
        out_file << xmin << " " << ymax << " " << zmax << "\n";

        out_file << "POLYGONS 6 30\n";

        out_file << "4 0 1 2 3 \n";
        out_file << "4 4 5 6 7 \n";
        out_file << "4 0 1 5 4 \n";
        out_file << "4 3 2 6 7 \n";
        out_file << "4 0 4 7 3 \n";
        out_file << "4 1 2 6 5 \n";

        out_file << "\n\n";
        out_file.close();
    }
    else {
        std::cerr << "vtk_write_aabb: Failed to open " << filename_oss.str() << std::endl;
        exit(-1);
    }

}

void vtk_write_subdom(PairsRuntime *ps, const char *filename, int timestep, int frequency){
    std::string output_filename(filename);
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

    double aabb[3][3];
    for (int d=0; d<3; ++d){
        aabb[d][0] = ps->getDomainPartitioner()->getSubdomMin(d);
        aabb[d][1] = ps->getDomainPartitioner()->getSubdomMax(d);
    }

    out_file << std::fixed << std::setprecision(prec);
    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Subdomains\n";
        out_file << "ASCII\n";
        out_file << "DATASET POLYDATA\n";
        out_file << "POINTS 8 double\n";

        out_file << aabb[0][0] << " " << aabb[1][0] << " " << aabb[2][0] << "\n";
        out_file << aabb[0][1] << " " << aabb[1][0] << " " << aabb[2][0] << "\n";
        out_file << aabb[0][1] << " " << aabb[1][1] << " " << aabb[2][0] << "\n";
        out_file << aabb[0][0] << " " << aabb[1][1] << " " << aabb[2][0] << "\n";
        out_file << aabb[0][0] << " " << aabb[1][0] << " " << aabb[2][1] << "\n";
        out_file << aabb[0][1] << " " << aabb[1][0] << " " << aabb[2][1] << "\n";
        out_file << aabb[0][1] << " " << aabb[1][1] << " " << aabb[2][1] << "\n";
        out_file << aabb[0][0] << " " << aabb[1][1] << " " << aabb[2][1] << "\n";

        out_file << "POLYGONS 6 30\n";

        out_file << "4 0 1 2 3 \n";
        out_file << "4 4 5 6 7 \n";
        out_file << "4 0 1 5 4 \n";
        out_file << "4 3 2 6 7 \n";
        out_file << "4 0 4 7 3 \n";
        out_file << "4 1 2 6 5 \n";

        out_file << "\n\n";
        out_file.close();
    }
    else {
        std::cerr << "vtk_write_subdoms: Failed to open " << filename_oss.str() << std::endl;
        exit(-1);
    }
}

void vtk_write_data(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency) {

    std::string output_filename(filename);
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
    auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
    const int prec = 8;
    int n = end - start;
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

    ps->copyPropertyToHost(masses, ReadOnly);
    ps->copyPropertyToHost(positions, ReadOnly);
    ps->copyPropertyToHost(flags, ReadOnly);
    ps->copyPropertyToHost(radius, ReadOnly);

    for(int i = start; i < end; i++) {
        if(flags(i) & flags::INFINITE) {
            n--;
        }
    }

    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Particle data\n";
        out_file << "ASCII\n";
        out_file << "DATASET POLYDATA\n";
        out_file << "POINTS " << n << " double\n";

        for(int i = start; i < end; i++) {
            if(!(flags(i) & flags::INFINITE)) {
                out_file << std::fixed << std::setprecision(prec) << positions(i, 0) << " ";
                out_file << std::fixed << std::setprecision(prec) << positions(i, 1) << " ";
                out_file << std::fixed << std::setprecision(prec) << positions(i, 2) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "POINT_DATA " << n << "\n";
        out_file << "SCALARS mass double 1\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            if(!(flags(i) & flags::INFINITE)) {
                out_file << std::fixed << std::setprecision(prec) << masses(i) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "SCALARS radius double 1\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            if(!(flags(i) & flags::INFINITE)) {
                out_file << std::fixed << std::setprecision(prec) << radius(i) << "\n";
            }
        }

        out_file << "\n\n";
        out_file.close();
    }
    else {
        std::cerr << "vtk_write_data: Failed to open " << filename_oss.str() << std::endl;
        exit(-1);
    }
}

}
