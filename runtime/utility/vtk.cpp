#include <iomanip>
#include <iostream>
#include <fstream>
//---
#include "pairs.hpp"

namespace pairs {


void vtk_write_halo_cells(PairsRuntime *ps, const char *filename, int timestep, 
        int nhalo_cells, int *halo_cells, int *dim_cells, double *spacing, double *subdom){

    // if(ts%20 == 0)
    // vtk_write_halo_cells(pairs_runtime, "output/halo_cells", ts, 
    //     pobj->halo_ncells, pobj->halo_cells, pobj->dim_cells, pobj->spacing, pobj->subdom);

    std::ostringstream filename_oss;

    filename_oss << filename << "_";
    if(ps->getDomainPartitioner()->getWorldSize() > 1) {
        filename_oss << "r" << ps->getDomainPartitioner()->getRank() << "_";
    }

    filename_oss << timestep << ".vtk";
    std::ofstream out_file(filename_oss.str());

    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Halo cells\n";
        out_file << "ASCII\n";
        out_file << "DATASET POLYDATA\n";
        out_file << "POINTS " << nhalo_cells-1 << " double\n";

        out_file << std::fixed << std::setprecision(6);
        int halo_idx = 1;
        for(int i = 0; i < dim_cells[0]; i++) {
            for(int j = 0; j < dim_cells[1]; j++) {
                for(int k = 0; k < dim_cells[2]; k++) {
                    int flat_idx = i*dim_cells[1]*dim_cells[2] + j*dim_cells[2] + k + 1;
                    if(halo_cells[halo_idx] == flat_idx){
                        // Cell centers:
                        out_file << (i-0.5)*spacing[0] + subdom[0] << " ";
                        out_file << (j-0.5)*spacing[1] + subdom[2] << " ";
                        out_file << (k-0.5)*spacing[2] + subdom[4] << "\n";
                        ++halo_idx;
                    }
                }
            }
        }

        out_file << "\n\n";
        out_file.close();
    }
    else {
        std::cerr << "Failed to open " << filename_oss.str() << std::endl;
        exit(-1);
    }

}


void vtk_with_rotation(
    PairsRuntime *ps, Shapes shape, const char *filename, int start, int end, int timestep, int frequency) {

    std::string output_filename(filename);
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
    auto rotation_matrix = ps->getAsMatrixProperty(ps->getPropertyByName("rotation_matrix"));
    auto shapes = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
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
    ps->copyPropertyToHost(radius, ReadOnly);
    ps->copyPropertyToHost(rotation_matrix, ReadOnly);
    ps->copyPropertyToHost(shapes, ReadOnly);

    for(int i = start; i < end; i++) {
        if(shapes(i) != shape) {
            n--;
        }
    }

    if(out_file.is_open()) {
        out_file << "# vtk DataFile Version 2.0\n";
        out_file << "Particle data\n";
        out_file << "ASCII\n";
        out_file << "DATASET POLYDATA\n";
        out_file << "POINTS " << n << " double\n";

        out_file << std::fixed << std::setprecision(prec);
        for(int i = start; i < end; i++) {
            if (shapes(i) == shape) {
                out_file << positions(i, 0) << " ";
                out_file << positions(i, 1) << " ";
                out_file << positions(i, 2) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "POINT_DATA " << n << "\n";
        out_file << "SCALARS mass double 1\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            if (shapes(i) == shape) {
                out_file << masses(i) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "SCALARS radius double 1\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            if (shapes(i) == shape) {
                out_file << radius(i) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "TENSORS rotation float\n";
        for(int i = start; i < end; i++) {
            if (shapes(i) == shape) {
                out_file    << rotation_matrix(i, 0) << " " 
                            << rotation_matrix(i, 3) << " " 
                            << rotation_matrix(i, 6) << "\n";
            
                out_file    << rotation_matrix(i, 1) << " " 
                            << rotation_matrix(i, 4) << " " 
                            << rotation_matrix(i, 7) << "\n";
            
                out_file    << rotation_matrix(i, 2) << " " 
                            << rotation_matrix(i, 5) << " " 
                            << rotation_matrix(i, 8) << "\n";
            }
        }

        out_file << "\n\n";
        out_file.close();
    }
    else {
        std::cerr << "Failed to open " << filename_oss.str() << std::endl;
        exit(-1);
    }
}


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


void vtk_write_clump(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency) {
        std::string output_filename(filename);
        auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
        auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
        auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
        auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
        auto shapes = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
        auto rotation_matrix = ps->getAsMatrixProperty(ps->getPropertyByName("rotation_matrix"));
    
        const real_t * local_positions = static_cast<real_t *>((ps->getArrayByName("local_positions")).getHostPointer());
        const real_t * local_radius = static_cast<real_t *>((ps->getArrayByName("local_radius")).getHostPointer());
    
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
            if(shapes(i) != Shapes::Clump) {
                n--;
            }
        }
        
        int nb = 3;
        int num_data_points = nb * n;

        if(out_file.is_open()) {
            out_file << "# vtk DataFile Version 2.0\n";
            out_file << "Particle data\n";
            out_file << "ASCII\n";
            out_file << "DATASET POLYDATA\n";
            out_file << "POINTS " << num_data_points << " double\n";
    
            for(int i = start; i < end; i++) {
                if(shapes(i)==Shapes::Clump){
                    for(int sub_n=0; sub_n<nb; ++sub_n){
                        double local_pos[3] = {local_positions[3 * sub_n + 0], local_positions[3 * sub_n + 1], local_positions[3 * sub_n + 2]};
                        double p0 = local_pos[0]*rotation_matrix(i, 0) + local_pos[1]*rotation_matrix(i, 1) + local_pos[2]*rotation_matrix(i, 2);
                        double p1 = local_pos[0]*rotation_matrix(i, 3) + local_pos[1]*rotation_matrix(i, 4) + local_pos[2]*rotation_matrix(i, 5);
                        double p2 = local_pos[0]*rotation_matrix(i, 6) + local_pos[1]*rotation_matrix(i, 7) + local_pos[2]*rotation_matrix(i, 8);

                        out_file << std::fixed << std::setprecision(prec) << positions(i, 0) + p0 << " ";
                        out_file << std::fixed << std::setprecision(prec) << positions(i, 1) + p1 << " ";
                        out_file << std::fixed << std::setprecision(prec) << positions(i, 2) + p2 << "\n";
                    }
            
                }
            }
    
            out_file << "\n\n";
            out_file << "POINT_DATA " << num_data_points << "\n";
            out_file << "SCALARS radius double 1\n";
            out_file << "LOOKUP_TABLE default\n";
            for(int i = start; i < end; i++) {
                if(shapes(i)==Shapes::Clump){
                    for(int sub_n=0; sub_n<nb; ++sub_n){
                        out_file << std::fixed << std::setprecision(prec) << local_radius[sub_n] << "\n";
                    }
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


void vtk_write_sphere(
    PairsRuntime *ps, const char *filename, int start, int end, int timestep, int frequency) {

    std::string output_filename(filename);
    auto masses = ps->getAsFloatProperty(ps->getPropertyByName("mass"));
    auto positions = ps->getAsVectorProperty(ps->getPropertyByName("position"));
    auto flags = ps->getAsIntegerProperty(ps->getPropertyByName("flags"));
    auto radius = ps->getAsFloatProperty(ps->getPropertyByName("radius"));
    auto shapes = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
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
        if(shapes(i)!=Shapes::Sphere){
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
            if(shapes(i)==Shapes::Sphere){
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
            if(shapes(i)==Shapes::Sphere){
                out_file << std::fixed << std::setprecision(prec) << masses(i) << "\n";
            }
        }

        out_file << "\n\n";
        out_file << "SCALARS radius double 1\n";
        out_file << "LOOKUP_TABLE default\n";
        for(int i = start; i < end; i++) {
            if(shapes(i)==Shapes::Sphere){
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
