#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "pairs.hpp"

#pragma once

namespace pairs {

size_t read_particle_data(PairsSim *ps, const char *filename, double *grid_buffer, const property_t properties[], size_t nprops) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;
    size_t n = 0;
    int read_grid_data = 0;

    if(in_file.is_open()) {
        while(getline(in_file, line)) {
            std::stringstream line_stream(line);
            std::string in0;
            int i = 0;

            while(std::getline(line_stream, in0, ',')) {
                if(!read_grid_data) {
                    PAIRS_ASSERT(i < ps->getNumDims() * 2);
                    grid_buffer[i] = std::stod(in0);
                    read_grid_data = 1;
                } else {
                    PAIRS_ASSERT(i < nprops);
                    property_t p_id = properties[i];
                    auto prop = ps->getProperty(p_id);
                    auto prop_type = prop.getType();

                    if(prop_type == Prop_Vector) {
                        auto vector_ptr = ps->getAsVectorProperty(prop);
                        std::string in1, in2;
                        std::getline(line_stream, in1, ',');
                        std::getline(line_stream, in2, ',');
                        vector_ptr(n, 0) = std::stod(in0);
                        vector_ptr(n, 1) = std::stod(in1);
                        vector_ptr(n, 2) = std::stod(in2);
                    } else if(prop_type == Prop_Integer) {
                        auto int_ptr = ps->getAsIntegerProperty(prop);
                        int_ptr(n) = std::stoi(in0);
                    } else if(prop_type == Prop_Float) {
                        auto float_ptr = ps->getAsFloatProperty(prop);
                        float_ptr(n) = std::stod(in0);
                    } else {
                        std::cerr << "read_particle_data(): Invalid property type!" << std::endl;
                        return -1;
                    }
                }

                i++;
            }

            n++;
        }

        in_file.close();
    }

    return n;
}

}
