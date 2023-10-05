#include <iostream>
#include <string.h>
#include <fstream>
#include <sstream>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"

#pragma once

namespace pairs {

void read_grid_data(PairsSimulation *ps, const char *filename, double *grid_buffer) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;

    if(in_file.is_open()) {
        std::getline(in_file, line);
        std::stringstream line_stream(line);
        std::string in0;
        int i = 0;

        while(std::getline(line_stream, in0, ',')) {
            //PAIRS_ASSERT(i < ndims * 2);
            grid_buffer[i] = std::stod(in0);
            i++;
        }

        in_file.close();
    }
}

size_t read_particle_data(PairsSimulation *ps, const char *filename, const property_t properties[], size_t nprops, int shape_id) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;
    auto shape_ptr = ps->getAsIntegerProperty(ps->getPropertyByName("shape"));
    size_t n = 0;

    if(in_file.is_open()) {
        //std::getline(in_file, line);
        while(std::getline(in_file, line)) {
            std::stringstream line_stream(line);
            std::string in0;
            int within_domain = 1;
            int i = 0;

            while(std::getline(line_stream, in0, ',')) {
                property_t p_id = properties[i];
                auto prop = ps->getProperty(p_id);
                auto prop_type = prop.getType();

                if(prop_type == Prop_Vector) {
                    auto vector_ptr = ps->getAsVectorProperty(prop);
                    std::string in1, in2;
                    std::getline(line_stream, in1, ',');
                    std::getline(line_stream, in2, ',');
                    real_t x = std::stod(in0);
                    real_t y = std::stod(in1);
                    real_t z = std::stod(in2);
                    vector_ptr(n, 0) = x;
                    vector_ptr(n, 1) = y;
                    vector_ptr(n, 2) = z;

                    if(prop.getName() == "position") {
                        within_domain = ps->getDomainPartitioner()->isWithinSubdomain(x, y, z);
                    }
                } else if(prop_type == Prop_Integer) {
                    auto int_ptr = ps->getAsIntegerProperty(prop);
                    int_ptr(n) = std::stoi(in0);
                } else if(prop_type == Prop_Float) {
                    auto float_ptr = ps->getAsFloatProperty(prop);
                    float_ptr(n) = std::stod(in0);
                } else {
                    std::cerr << "read_particle_data(): Invalid property type!" << std::endl;
                    return 0;
                }

                i++;
            }

            if(within_domain) {
                shape_ptr(n++) = shape_id;
            }
        }

        in_file.close();
    }

    return n;
}

/*
size_t read_feature_data(PairsSimulation *ps, const char *filename, const int feature_id, const property_t properties[], size_t nprops) {
    std::ifstream in_file(filename, std::ifstream::in);
    std::string line;

    if(in_file.is_open()) {
        while(std::getline(in_file, line)) {
            std::stringstream line_stream(line);
            std::string istr, jstr, in0;
            std::getline(line_stream, istr, ',');
            std::getline(line_stream, jstr, ',');
            int i = std::stoi(istr);
            int j = std::stoi(jstr);

            while(std::getline(line_stream, in0, ',')) {
                property_t p_id = properties[i];
                auto prop = ps->getProperty(p_id);
                auto prop_type = prop.getType();

                if(prop_type == Prop_Vector) {
                    auto vector_ptr = ps->getAsVectorFeatureProperty(prop);
                    std::string in1, in2;
                    std::getline(line_stream, in1, ',');
                    std::getline(line_stream, in2, ',');
                    real_t x = std::stod(in0);
                    real_t y = std::stod(in1);
                    real_t z = std::stod(in2);
                    vector_ptr(i, j, 0) = x;
                    vector_ptr(i, j, 1) = y;
                    vector_ptr(i, j, 2) = z;
                } else if(prop_type == Prop_Integer) {
                    auto int_ptr = ps->getAsIntegerFeatureProperty(prop);
                    int_ptr(i, j) = std::stoi(in0);
                } else if(prop_type == Prop_Float) {
                    auto float_ptr = ps->getAsFloatFeatureProperty(prop);
                    float_ptr(i, j) = std::stod(in0);
                } else {
                    std::cerr << "read_feature_data(): Invalid property type!" << std::endl;
                    return 0;
                }
            }
        }

        in_file.close();
    }

    return n;
}
*/

}
