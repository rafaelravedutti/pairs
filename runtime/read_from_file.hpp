#include <stdio.h>
//---
#include "pairs.hpp"

namespace pairs {

size_t read_particle_data(PairsSim *ps, const char *filename, double *grid_buffer, property_t properties[], size_t nprops) {
    std::ifstream in_file(filename);
    std::string line;
    size_t n = 0;

    if(in_file.is_open()) {
        while(getline(in_file, line)) {
            int i = 0;
            char *in0 = strtok(line, ",");
            while(in0 != NULL) {
                if(grid_data_unread) {
                    PAIRS_ASSERT(i < ps->getNumDims() * 2);
                    grid_buffer[i] = std::stod(in0);
                } else {
                    PAIRS_ASSERT(i < nprops);
                    property_t p = properties[i];
                    auto prop_type = ps->getProperty(p)->getType();

                    switch(prop_type) {
                        case Prop_Vector:
                            auto vector_ptr = ps->getVectorPropertyPtr(p);
                            char *in1 = strtok(NULL, ",");
                            PAIRS_ASSERT(in1 != NULL);
                            char *in2 = strtok(NULL, ",");
                            PAIRS_ASSERT(in2 != NULL);
                            vector_ptr[n] = Vector3(std::stod(in0), std::stod(in1), std::stod(in2));
                            break;

                        case Prop_Integer:
                            auto int_ptr = ps->getIntegerPropertyPtr(p);
                            int_ptr[n] = std::stoi(in0);
                            break;

                        case Prop_Float:
                            auto float_ptr = ps->getFloatPropertyPtr(p);
                            float_ptr[n] = std::stod(in0);
                            break;

                        default:
                            fprintf(stderr, "read_particle_data(): Invalid property type!");
                            return -1;
                    }
                }

                in0 = strtok(NULL, ",");
                i++;
            }

            n++;
        }

        in_file.close();
    }

    return n;
}

}
