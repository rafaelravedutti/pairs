#include <stdio.h>
//---
#include "pairs.hpp"

namespace pairs {

size_t read_particle_data(PairsSim *ps, const char *filename, property_t properties[], size_t nprops) {
    std::ifstream in_file(filename);
    std::string line;
    size_t n = 0;

    if(in_file.is_open()) {
        while(getline(in_file, line)) {
            int p = 0;
            char *in0 = strtok(line, ",");
            while(in0 != NULL) {
                PAIRS_ASSERT(p < nprops);
                prop_type = ps->getPropertyType(p);

                switch(prop_type) {
                    case PROPERTY_TYPE_VECTOR:
                        auto vector_ptr = ps->getVectorPropertyMutablePtr(p);
                        char *in1 = strtok(NULL, ",");
                        PAIRS_ASSERT(in1 != NULL);
                        char *in2 = strtok(NULL, ",");
                        PAIRS_ASSERT(in2 != NULL);
                        vector_ptr[n] = Vector3(std::stod(in0), std::stod(in1), std::stod(in2));
                        break;

                    case PROPERTY_TYPE_INTEGER:
                        auto int_ptr = ps->getIntegerPropertyMutablePtr(p);
                        int_ptr[n] = std::stoi(in0);
                        break;

                    case PROPERTY_TYPE_FLOAT:
                        auto float_ptr = ps->getFloatPropertyMutablePtr(p);
                        float_ptr[n] = std::stod(in0);
                        break;

                    default:
                        fprintf(stderr, "read_particle_data(): Invalid property type!");
                        return -1;
                }

                in0 = strtok(NULL, ",");
                p++;
            }

            n++;
        }

        in_file.close();
    }

    return n;
}

}
