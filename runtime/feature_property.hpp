#include "pairs_common.hpp"

#pragma once

namespace pairs {

class FeatureProperty {
protected:
    property_t id;
    std::string name;
    void *h_ptr, *d_ptr;
    PropertyType type;
    size_t nkinds, array_size;

public:
    FeatureProperty(property_t id_, std::string name_, void *h_ptr_, void *d_ptr_, PropertyType type_, int nkinds_, int array_size_) :
        id(id_),
        name(name_),
        h_ptr(h_ptr_),
        d_ptr(d_ptr_),
        type(type_),
        nkinds(nkinds_),
        array_size(array_size_) {}

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getHostPointer() { return h_ptr; }
    void *getDevicePointer() { return d_ptr; }
    PropertyType getType() { return type; }
    const int getNumberOfKinds() { return nkinds; }
    const int getArraySize() { return array_size; }
};

}
