#include "pairs_common.hpp"

#pragma once

namespace pairs {

class Array {
protected:
    array_t id;
    std::string name;
    void *h_ptr, *d_ptr;
    size_t size;
    bool is_static;

public:
    Array(array_t id_, std::string name_, void *h_ptr_, void *d_ptr_, size_t size_, bool is_static_ = false) :
        id(id_),
        name(name_),
        h_ptr(h_ptr_),
        d_ptr(d_ptr_),
        size(size_),
        is_static(is_static_) {

        PAIRS_ASSERT(size_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getHostPointer() { return h_ptr; }
    void *getDevicePointer() { return d_ptr; }
    void setPointers(void *h_ptr_, void *d_ptr_) { h_ptr = h_ptr_, d_ptr = d_ptr_; }
    void setSize(size_t size_) { size = size_; }
    size_t getSize() { return size; };
    bool isStatic() { return is_static; }
};

}
