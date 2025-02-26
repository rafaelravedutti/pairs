#include "pairs_common.hpp"

#pragma once

namespace pairs {

class TrackedVariable {
protected:
    std::string name;
    void *ptr;

public:
    TrackedVariable(std::string name_, void *ptr_) : name(name_), ptr(ptr_) {}
    std::string getName() { return name; }
    void *getPointer() { return ptr; }
};

}
