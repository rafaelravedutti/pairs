#include "pairs_common.hpp"

#pragma once

namespace pairs {

template<typename real_t>
class Vector3 {
private:
    real_t x, y, z;

public:
    Vector3() : x((real_t) 0.0), y((real_t) 0.0), z((real_t) 0.0) {}
    Vector3(real_t v) : x(v), y(v), z(v) {}
    Vector3(real_t x_, real_t y_, real_t z_) : x(x_), y(y_), z(z_) {}
};

}
