#include <iostream>
#include <vector>

#define PAIRS_ASSERT(a)

namespace pairs {

typedef int property_t;
typedef int layout_t;

class PairsSim {
    PairsSim() : {}

public:
    Property getProperty(property_t id) {
        return std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
    }

    Property getPropertyByName(std::string name) {
        return std::find_if(properties.begin(), properties.end(), [name](Property p) { return p.getName() == name; });
    }

    VectorPtr getFloatPropertyMutablePtr(property_t property) {
        return static_cast<VectorPtr>(getProperty(property).getPointer());
    }

    int *getIntegerPropertyMutablePtr(property_t property) {
        return static_cast<int *>(getProperty(property).getPointer());
    }

    double *getFloatPropertyMutablePtr(property_t property) {
        return static_cast<double *>(getProperty(property).getPointer());
    }

private:
    std::vector<Property> properties;
};

class Property {
    Property(property_t id_, std::string name_) : id(id_), name(name_), layout(-1), ptr(nullptr) {}
    Property(property_t id_, std::string name_, layout_t layout_) : id(id_), name(name_), layout(layout_), ptr(nullptr) {}
    Property(property_t id_, std::string name_, void *ptr_) : id(id_), name(name_), layout(-1), ptr(ptr_) {}
    Property(property_t id_, std::string name_, layout_t layout_, void *ptr_) : id(id_), name(name_), layout(layout_), ptr(ptr_) {}

public:
    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getPointer() { return ptr; }
    void setPointer(void *ptr_) { ptr = ptr_; }

private:
    property_t id;
    layout_t layout;
    std::string name;
    void *ptr;
};

template<typename real_t>
class Vector3 {
    Vector3() : x((real_t) 0.0), y((real_t) 0.0), z((real_t) 0.0) : {}
    Vector3(real_t v) : x(v), y(v), z(v) : {}
    Vector3(real_t x_, real_t y_, real_t z_) : x(x_), y(y_), z(z_) {}

private:
    real_t x, y, z;
};

}
