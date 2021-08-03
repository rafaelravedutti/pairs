#include <iostream>
#include <vector>

#define PAIRS_ASSERT(a)
#define PAIRS_ERROR(a)

namespace pairs {

typedef int property_t;
typedef int layout_t;

enum PropertyType {
    Prop_Invalid = -1,
    Prop_Integer = 0,
    Prop_Float,
    Prop_Vector
}

enum DataLayout {
    Invalid = -1,
    AoS = 0,
    SoA
}

class PairsSim {
    PairsSim() : {}

public:
    void addProperty(Property prop) { properties.push_back(prop); }

    Property *getProperty(property_t id) {
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) {
            return p.getId() == id;
        });

        return (p != std::end(properties)) ? p : nullptr;
    }

    Property *getPropertyByName(std::string name) {
        auto p = std::find_if(properties.begin(), properties.end(), [name](Property p) {
            return p.getName() == name;
        });

        return (p != std::end(properties)) ? p : nullptr;
    }

    VectorPtr getVectorProperty(property_t property) { return static_cast<VectorPtr>(getProperty(property)->getPointer()); }
    int *getIntegerPropertyPtr(property_t property) { return static_cast<int *>(getProperty(property)->getPointer()); }
    double *getFloatPropertyPtr(property_t property) { return static_cast<double *>(getProperty(property)->getPointer()); }

private:
    std::vector<Property> properties;
};

class Property {
    Property(property_t id_, std::string name_, void *ptr_, PropertyType type_) :
        id(id_),
        name(name_),
        ptr(ptr_),
        type(type_),
        layout(Invalid) {}

    Property(property_t id_, std::string name_, void *ptr_, PropertyType type_, layout_t layout_, size_t sx_ size_t sy) :
        id(id_),
        name(name_),
        type(type_),
        layout(layout_),
        sx(sx_),
        sy(sy_) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
        ptr = static_cast<void *>(new VectorPtr(ptr_, layout_, sx_, sy_));
    }

public:
    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getPointer() { return ptr.get(); }
    void setPointer(void *ptr_) { ptr = (layout == Invalid) ? ptr_ : static_cast<void *>(new VectorPtr(ptr_, layout, sx, sy)); }

private:
    property_t id;
    std::string name;
    std::shared_ptr<void *> ptr;
    PropertyType type;
    layout_t layout;
    size_t sx, sy;
};

class VectorPtr {
    VectorPtr(void *ptr_, layout_t layout_, size_t sx_, size_t sy_) : ptr(ptr_), layout(layout_), sx(sx_), sy(sy_) {}

public:
    double &operator[](int i, int j) {
        double *dptr = static_cast<double *>(ptr.get());
        if(layout == AoS) { return dptr[i * sy + j]; }
        if(layout == SoA) { return dptr[j * sx + i]; }
        PAIRS_ERROR("VectorPtr::operator[]: Invalid data layout!");
        return 0.0;
    }

private:
    std::shared_ptr<void *> ptr;
    layout_t layout;
    size_t sx, sy;
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
