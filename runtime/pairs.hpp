#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#pragma once

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
};

enum DataLayout {
    Invalid = -1,
    AoS = 0,
    SoA
};

template<typename real_t>
class Vector3 {
private:
    real_t x, y, z;

public:
    Vector3() : x((real_t) 0.0), y((real_t) 0.0), z((real_t) 0.0) {}
    Vector3(real_t v) : x(v), y(v), z(v) {}
    Vector3(real_t x_, real_t y_, real_t z_) : x(x_), y(y_), z(z_) {}
};

class Property {
protected:
    property_t id;
    std::string name;
    void *ptr;
    PropertyType type;
    layout_t layout;
    size_t sx, sy;

public:
    Property(property_t id_, std::string name_, void *ptr_, PropertyType type_) :
        id(id_),
        name(name_),
        ptr(ptr_),
        type(type_),
        layout(Invalid) {}

    Property(property_t id_, std::string name_, void *ptr_, PropertyType type_, layout_t layout_, size_t sx_, size_t sy_) :
        id(id_),
        name(name_),
        ptr(ptr_),
        type(type_),
        layout(layout_),
        sx(sx_),
        sy(sy_) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getPointer() { return ptr; }
    void setPointer(void *ptr_) { ptr = ptr_; }
    void setSizes(int sx_, int sy_) { sx = sx_, sy = sy_; }
    PropertyType getType() { return type; }
    layout_t getLayout() { return layout; }
};

class IntProperty : public Property {
public:
    inline int &operator()(int i) { return static_cast<int *>(ptr)[i]; }
};

class FloatProperty : public Property {
public:
    inline double &operator()(int i) { return static_cast<double *>(ptr)[i]; }
};

class VectorProperty : public Property {
public:
    double &operator()(int i, int j) {
        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
        double *dptr = static_cast<double *>(ptr);
        if(layout == AoS) { return dptr[i * sy + j]; }
        if(layout == SoA) { return dptr[j * sx + i]; }
        PAIRS_ERROR("VectorProperty::operator[]: Invalid data layout!");
        return dptr[0];
    }
};

class PairsSim {
private:
    std::vector<Property> properties;

public:
    PairsSim() {}
    void addProperty(Property prop) { properties.push_back(prop); }
    void updateProperty(property_t id, void *ptr, int sx = 0, int sy = 0) {
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) {
            return p.getId() == id;
        });

        PAIRS_ASSERT(p != std::end(properties));
        p->setPointer(ptr);
        p->setSizes(sx, sy);
    }

    Property &getProperty(property_t id) {
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) {
            return p.getId() == id;
        });

        PAIRS_ASSERT(p != std::end(properties));
        return *p;
    }

    Property &getPropertyByName(std::string name) {
        auto p = std::find_if(properties.begin(), properties.end(), [name](Property p) {
            return p.getName() == name;
        });

        PAIRS_ASSERT(p != std::end(properties));
        return *p;
    }

    inline IntProperty &getAsIntegerProperty(Property &prop) { return static_cast<IntProperty&>(prop); }
    inline FloatProperty &getAsFloatProperty(Property &prop) { return static_cast<FloatProperty&>(prop); }
    inline VectorProperty &getAsVectorProperty(Property &prop) { return static_cast<VectorProperty&>(prop); }
    inline IntProperty &getIntegerProperty(property_t property) { return static_cast<IntProperty&>(getProperty(property)); }
    inline FloatProperty &getFloatProperty(property_t property) { return static_cast<FloatProperty&>(getProperty(property)); }
    inline VectorProperty &getVectorProperty(property_t property) { return static_cast<VectorProperty&>(getProperty(property)); }
};

}
