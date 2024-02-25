#include "pairs_common.hpp"

#pragma once

namespace pairs {

class Property {
protected:
    property_t id;
    std::string name;
    void *h_ptr, *d_ptr;
    PropertyType type;
    layout_t layout;
    size_t sx, sy;
    int vol;

public:
    Property(
        property_t id_, std::string name_, void *h_ptr_, void *d_ptr_,
        PropertyType type_, layout_t layout_, int vol_, size_t sx_, size_t sy_=1) :

        id(id_),
        name(name_),
        h_ptr(h_ptr_),
        d_ptr(d_ptr_),
        type(type_),
        layout(layout_),
        vol(vol_),
        sx(sx_), sy(sy_) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getHostPointer() { return h_ptr; }
    void *getDevicePointer() { return d_ptr; }
    void setPointers(void *h_ptr_, void *d_ptr_) { h_ptr = h_ptr_, d_ptr = d_ptr_; }
    void setSizes(size_t sx_, size_t sy_) { sx = sx_, sy = sy_; }
    size_t getTotalSize() { return sx * sy * getPrimitiveTypeSize(); };
    PropertyType getType() { return type; }
    layout_t getLayout() { return layout; }
    int isVolatile() { return vol != 0; }
    size_t getPrimitiveTypeSize() {
        return  (type == Prop_Integer) ? sizeof(int) :
                (type == Prop_Real) ? sizeof(real_t) :
                (type == Prop_Vector) ? sizeof(real_t) :
                (type == Prop_Matrix) ? sizeof(real_t) :
                (type == Prop_Quaternion) ? sizeof(real_t) : 0;
    }
};

class IntProperty : public Property {
public:
    inline int &operator()(int i) { return static_cast<int *>(h_ptr)[i]; }
};

class FloatProperty : public Property {
public:
    inline real_t &operator()(int i) { return static_cast<real_t *>(h_ptr)[i]; }
};

class VectorProperty : public Property {
public:
    real_t &operator()(int i, int j) {
        PAIRS_ASSERT(type != Prop_Invalid && layout != Invalid && sx > 0 && sy > 0);
        real_t *dptr = static_cast<real_t *>(h_ptr);
        if(layout == AoS) { return dptr[i * sy + j]; }
        if(layout == SoA) { return dptr[j * sx + i]; }
        PAIRS_ERROR("VectorProperty::operator[]: Invalid data layout!");
        return dptr[0];
    }
};

class MatrixProperty : public Property {
public:
    real_t &operator()(int i, int j) {
        PAIRS_ASSERT(type != Prop_Invalid && layout != Invalid && sx > 0 && sy > 0);
        real_t *dptr = static_cast<real_t *>(h_ptr);
        if(layout == AoS) { return dptr[i * sy + j]; }
        if(layout == SoA) { return dptr[j * sx + i]; }
        PAIRS_ERROR("MatrixProperty::operator[]: Invalid data layout!");
        return dptr[0];
    }
};

class QuaternionProperty : public Property {
public:
    real_t &operator()(int i, int j) {
        PAIRS_ASSERT(type != Prop_Invalid && layout != Invalid && sx > 0 && sy > 0);
        real_t *dptr = static_cast<real_t *>(h_ptr);
        if(layout == AoS) { return dptr[i * sy + j]; }
        if(layout == SoA) { return dptr[j * sx + i]; }
        PAIRS_ERROR("QuaternionProperty::operator[]: Invalid data layout!");
        return dptr[0];
    }
};

}
