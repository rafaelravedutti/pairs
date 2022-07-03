#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
//---
#ifdef PAIRS_TARGET_CUDA
#   include "devices/cuda.hpp"
#else
#   include "devices/dummy.hpp"
#endif

#pragma once

#define PAIRS_ASSERT(a)
#define PAIRS_ERROR(a)

namespace pairs {

typedef int array_t;
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

class Array {
protected:
    array_t id;
    std::string name;
    void *ptr, *d_ptr;
    size_t size;
    bool is_static;

public:
    Array(array_t id_, std::string name_, void *ptr_, void *d_ptr_, size_t size_, bool is_static_ = false) :
        id(id_),
        name(name_),
        ptr(ptr_),
        d_ptr(d_ptr_),
        size(size_),
        is_static(is_static_) {

        PAIRS_ASSERT(size_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getPointer() { return ptr; }
    void *getDevicePointer() { return d_ptr; }
    void setPointers(void *ptr_, void *d_ptr_) { ptr = ptr_, d_ptr = d_ptr_; }
    void setSize(int size_) { size = size_; }
    size_t getSize() { return size; };
    bool isStatic() { return is_static; }
};

class Property {
protected:
    property_t id;
    std::string name;
    void *ptr, *d_ptr;
    PropertyType type;
    layout_t layout;
    size_t sx, sy;

public:
    /*Property(property_t id_, std::string name_, void *ptr_, void *d_ptr_, PropertyType type_, layout_t layout, size_t sx_) :
        id(id_),
        name(name_),
        ptr(ptr_),
        d_ptr(d_ptr_),
        type(type_),
        layout(Invalid),
        sx(sx_), sy(1) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0);
    }*/

    Property(property_t id_, std::string name_, void *ptr_, void *d_ptr_, PropertyType type_, layout_t layout_, size_t sx_, size_t sy_=1) :
        id(id_),
        name(name_),
        ptr(ptr_),
        d_ptr(d_ptr_),
        type(type_),
        layout(layout_),
        sx(sx_), sy(sy_) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getPointer() { return ptr; }
    void *getDevicePointer() { return d_ptr; }
    void setPointers(void *ptr_, void *d_ptr_) { ptr = ptr_, d_ptr = d_ptr_; }
    void setSizes(int sx_, int sy_) { sx = sx_, sy = sy_; }
    size_t getTotalSize() { return sx * sy * getElemSize(); };
    PropertyType getType() { return type; }
    layout_t getLayout() { return layout; }
    size_t getElemSize() {
        return  (type == Prop_Integer) ? sizeof(int) :
                (type == Prop_Float) ? sizeof(double) :
                (type == Prop_Vector) ? sizeof(double) : 0;
    }
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

class DeviceFlags {
private:
    unsigned long long int *hflags;
    unsigned long long int *dflags;
    int narrays;
    int nflags;
    static const int narrays_per_flag = 64;
public:
    DeviceFlags(int narrays_) : narrays(narrays_) {
        nflags = std::ceil(narrays_ / narrays_per_flag);
        hflags = new unsigned long long int[nflags];
        dflags = new unsigned long long int[nflags];

        for(int i = 0; i < nflags; ++i) {
            hflags[i] = 0xfffffffffffffffful;
            dflags[i] = 0x0;
        }
    }

    inline bool isHostFlagSet(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        return (hflags[flag_index] & (1 << bit)) != 0;
    }

    inline void setHostFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        hflags[flag_index] |= 1 << bit;
    }

    inline void clearHostFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        hflags[flag_index] &= ~(1 << bit);
    }

    inline bool isDeviceFlagSet(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        return (dflags[flag_index] & (1 << bit)) != 0;
    }

    inline void setDeviceFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        dflags[flag_index] |= 1 << bit;
    }

    inline void clearDeviceFlag(int array_id) {
        int flag_index = array_id / narrays_per_flag;
        unsigned long long int bit = array_id % narrays_per_flag;
        dflags[flag_index] &= ~(1 << bit);
    }

    ~DeviceFlags() {
        delete[] hflags;
        delete[] dflags;
    }
};

class PairsSim {
private:
    std::vector<Property> properties;
    std::vector<Array> arrays;
    DeviceFlags *prop_flags, *array_flags;
    int nprops, narrays;
public:
    PairsSim(int nprops_, int narrays_) {
        prop_flags = new DeviceFlags(nprops_);
        array_flags = new DeviceFlags(narrays_);
    }

    ~PairsSim() {
        delete prop_flags;
        delete array_flags;
    }

    void addArray(Array array) { arrays.push_back(array); }
    void updateArray(array_t id, void *ptr, void *d_ptr, int size) {
        auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
        PAIRS_ASSERT(a != std::end(arrays));
        a->setPointers(ptr, d_ptr);
        a->setSize(size);
    }

    Array &getArray(array_t id) {
        auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
        PAIRS_ASSERT(a != std::end(arrays));
        return *a;
    }

    Array &getArrayByName(std::string name) {
        auto a = std::find_if(arrays.begin(), arrays.end(), [name](Array a) { return a.getName() == name; });
        PAIRS_ASSERT(a != std::end(arrays));
        return *a;
    }

    void addProperty(Property prop) { properties.push_back(prop); }
    void updateProperty(property_t id, void *ptr, void *d_ptr, int sx = 0, int sy = 0) {
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
        PAIRS_ASSERT(p != std::end(properties));
        p->setPointers(ptr, d_ptr);
        p->setSizes(sx, sy);
    }

    Property &getProperty(property_t id) {
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
        PAIRS_ASSERT(p != std::end(properties));
        return *p;
    }

    Property &getPropertyByName(std::string name) {
        auto p = std::find_if(properties.begin(), properties.end(), [name](Property p) { return p.getName() == name; });
        PAIRS_ASSERT(p != std::end(properties));
        return *p;
    }

    inline IntProperty &getAsIntegerProperty(Property &prop) { return static_cast<IntProperty&>(prop); }
    inline FloatProperty &getAsFloatProperty(Property &prop) { return static_cast<FloatProperty&>(prop); }
    inline VectorProperty &getAsVectorProperty(Property &prop) { return static_cast<VectorProperty&>(prop); }
    inline IntProperty &getIntegerProperty(property_t property) { return static_cast<IntProperty&>(getProperty(property)); }
    inline FloatProperty &getFloatProperty(property_t property) { return static_cast<FloatProperty&>(getProperty(property)); }
    inline VectorProperty &getVectorProperty(property_t property) { return static_cast<VectorProperty&>(getProperty(property)); }

    void clearArrayDeviceFlag(array_t id) { clearArrayDeviceFlag(getArray(id)); }
    void clearArrayDeviceFlag(Array &array) { array_flags->clearDeviceFlag(array.getId()); }
    void copyArrayToDevice(array_t id) { copyArrayToDevice(getArray(id)); }
    void copyArrayToDevice(Array &array) {
        if(!array_flags->isDeviceFlagSet(array.getId())) {
            if(array.isStatic()) {
                pairs::copy_static_symbol_to_device(array.getPointer(), array.getDevicePointer(), array.getSize());
            } else {
                pairs::copy_to_device(array.getPointer(), array.getDevicePointer(), array.getSize());
            }

            array_flags->setDeviceFlag(array.getId());
        }
    }

    void clearArrayHostFlag(array_t id) { clearArrayHostFlag(getArray(id)); }
    void clearArrayHostFlag(Array &array) { array_flags->clearHostFlag(array.getId()); }
    void copyArrayToHost(array_t id) { copyArrayToHost(getArray(id)); }
    void copyArrayToHost(Array &array) {
        if(!array_flags->isHostFlagSet(array.getId())) {
            if(array.isStatic()) {
                pairs::copy_static_symbol_to_host(array.getDevicePointer(), array.getPointer(), array.getSize());
            } else {
                pairs::copy_to_host(array.getDevicePointer(), array.getPointer(), array.getSize());
            }

            array_flags->setHostFlag(array.getId());
        }
    }

    void clearPropertyDeviceFlag(property_t id) { clearPropertyDeviceFlag(getProperty(id)); }
    void clearPropertyDeviceFlag(Property &prop) { prop_flags->clearDeviceFlag(prop.getId()); }
    void copyPropertyToDevice(property_t id) { copyPropertyToDevice(getProperty(id)); }
    void copyPropertyToDevice(Property &prop) {
        if(!prop_flags->isDeviceFlagSet(prop.getId())) {
            pairs::copy_to_device(prop.getPointer(), prop.getDevicePointer(), prop.getTotalSize());
            prop_flags->setDeviceFlag(prop.getId());
        }
    }

    void clearPropertyHostFlag(property_t id) { clearPropertyHostFlag(getProperty(id)); }
    void clearPropertyHostFlag(Property &prop) { prop_flags->clearHostFlag(prop.getId()); }
    void copyPropertyToHost(property_t id) { copyPropertyToHost(getProperty(id)); }
    void copyPropertyToHost(Property &prop) {
        if(!prop_flags->isHostFlagSet(prop.getId())) {
            pairs::copy_to_host(prop.getDevicePointer(), prop.getPointer(), prop.getTotalSize());
            prop_flags->setHostFlag(prop.getId());
        }
    }

};

}
