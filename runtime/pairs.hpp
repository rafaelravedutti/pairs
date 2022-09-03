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

#include "domain/domain_partitioning.hpp"

#pragma once

#ifdef DEBUG
#   include <assert.h>
#   define PAIRS_DEBUG(...)     fprintf(stderr, __VA_ARGS__)
#   define PAIRS_ASSERT(a)      assert(a)
#   define PAIRS_EXCEPTION(a)
#else
#   define PAIRS_DEBUG(...)
#   define PAIRS_ASSERT(a)
#   define PAIRS_EXCEPTION(a)
#endif

#define PAIRS_ERROR(...)    fprintf(stderr, __VA_ARGS__)

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

enum DomainPartitioning {
    DimRanges = 0,
    BoxList,
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

class Property {
protected:
    property_t id;
    std::string name;
    void *h_ptr, *d_ptr;
    PropertyType type;
    layout_t layout;
    size_t sx, sy;

public:
    /*Property(property_t id_, std::string name_, void *h_ptr_, void *d_ptr_, PropertyType type_, layout_t layout, size_t sx_) :
        id(id_),
        name(name_),
        h_ptr(h_ptr_),
        d_ptr(d_ptr_),
        type(type_),
        layout(Invalid),
        sx(sx_), sy(1) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0);
    }*/

    Property(property_t id_, std::string name_, void *h_ptr_, void *d_ptr_, PropertyType type_, layout_t layout_, size_t sx_, size_t sy_=1) :
        id(id_),
        name(name_),
        h_ptr(h_ptr_),
        d_ptr(d_ptr_),
        type(type_),
        layout(layout_),
        sx(sx_), sy(sy_) {

        PAIRS_ASSERT(type != Prop_Invalid && layout_ != Invalid && sx_ > 0 && sy_ > 0);
    }

    property_t getId() { return id; }
    std::string getName() { return name; }
    void *getHostPointer() { return h_ptr; }
    void *getDevicePointer() { return d_ptr; }
    void setPointers(void *h_ptr_, void *d_ptr_) { h_ptr = h_ptr_, d_ptr = d_ptr_; }
    void setSizes(size_t sx_, size_t sy_) { sx = sx_, sy = sy_; }
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
    inline int &operator()(int i) { return static_cast<int *>(h_ptr)[i]; }
};

class FloatProperty : public Property {
public:
    inline double &operator()(int i) { return static_cast<double *>(h_ptr)[i]; }
};

class VectorProperty : public Property {
public:
    double &operator()(int i, int j) {
        PAIRS_ASSERT(type != Prop_Invalid && layout != Invalid && sx > 0 && sy > 0);
        double *dptr = static_cast<double *>(h_ptr);
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
        nflags = std::ceil((double) narrays_ / (double) narrays_per_flag);
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

    void printFlags() {
        fprintf(stderr, "hflags = ");
        for(int i = 0; i < nflags; i++) {
            for(int b = 63; b >= 0; b--) {
                if(hflags[i] & (1 << b)) {
                    fprintf(stderr, "1");
                } else {
                    fprintf(stderr, "0");
                }
            }
        }

        fprintf(stderr, "\n");
        fprintf(stderr, "dflags = ");
        for(int i = 0; i < nflags; i++) {
            for(int b = 63; b >= 0; b--) {
                if(dflags[i] & (1 << b)) {
                    fprintf(stderr, "1");
                } else {
                    fprintf(stderr, "0");
                }
            }
        }

        fprintf(stderr, "\n");
    }

    ~DeviceFlags() {
        delete[] hflags;
        delete[] dflags;
    }
};

template<int ndims>
class PairsSimulation {
private:
    DomainPartitioner<ndims> *dom_part;
    std::vector<Property> properties;
    std::vector<Array> arrays;
    DeviceFlags *prop_flags, *array_flags;
    DomainPartitioning dom_part_type;
    int nprops, narrays;
public:
    PairsSimulation(int nprops_, int narrays_, DomainPartitioning dom_part_type_) {
        dom_part_type = dom_part_type_;
        prop_flags = new DeviceFlags(nprops_);
        array_flags = new DeviceFlags(narrays_);
    }

    ~PairsSimulation() {
        dom_part->finalize();
        delete prop_flags;
        delete array_flags;
    }

    void initDomain(real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {
        if(dom_part_type == DimRanges) {
            dom_part = new DimensionRanges<ndims>(xmin, xmax, ymin, ymax, zmin, zmax);
        } else {
            PAIRS_EXCEPTION("Domain partitioning type not implemented!\n");
        }

        dom_part->initialize();
    }

    DomainPartitioner<ndims> *getDomainPartitioner() { return dom_part; }

    template<typename T_ptr>
    void addArray(array_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, size_t size) {
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) malloc(size);
        PAIRS_ASSERT(*h_ptr != nullptr);
        addArray(Array(id, name, *h_ptr, nullptr, size, false));
    }

    template<typename T_ptr>
    void addArray(array_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, size_t size) {
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) malloc(size);
        *d_ptr = (T_ptr *) pairs::device_alloc(size);
        PAIRS_ASSERT(*h_ptr != nullptr && *d_ptr != nullptr);
        addArray(Array(id, name, *h_ptr, *d_ptr, size, false));
    }

    template<typename T_ptr>
    void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, size_t size) {
        addArray(Array(id, name, h_ptr, nullptr, size, true));
    }

    template<typename T_ptr>
    void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, size_t size) {
        addArray(Array(id, name, h_ptr, d_ptr, size, true));
    }

    void addArray(Array array) {
        int id = array.getId();
        auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
        PAIRS_ASSERT(a == std::end(arrays));
        arrays.push_back(array);
    }

    template<typename T_ptr>
    void reallocArray(array_t id, T_ptr **h_ptr, std::nullptr_t, size_t size) {
        // This should be a pointer (and not a reference) in order to be modified
        auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
        PAIRS_ASSERT(a != std::end(arrays));
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) realloc(*h_ptr, size);
        PAIRS_ASSERT(*h_ptr != nullptr);

        a->setPointers(*h_ptr, nullptr);
        a->setSize(size);
    }

    template<typename T_ptr>
    void reallocArray(array_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t size) {
        // This should be a pointer (and not a reference) in order to be modified
        auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
        PAIRS_ASSERT(a != std::end(arrays));
        PAIRS_ASSERT(size > 0);

        void *new_h_ptr = realloc(*h_ptr, size);
        void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
        PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

        a->setPointers(new_h_ptr, new_d_ptr);
        a->setSize(size);

        *h_ptr = (T_ptr *) new_h_ptr;
        *d_ptr = (T_ptr *) new_d_ptr;
        if(array_flags->isDeviceFlagSet(id)) {
            copyArrayToDevice(id);
        }
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

    template<typename T_ptr>
    void addProperty(property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy=1) {
        size_t size = sx * sy * sizeof(T_ptr);
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) malloc(size);
        PAIRS_ASSERT(*h_ptr != nullptr);
        addProperty(Property(id, name, *h_ptr, nullptr, type, layout, sx, sy));
    }

    template<typename T_ptr>
    void addProperty(property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy=1) {
        size_t size = sx * sy * sizeof(T_ptr);
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) malloc(size);
        *d_ptr = (T_ptr *) pairs::device_alloc(size);
        PAIRS_ASSERT(*h_ptr != nullptr && *d_ptr != nullptr);
        addProperty(Property(id, name, *h_ptr, *d_ptr, type, layout, sx, sy));
    }

    void addProperty(Property prop) {
        int id = prop.getId();
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
        PAIRS_ASSERT(p == std::end(properties));
        properties.push_back(prop);
    }

    template<typename T_ptr>
    void reallocProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx = 1, size_t sy = 1) {
        // This should be a pointer (and not a reference) in order to be modified
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
        PAIRS_ASSERT(p != std::end(properties));

        size_t size = sx * sy * p->getElemSize();
        PAIRS_ASSERT(size > 0);

        *h_ptr = (T_ptr *) realloc(*h_ptr, size);
        PAIRS_ASSERT(*h_ptr != nullptr);

        p->setPointers(*h_ptr, nullptr);
        p->setSizes(sx, sy);
    }

    template<typename T_ptr>
    void reallocProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx = 1, size_t sy = 1) {
        // This should be a pointer (and not a reference) in order to be modified
        auto p = std::find_if(properties.begin(), properties.end(), [id](Property p) { return p.getId() == id; });
        PAIRS_ASSERT(p != std::end(properties));

        size_t size = sx * sy * p->getElemSize();
        PAIRS_ASSERT(size > 0);

        void *new_h_ptr = realloc(*h_ptr, size);
        void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
        PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

        p->setPointers(new_h_ptr, new_d_ptr);
        p->setSizes(sx, sy);

        *h_ptr = (T_ptr *) new_h_ptr;
        *d_ptr = (T_ptr *) new_d_ptr;
        if(prop_flags->isDeviceFlagSet(id)) {
            copyPropertyToDevice(id);
        }
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

    void setArrayDeviceFlag(array_t id) { setArrayDeviceFlag(getArray(id)); }
    void setArrayDeviceFlag(Array &array) { array_flags->setDeviceFlag(array.getId()); }
    void clearArrayDeviceFlag(array_t id) { clearArrayDeviceFlag(getArray(id)); }
    void clearArrayDeviceFlag(Array &array) { array_flags->clearDeviceFlag(array.getId()); }
    void copyArrayToDevice(array_t id) { copyArrayToDevice(getArray(id)); }
    void copyArrayToDevice(Array &array) {
        int array_id = array.getId();
        if(!array_flags->isDeviceFlagSet(array_id)) {
            if(array.isStatic()) {
                PAIRS_DEBUG("Copying static array %s to device\n", array.getName().c_str());
                pairs::copy_static_symbol_to_device(array.getHostPointer(), array.getDevicePointer(), array.getSize());
            } else {
                PAIRS_DEBUG("Copying array %s to device\n", array.getName().c_str());
                pairs::copy_to_device(array.getHostPointer(), array.getDevicePointer(), array.getSize());
            }
        }
    }

    void setArrayHostFlag(array_t id) { setArrayHostFlag(getArray(id)); }
    void setArrayHostFlag(Array &array) { array_flags->setHostFlag(array.getId()); }
    void clearArrayHostFlag(array_t id) { clearArrayHostFlag(getArray(id)); }
    void clearArrayHostFlag(Array &array) { array_flags->clearHostFlag(array.getId()); }
    void copyArrayToHost(array_t id) { copyArrayToHost(getArray(id)); }
    void copyArrayToHost(Array &array) {
        int array_id = array.getId();
        if(!array_flags->isHostFlagSet(array_id)) {
            if(array.isStatic()) {
                PAIRS_DEBUG("Copying static array %s to host\n", array.getName().c_str());
                pairs::copy_static_symbol_to_host(array.getDevicePointer(), array.getHostPointer(), array.getSize());
            } else {
                PAIRS_DEBUG("Copying array %s to host\n", array.getName().c_str());
                pairs::copy_to_host(array.getDevicePointer(), array.getHostPointer(), array.getSize());
            }
        }
    }

    void setPropertyDeviceFlag(property_t id) { setPropertyDeviceFlag(getProperty(id)); }
    void setPropertyDeviceFlag(Property &prop) { prop_flags->setDeviceFlag(prop.getId()); }
    void clearPropertyDeviceFlag(property_t id) { clearPropertyDeviceFlag(getProperty(id)); }
    void clearPropertyDeviceFlag(Property &prop) { prop_flags->clearDeviceFlag(prop.getId()); }
    void copyPropertyToDevice(property_t id) { copyPropertyToDevice(getProperty(id)); }
    void copyPropertyToDevice(Property &prop) {
        if(!prop_flags->isDeviceFlagSet(prop.getId())) {
            PAIRS_DEBUG("Copying property %s to device\n", prop.getName().c_str());
            pairs::copy_to_device(prop.getHostPointer(), prop.getDevicePointer(), prop.getTotalSize());
        }
    }

    void setPropertyHostFlag(property_t id) { setPropertyHostFlag(getProperty(id)); }
    void setPropertyHostFlag(Property &prop) { prop_flags->setHostFlag(prop.getId()); }
    void clearPropertyHostFlag(property_t id) { clearPropertyHostFlag(getProperty(id)); }
    void clearPropertyHostFlag(Property &prop) { prop_flags->clearHostFlag(prop.getId()); }
    void copyPropertyToHost(property_t id) { copyPropertyToHost(getProperty(id)); }
    void copyPropertyToHost(Property &prop) {
        if(!prop_flags->isHostFlagSet(prop.getId())) {
            PAIRS_DEBUG("Copying property %s to host\n", prop.getName().c_str());
            pairs::copy_to_host(prop.getDevicePointer(), prop.getHostPointer(), prop.getTotalSize());
        }
    }
};

}
