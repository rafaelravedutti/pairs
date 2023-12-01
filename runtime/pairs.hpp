#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
//---
#include "allocate.hpp"
#include "array.hpp"
#include "contact_property.hpp"
#include "device_flags.hpp"
#include "feature_property.hpp"
#include "pairs_common.hpp"
#include "property.hpp"
#include "runtime_var.hpp"
#include "timers.hpp"
#include "devices/device.hpp"
#include "domain/regular_6d_stencil.hpp"

#pragma once

#define FLAGS_INFINITE  (1 << 0)
#define FLAGS_GHOST     (1 << 1)
#define FLAGS_FIXED     (1 << 2)
#define FLAGS_GLOBAL    (1 << 3)

namespace pairs {

class PairsSimulation {
private:
    Regular6DStencil *dom_part;
    //DomainPartitioner *dom_part;
    std::vector<Property> properties;
    std::vector<ContactProperty> contact_properties;
    std::vector<FeatureProperty> feature_properties;
    std::vector<Array> arrays;
    DeviceFlags *prop_flags, *contact_prop_flags, *array_flags;
    DomainPartitioning dom_part_type;
    Timers<double> *timers;

public:
    PairsSimulation(int nprops_, int ncontactprops_, int narrays_, DomainPartitioning dom_part_type_) {
        dom_part_type = dom_part_type_;
        prop_flags = new DeviceFlags(nprops_);
        contact_prop_flags = new DeviceFlags(ncontactprops_);
        array_flags = new DeviceFlags(narrays_);
        timers = new Timers<double>(1e-6);
    }

    ~PairsSimulation() {
        dom_part->finalize();
        delete prop_flags;
        delete contact_prop_flags;
        delete array_flags;
        delete timers;
    }

    // Variables
    template<typename T>
    RuntimeVar<T> addDeviceVariable(T *h_ptr) {
       return RuntimeVar<T>(h_ptr); 
    }

    // Arrays
    Array &getArray(array_t id);
    Array &getArrayByName(std::string name);
    Array &getArrayByHostPointer(const void *h_ptr);
    void addArray(Array array);

    template<typename T_ptr>
    void addArray(array_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, size_t size);

    template<typename T_ptr>
    void addArray(array_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, size_t size);

    template<typename T_ptr>
    void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, size_t size);

    template<typename T_ptr>
    void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, size_t size);

    template<typename T_ptr>
    void reallocArray(array_t id, T_ptr **h_ptr, std::nullptr_t, size_t size);

    template<typename T_ptr>
    void reallocArray(array_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t size);

    void copyArrayToDevice(array_t id, action_t action, size_t size = 0) {
        copyArrayToDevice(getArray(id), action, size);
    }

    void copyArrayToDevice(Array &array, action_t action, size_t size = 0);
    void copyArrayToHost(array_t id, action_t action, size_t size = 0) {
        copyArrayToHost(getArray(id), action, size);
    }

    void copyArrayToHost(Array &array, action_t action, size_t size = 0);

    // Properties
    Property &getProperty(property_t id);
    Property &getPropertyByName(std::string name);
    void addProperty(Property prop);

    template<typename T_ptr>
    void addProperty(
        property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t,
        PropertyType type, layout_t layout, size_t sx, size_t sy = 1);

    template<typename T_ptr> void addProperty(
        property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr,
        PropertyType type, layout_t layout, size_t sx, size_t sy = 1);

    template<typename T_ptr>
    void reallocProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx = 1, size_t sy = 1);

    template<typename T_ptr>
    void reallocProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx = 1, size_t sy = 1);

    inline IntProperty &getAsIntegerProperty(Property &prop) {
        return static_cast<IntProperty&>(prop);
    }

    inline FloatProperty &getAsFloatProperty(Property &prop) {
        return static_cast<FloatProperty&>(prop);
    }

    inline VectorProperty &getAsVectorProperty(Property &prop) {
        return static_cast<VectorProperty&>(prop);
    }

    inline MatrixProperty &getAsMatrixProperty(Property &prop) {
        return static_cast<MatrixProperty&>(prop);
    }

    inline QuaternionProperty &getAsQuaternionProperty(Property &prop) {
        return static_cast<QuaternionProperty&>(prop);
    }

    inline IntProperty &getIntegerProperty(property_t property) {
        return static_cast<IntProperty&>(getProperty(property));
    }

    inline FloatProperty &getFloatProperty(property_t property) {
        return static_cast<FloatProperty&>(getProperty(property));
    }

    inline VectorProperty &getVectorProperty(property_t property) {
        return static_cast<VectorProperty&>(getProperty(property));
    }

    inline MatrixProperty &getMatrixProperty(property_t property) {
        return static_cast<MatrixProperty&>(getProperty(property));
    }

    inline QuaternionProperty &getQuaternionProperty(property_t property) {
        return static_cast<QuaternionProperty&>(getProperty(property));
    }

    void copyPropertyToDevice(property_t id, action_t action, size_t size = 0) {
        copyPropertyToDevice(getProperty(id), action, size);
    }

    void copyPropertyToDevice(Property &prop, action_t action, size_t size = 0);

    void copyPropertyToHost(property_t id, action_t action, size_t size = 0) {
        copyPropertyToHost(getProperty(id), action, size);
    }

    void copyPropertyToHost(Property &prop, action_t action, size_t size = 0);

    // Contact properties
    ContactProperty &getContactProperty(property_t id);
    ContactProperty &getContactPropertyByName(std::string name);
    void addContactProperty(ContactProperty prop);

    template<typename T_ptr>
    void addContactProperty(
        property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t,
        PropertyType type, layout_t layout, size_t sx, size_t sy = 1);

    template<typename T_ptr>
    void addContactProperty(
        property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr,
        PropertyType type, layout_t layout, size_t sx, size_t sy = 1);

    template<typename T_ptr>
    void reallocContactProperty(
        property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx = 1, size_t sy = 1);

    template<typename T_ptr>
    void reallocContactProperty(
        property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx = 1, size_t sy = 1);

    void copyContactPropertyToDevice(property_t id, action_t action, size_t size = 0) {
        copyContactPropertyToDevice(getContactProperty(id), action, size);
    }

    void copyContactPropertyToDevice(ContactProperty &prop, action_t action, size_t size = 0);

    void copyContactPropertyToHost(property_t id, action_t action, size_t size = 0) {
        copyContactPropertyToHost(getContactProperty(id), action, size);
    }

    void copyContactPropertyToHost(ContactProperty &prop, action_t action, size_t size = 0);

    // Feature properties
    FeatureProperty &getFeatureProperty(property_t id);
    FeatureProperty &getFeaturePropertyByName(std::string name);
    void addFeatureProperty(FeatureProperty feature_prop);

    template<typename T_ptr>
    void addFeatureProperty(
        property_t id, std::string name, T_ptr *h_ptr, std::nullptr_t,
        PropertyType type, int nkinds, int array_size);

    template<typename T_ptr>
    void addFeatureProperty(
        property_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr,
        PropertyType type, int nkinds, int array_size);

    void copyFeaturePropertyToDevice(property_t id) {
        copyFeaturePropertyToDevice(getFeatureProperty(id));
    }

    void copyFeaturePropertyToDevice(FeatureProperty &feature_prop);

    // Communication
    void initDomain(
        int *argc, char ***argv,
        real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax);

    Regular6DStencil *getDomainPartitioner() { return dom_part; }
    void communicateSizes(int dim, const int *send_sizes, int *recv_sizes);
    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv);

    void fillCommunicationArrays(int neighbor_ranks[], int pbc[], real_t subdom[]);

    // Device functions
    void sync() { device_synchronize(); }

    // Timers
    Timers<double> *getTimers() { return timers; }
    void printTimers() {
        if(this->getDomainPartitioner()->getRank() == 0) {
            this->getTimers()->print();
        }
    }
};

template<typename T_ptr>
void PairsSimulation::addArray(array_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, size_t size) {
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addArray(Array(id, name, *h_ptr, nullptr, size, false));
}

template<typename T_ptr>
void PairsSimulation::addArray(array_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, size_t size) {
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    *d_ptr = (T_ptr *) pairs::device_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr && *d_ptr != nullptr);
    addArray(Array(id, name, *h_ptr, *d_ptr, size, false));
}

template<typename T_ptr>
void PairsSimulation::addStaticArray(array_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, size_t size) {
    addArray(Array(id, name, h_ptr, nullptr, size, true));
}

template<typename T_ptr>
void PairsSimulation::addStaticArray(array_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, size_t size) {
    addArray(Array(id, name, h_ptr, d_ptr, size, true));
}

template<typename T_ptr>
void PairsSimulation::reallocArray(array_t id, T_ptr **h_ptr, std::nullptr_t, size_t size) {
    // This should be a pointer (and not a reference) in order to be modified
    auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
    PAIRS_ASSERT(a != std::end(arrays));
    PAIRS_ASSERT(size > 0);

    size_t old_size = a->getSize();
    *h_ptr = (T_ptr *) pairs::host_realloc(*h_ptr, size, old_size);
    PAIRS_ASSERT(*h_ptr != nullptr);

    a->setPointers(*h_ptr, nullptr);
    a->setSize(size);
}

template<typename T_ptr>
void PairsSimulation::reallocArray(array_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t size) {
    // This should be a pointer (and not a reference) in order to be modified
    auto a = std::find_if(arrays.begin(), arrays.end(), [id](Array a) { return a.getId() == id; });
    PAIRS_ASSERT(a != std::end(arrays));
    PAIRS_ASSERT(size > 0);

    size_t old_size = a->getSize();
    void *new_h_ptr = pairs::host_realloc(*h_ptr, size, old_size);
    void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
    PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

    a->setPointers(new_h_ptr, new_d_ptr);
    a->setSize(size);

    *h_ptr = (T_ptr *) new_h_ptr;
    *d_ptr = (T_ptr *) new_d_ptr;
    if(array_flags->isDeviceFlagSet(id)) {
        copyArrayToDevice(id, false);
    }
}

template<typename T_ptr>
void PairsSimulation::addProperty(
    property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addProperty(Property(id, name, *h_ptr, nullptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::addProperty(
    property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    *d_ptr = (T_ptr *) pairs::device_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr && *d_ptr != nullptr);
    addProperty(Property(id, name, *h_ptr, *d_ptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::reallocProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx, size_t sy) {
    // This should be a pointer (and not a reference) in order to be modified
    auto p = std::find_if(properties.begin(),
		    	  properties.end(),
			  [id](Property _p) { return _p.getId() == id; });
    PAIRS_ASSERT(p != std::end(properties));

    size_t size = sx * sy * p->getPrimitiveTypeSize();
    PAIRS_ASSERT(size > 0);

    size_t old_size = p->getTotalSize();
    *h_ptr = (T_ptr *) pairs::host_realloc(*h_ptr, size, old_size);
    PAIRS_ASSERT(*h_ptr != nullptr);

    p->setPointers(*h_ptr, nullptr);
    p->setSizes(sx, sy);
}

template<typename T_ptr>
void PairsSimulation::reallocProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx, size_t sy) {
    // This should be a pointer (and not a reference) in order to be modified
    auto p = std::find_if(properties.begin(),
		    	  properties.end(),
			  [id](Property _p) { return _p.getId() == id; });
    PAIRS_ASSERT(p != std::end(properties));

    size_t size = sx * sy * p->getPrimitiveTypeSize();
    PAIRS_ASSERT(size > 0);

    size_t old_size = p->getTotalSize();
    void *new_h_ptr = pairs::host_realloc(*h_ptr, size, old_size);
    void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
    PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

    p->setPointers(new_h_ptr, new_d_ptr);
    p->setSizes(sx, sy);

    *h_ptr = (T_ptr *) new_h_ptr;
    *d_ptr = (T_ptr *) new_d_ptr;
    if(prop_flags->isDeviceFlagSet(id)) {
        copyPropertyToDevice(id, false);
    }
}

template<typename T_ptr>
void PairsSimulation::addContactProperty(
    property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addContactProperty(ContactProperty(id, name, *h_ptr, nullptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::addContactProperty(
    property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) pairs::host_alloc(size);
    *d_ptr = (T_ptr *) pairs::device_alloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr && *d_ptr != nullptr);
    addContactProperty(ContactProperty(id, name, *h_ptr, *d_ptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::reallocContactProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx, size_t sy) {
    // This should be a pointer (and not a reference) in order to be modified
    auto cp = std::find_if(contact_properties.begin(),
		    	   contact_properties.end(),
			   [id](ContactProperty _cp) { return _cp.getId() == id; });
    PAIRS_ASSERT(cp != std::end(contact_properties));

    size_t size = sx * sy * cp->getPrimitiveTypeSize();
    PAIRS_ASSERT(size > 0);

    size_t old_size = cp->getTotalSize();
    *h_ptr = (T_ptr *) pairs::host_realloc(*h_ptr, size, old_size);
    PAIRS_ASSERT(*h_ptr != nullptr);

    cp->setPointers(*h_ptr, nullptr);
    cp->setSizes(sx, sy);
}

template<typename T_ptr>
void PairsSimulation::reallocContactProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx, size_t sy) {
    // This should be a pointer (and not a reference) in order to be modified
    auto cp = std::find_if(contact_properties.begin(),
		    	   contact_properties.end(),
			   [id](ContactProperty _cp) { return _cp.getId() == id; });
    PAIRS_ASSERT(cp != std::end(contact_properties));

    size_t size = sx * sy * cp->getPrimitiveTypeSize();
    PAIRS_ASSERT(size > 0);

    size_t old_size = cp->getTotalSize();
    void *new_h_ptr = pairs::host_realloc(*h_ptr, size, old_size);
    void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
    PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

    cp->setPointers(new_h_ptr, new_d_ptr);
    cp->setSizes(sx, sy);

    *h_ptr = (T_ptr *) new_h_ptr;
    *d_ptr = (T_ptr *) new_d_ptr;
    if(contact_prop_flags->isDeviceFlagSet(id)) {
        copyContactPropertyToDevice(id, false);
    }
}

template<typename T_ptr>
void PairsSimulation::addFeatureProperty(property_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, PropertyType type, int nkinds, int array_size) {
    PAIRS_ASSERT(nkinds > 0 && array_size > 0);
    PAIRS_ASSERT(h_ptr != nullptr);
    addFeatureProperty(FeatureProperty(id, name, h_ptr, nullptr, type, nkinds, array_size));
}

template<typename T_ptr>
void PairsSimulation::addFeatureProperty(property_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, PropertyType type, int nkinds, int array_size) {
    PAIRS_ASSERT(nkinds > 0 && array_size > 0);
    PAIRS_ASSERT(h_ptr != nullptr && d_ptr != nullptr);
    addFeatureProperty(FeatureProperty(id, name, h_ptr, d_ptr, type, nkinds, array_size));
}

}
