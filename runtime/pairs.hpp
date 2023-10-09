#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
//---
#include "array.hpp"
#include "contact_property.hpp"
#include "device_flags.hpp"
#include "feature_property.hpp"
#include "pairs_common.hpp"
#include "property.hpp"
#include "runtime_var.hpp"
#include "vector3.hpp"
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
public:
    PairsSimulation(int nprops_, int ncontactprops_, int narrays_, DomainPartitioning dom_part_type_) {
        dom_part_type = dom_part_type_;
        prop_flags = new DeviceFlags(nprops_);
        contact_prop_flags = new DeviceFlags(ncontactprops_);
        array_flags = new DeviceFlags(narrays_);
    }

    ~PairsSimulation() {
        dom_part->finalize();
        delete prop_flags;
        delete contact_prop_flags;
        delete array_flags;
    }

    void initDomain(int *argc, char ***argv, real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax);
    Regular6DStencil *getDomainPartitioner() { return dom_part; }

    template<typename T>
    RuntimeVar<T> addDeviceVariable(T *h_ptr) {
       return RuntimeVar<T>(h_ptr); 
    }

    template<typename T_ptr> void addArray(array_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, size_t size);
    template<typename T_ptr> void addArray(array_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, size_t size);
    template<typename T_ptr> void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, size_t size);
    template<typename T_ptr> void addStaticArray(array_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, size_t size);
    void addArray(Array array);

    template<typename T_ptr> void reallocArray(array_t id, T_ptr **h_ptr, std::nullptr_t, size_t size);
    template<typename T_ptr> void reallocArray(array_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t size);

    Array &getArray(array_t id);
    Array &getArrayByName(std::string name);
    Array &getArrayByHostPointer(const void *h_ptr);

    template<typename T_ptr> void addProperty(
        property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy = 1);
    template<typename T_ptr> void addProperty(
        property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy = 1);
    void addProperty(Property prop);

    template<typename T_ptr> void reallocProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx = 1, size_t sy = 1);
    template<typename T_ptr> void reallocProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx = 1, size_t sy = 1);

    Property &getProperty(property_t id);
    Property &getPropertyByName(std::string name);

    inline IntProperty &getAsIntegerProperty(Property &prop) { return static_cast<IntProperty&>(prop); }
    inline FloatProperty &getAsFloatProperty(Property &prop) { return static_cast<FloatProperty&>(prop); }
    inline VectorProperty &getAsVectorProperty(Property &prop) { return static_cast<VectorProperty&>(prop); }
    inline IntProperty &getIntegerProperty(property_t property) { return static_cast<IntProperty&>(getProperty(property)); }
    inline FloatProperty &getFloatProperty(property_t property) { return static_cast<FloatProperty&>(getProperty(property)); }
    inline VectorProperty &getVectorProperty(property_t property) { return static_cast<VectorProperty&>(getProperty(property)); }

    template<typename T_ptr> void addContactProperty(
        property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy = 1);
    template<typename T_ptr> void addContactProperty(
        property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy = 1);
    void addContactProperty(ContactProperty prop);

    template<typename T_ptr> void reallocContactProperty(property_t id, T_ptr **h_ptr, std::nullptr_t, size_t sx = 1, size_t sy = 1);
    template<typename T_ptr> void reallocContactProperty(property_t id, T_ptr **h_ptr, T_ptr **d_ptr, size_t sx = 1, size_t sy = 1);

    ContactProperty &getContactProperty(property_t id);
    ContactProperty &getContactPropertyByName(std::string name);

    template<typename T_ptr> void addFeatureProperty(property_t id, std::string name, T_ptr *h_ptr, std::nullptr_t, PropertyType type, int nkinds, int array_size);
    template<typename T_ptr> void addFeatureProperty(property_t id, std::string name, T_ptr *h_ptr, T_ptr *d_ptr, PropertyType type, int nkinds, int array_size);
    void addFeatureProperty(FeatureProperty feature_prop);

    FeatureProperty &getFeatureProperty(property_t id);
    FeatureProperty &getFeaturePropertyByName(std::string name);

    void setArrayDeviceFlag(array_t id) { setArrayDeviceFlag(getArray(id)); }
    void setArrayDeviceFlag(Array &array) { array_flags->setDeviceFlag(array.getId()); }
    void clearArrayDeviceFlag(array_t id) { clearArrayDeviceFlag(getArray(id)); }
    void clearArrayDeviceFlag(Array &array) { array_flags->clearDeviceFlag(array.getId()); }
    void copyArrayToDevice(array_t id) { copyArrayToDevice(getArray(id)); }
    void copyArrayToDevice(Array &array);

    void setArrayHostFlag(array_t id) { setArrayHostFlag(getArray(id)); }
    void setArrayHostFlag(Array &array) { array_flags->setHostFlag(array.getId()); }
    void clearArrayHostFlag(array_t id) { clearArrayHostFlag(getArray(id)); }
    void clearArrayHostFlag(Array &array) { array_flags->clearHostFlag(array.getId()); }
    void copyArrayToHost(array_t id) { copyArrayToHost(getArray(id)); }
    void copyArrayToHost(Array &array);

    void setPropertyDeviceFlag(property_t id) { setPropertyDeviceFlag(getProperty(id)); }
    void setPropertyDeviceFlag(Property &prop) { prop_flags->setDeviceFlag(prop.getId()); }
    void clearPropertyDeviceFlag(property_t id) { clearPropertyDeviceFlag(getProperty(id)); }
    void clearPropertyDeviceFlag(Property &prop) { prop_flags->clearDeviceFlag(prop.getId()); }
    void copyPropertyToDevice(property_t id) { copyPropertyToDevice(getProperty(id)); }
    void copyPropertyToDevice(Property &prop);

    void setPropertyHostFlag(property_t id) { setPropertyHostFlag(getProperty(id)); }
    void setPropertyHostFlag(Property &prop) { prop_flags->setHostFlag(prop.getId()); }
    void clearPropertyHostFlag(property_t id) { clearPropertyHostFlag(getProperty(id)); }
    void clearPropertyHostFlag(Property &prop) { prop_flags->clearHostFlag(prop.getId()); }
    void copyPropertyToHost(property_t id) { copyPropertyToHost(getProperty(id)); }
    void copyPropertyToHost(Property &prop);

    void setContactPropertyDeviceFlag(property_t id) { setContactPropertyDeviceFlag(getContactProperty(id)); }
    void setContactPropertyDeviceFlag(ContactProperty &prop) { contact_prop_flags->setDeviceFlag(prop.getId()); }
    void clearContactPropertyDeviceFlag(property_t id) { clearContactPropertyDeviceFlag(getContactProperty(id)); }
    void clearContactPropertyDeviceFlag(ContactProperty &prop) { contact_prop_flags->clearDeviceFlag(prop.getId()); }
    void copyContactPropertyToDevice(property_t id) { copyContactPropertyToDevice(getContactProperty(id)); }
    void copyContactPropertyToDevice(ContactProperty &prop);

    void setContactPropertyHostFlag(property_t id) { setContactPropertyHostFlag(getContactProperty(id)); }
    void setContactPropertyHostFlag(ContactProperty &prop) { contact_prop_flags->setHostFlag(prop.getId()); }
    void clearContactPropertyHostFlag(property_t id) { clearContactPropertyHostFlag(getContactProperty(id)); }
    void clearContactPropertyHostFlag(ContactProperty &prop) { contact_prop_flags->clearHostFlag(prop.getId()); }
    void copyContactPropertyToHost(property_t id) { copyContactPropertyToHost(getContactProperty(id)); }
    void copyContactPropertyToHost(ContactProperty &prop);

    void copyFeaturePropertyToDevice(property_t id) { copyFeaturePropertyToDevice(getFeatureProperty(id)); }
    void copyFeaturePropertyToDevice(FeatureProperty &feature_prop);

    void communicateSizes(int dim, const int *send_sizes, int *recv_sizes);
    void communicateData(
        int dim, int elem_size,
        const real_t *send_buf, const int *send_offsets, const int *nsend,
        real_t *recv_buf, const int *recv_offsets, const int *nrecv);

    void fillCommunicationArrays(int neighbor_ranks[], int pbc[], real_t subdom[]);
    void sync() { device_synchronize(); }
};

template<typename T_ptr>
void PairsSimulation::addArray(array_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, size_t size) {
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addArray(Array(id, name, *h_ptr, nullptr, size, false));
}

template<typename T_ptr>
void PairsSimulation::addArray(array_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, size_t size) {
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
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

    *h_ptr = (T_ptr *) realloc(*h_ptr, size);
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

template<typename T_ptr>
void PairsSimulation::addProperty(
    property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addProperty(Property(id, name, *h_ptr, nullptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::addProperty(
    property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
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

    size_t size = sx * sy * p->getElemSize();
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) realloc(*h_ptr, size);
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

template<typename T_ptr>
void PairsSimulation::addContactProperty(
    property_t id, std::string name, T_ptr **h_ptr, std::nullptr_t, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
    PAIRS_ASSERT(*h_ptr != nullptr);
    addContactProperty(ContactProperty(id, name, *h_ptr, nullptr, type, layout, sx, sy));
}

template<typename T_ptr>
void PairsSimulation::addContactProperty(
    property_t id, std::string name, T_ptr **h_ptr, T_ptr **d_ptr, PropertyType type, layout_t layout, size_t sx, size_t sy) {

    size_t size = sx * sy * sizeof(T_ptr);
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) malloc(size);
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

    size_t size = sx * sy * cp->getElemSize();
    PAIRS_ASSERT(size > 0);

    *h_ptr = (T_ptr *) realloc(*h_ptr, size);
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

    size_t size = sx * sy * cp->getElemSize();
    PAIRS_ASSERT(size > 0);

    void *new_h_ptr = realloc(*h_ptr, size);
    void *new_d_ptr = pairs::device_realloc(*d_ptr, size);
    PAIRS_ASSERT(new_h_ptr != nullptr && new_d_ptr != nullptr);

    cp->setPointers(new_h_ptr, new_d_ptr);
    cp->setSizes(sx, sy);

    *h_ptr = (T_ptr *) new_h_ptr;
    *d_ptr = (T_ptr *) new_d_ptr;
    if(contact_prop_flags->isDeviceFlagSet(id)) {
        copyContactPropertyToDevice(id);
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
