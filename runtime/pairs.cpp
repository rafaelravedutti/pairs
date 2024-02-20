#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
//---
#include "pairs.hpp"
#include "pairs_common.hpp"
#include "devices/device.hpp"
#include "domain/regular_6d_stencil.hpp"

namespace pairs {

void PairsSimulation::initDomain(
    int *argc, char ***argv,
    real_t xmin, real_t xmax, real_t ymin, real_t ymax, real_t zmin, real_t zmax) {

    if(dom_part_type == Regular) {
        const int flags[] = {1, 1, 1};
        dom_part = new Regular6DStencil(xmin, xmax, ymin, ymax, zmin, zmax, flags);
    } else if(dom_part_type == RegularXY) {
        const int flags[] = {1, 1, 0};
        dom_part = new Regular6DStencil(xmin, xmax, ymin, ymax, zmin, zmax, flags);
    } else if(dom_part_type == BlockForest) {
        dom_part = new BlockForest(xmin, xmax, ymin, ymax, zmin, zmax);
    } else {
        PAIRS_EXCEPTION("Domain partitioning type not implemented!\n");
    }

    dom_part->initialize(argc, argv);
}

void PairsSimulation::addArray(Array array) {
    int id = array.getId();
    auto a = std::find_if(
        arrays.begin(),
        arrays.end(),
        [id](Array _a) { return _a.getId() == id; });

    PAIRS_ASSERT(a == std::end(arrays));
    arrays.push_back(array);
}

Array &PairsSimulation::getArray(array_t id) {
    auto a = std::find_if(
        arrays.begin(),
        arrays.end(),
        [id](Array _a) { return _a.getId() == id; });

    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

Array &PairsSimulation::getArrayByName(std::string name) {
    auto a = std::find_if(
        arrays.begin(),
        arrays.end(),
        [name](Array _a) { return _a.getName() == name; });

    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

Array &PairsSimulation::getArrayByHostPointer(const void *h_ptr) {
    auto a = std::find_if(
        arrays.begin(),
        arrays.end(),
        [h_ptr](Array _a) { return _a.getHostPointer() == h_ptr; });

    PAIRS_ASSERT(a != std::end(arrays));
    return *a;
}

void PairsSimulation::addProperty(Property prop) {
    int id = prop.getId();
    auto p = std::find_if(
        properties.begin(),
        properties.end(),
        [id](Property _p) { return _p.getId() == id; });

    PAIRS_ASSERT(p == std::end(properties));
    properties.push_back(prop);
}

Property &PairsSimulation::getProperty(property_t id) {
    auto p = std::find_if(
        properties.begin(),
        properties.end(),
        [id](Property _p) { return _p.getId() == id; });

    PAIRS_ASSERT(p != std::end(properties));
    return *p;
}

Property &PairsSimulation::getPropertyByName(std::string name) {
    auto p = std::find_if(
        properties.begin(),
        properties.end(),
        [name](Property _p) { return _p.getName() == name; });

    PAIRS_ASSERT(p != std::end(properties));
    return *p;
}

void PairsSimulation::addContactProperty(ContactProperty contact_prop) {
    int id = contact_prop.getId();
    auto cp = std::find_if(
        contact_properties.begin(),
        contact_properties.end(),
        [id](ContactProperty _cp) { return _cp.getId() == id; });

    PAIRS_ASSERT(cp == std::end(contact_properties));
    contact_properties.push_back(contact_prop);
}

ContactProperty &PairsSimulation::getContactProperty(property_t id) {
    auto cp = std::find_if(
        contact_properties.begin(),
        contact_properties.end(),
        [id](ContactProperty _cp) { return _cp.getId() == id; });

    PAIRS_ASSERT(cp != std::end(contact_properties));
    return *cp;
}

ContactProperty &PairsSimulation::getContactPropertyByName(std::string name) {
    auto cp = std::find_if(
        contact_properties.begin(),
        contact_properties.end(),
        [name](ContactProperty _cp) { return _cp.getName() == name; });

    PAIRS_ASSERT(cp != std::end(contact_properties));
    return *cp;
}

void PairsSimulation::addFeatureProperty(FeatureProperty feature_prop) {
    int id = feature_prop.getId();
    auto fp = std::find_if(
        feature_properties.begin(),
        feature_properties.end(),
        [id](FeatureProperty _fp) { return _fp.getId() == id; });

    PAIRS_ASSERT(fp == std::end(feature_properties));
    feature_properties.push_back(feature_prop);
}

FeatureProperty &PairsSimulation::getFeatureProperty(property_t id) {
    auto fp = std::find_if(feature_properties.begin(),
                           feature_properties.end(),
                           [id](FeatureProperty _fp) { return _fp.getId() == id; });
    PAIRS_ASSERT(fp != std::end(feature_properties));
    return *fp;
}

FeatureProperty &PairsSimulation::getFeaturePropertyByName(std::string name) {
    auto fp = std::find_if(feature_properties.begin(),
                           feature_properties.end(),
                           [name](FeatureProperty _fp) { return _fp.getName() == name; });
    PAIRS_ASSERT(fp != std::end(feature_properties));
    return *fp;
}

void PairsSimulation::copyArraySliceToDevice(
    Array &array, action_t action, size_t offset, size_t size) {

    int array_id = array.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !array_flags->isDeviceFlagSet(array_id)) {
            if(!array.isStatic()) {
                PAIRS_DEBUG(
                    "Copying array %s to device (offset=%d, n=%d)\n",
                    array.getName().c_str(), offset, size);

                pairs::copy_slice_to_device(
                    array.getHostPointer(), array.getDevicePointer(), offset, size);
            }
        }
    }

    if(action != ReadOnly) {
        array_flags->clearHostFlag(array_id);
    }

    array_flags->setDeviceFlag(array_id);
}

void PairsSimulation::copyArrayToDevice(Array &array, action_t action, size_t size) {
    int array_id = array.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !array_flags->isDeviceFlagSet(array_id)) {
            if(array.isStatic()) {
                PAIRS_DEBUG("Copying static array %s to device (n=%d)\n", array.getName().c_str(), size);
                pairs::copy_static_symbol_to_device(array.getHostPointer(), array.getDevicePointer(), size);
            } else {
                PAIRS_DEBUG("Copying array %s to device (n=%d)\n", array.getName().c_str(), size);
                pairs::copy_to_device(array.getHostPointer(), array.getDevicePointer(), size);
            }
        }
    }

    if(action != ReadOnly) {
        array_flags->clearHostFlag(array_id);
    }

    array_flags->setDeviceFlag(array_id);
}

void PairsSimulation::copyArraySliceToHost(Array &array, action_t action, size_t offset, size_t size) {
    int array_id = array.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !array_flags->isHostFlagSet(array_id)) {
            if(!array.isStatic()) {
                PAIRS_DEBUG(
                    "Copying array %s to host (offset=%d, n=%d)\n",
                    array.getName().c_str(), offset, size);

                pairs::copy_slice_to_host(
                    array.getDevicePointer(), array.getHostPointer(), offset, size);
            }
        }
    }

    if(action != ReadOnly) {
        array_flags->clearDeviceFlag(array_id);
    }

    array_flags->setHostFlag(array_id);
}

void PairsSimulation::copyArrayToHost(Array &array, action_t action, size_t size) {
    int array_id = array.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !array_flags->isHostFlagSet(array_id)) {
            if(array.isStatic()) {
                PAIRS_DEBUG("Copying static array %s to host (n=%d)\n", array.getName().c_str(), size);
                pairs::copy_static_symbol_to_host(array.getDevicePointer(), array.getHostPointer(), size);
            } else {
                PAIRS_DEBUG("Copying array %s to host (n=%d)\n", array.getName().c_str(), size);
                pairs::copy_to_host(array.getDevicePointer(), array.getHostPointer(), size);
            }
        }
    }

    if(action != ReadOnly) {
        array_flags->clearDeviceFlag(array_id);
    }

    array_flags->setHostFlag(array_id);
}

void PairsSimulation::copyPropertyToDevice(Property &prop, action_t action, size_t size) {
    int prop_id = prop.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !prop_flags->isDeviceFlagSet(prop_id)) {
            PAIRS_DEBUG("Copying property %s to device (n=%d)\n", prop.getName().c_str(), size);
            pairs::copy_to_device(prop.getHostPointer(), prop.getDevicePointer(), size);
        }
    }

    if(action != ReadOnly) {
        prop_flags->clearHostFlag(prop_id);
    }

    prop_flags->setDeviceFlag(prop_id);
}

void PairsSimulation::copyPropertyToHost(Property &prop, action_t action, size_t size) {
    int prop_id = prop.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !prop_flags->isHostFlagSet(prop_id)) {
            PAIRS_DEBUG("Copying property %s to host (n=%d)\n", prop.getName().c_str(), size);
            pairs::copy_to_host(prop.getDevicePointer(), prop.getHostPointer(), size);
        }
    }

    if(action != ReadOnly) {
        prop_flags->clearDeviceFlag(prop_id);
    }

    prop_flags->setHostFlag(prop_id);
}

void PairsSimulation::copyContactPropertyToDevice(
    ContactProperty &contact_prop, action_t action, size_t size) {

    int prop_id = contact_prop.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(action == Ignore || !contact_prop_flags->isDeviceFlagSet(prop_id)) {
            PAIRS_DEBUG("Copying contact property %s to device (n=%d)\n", contact_prop.getName().c_str(), size);
            pairs::copy_to_device(contact_prop.getHostPointer(), contact_prop.getDevicePointer(), size);
            contact_prop_flags->setDeviceFlag(prop_id);
        }
    }

    if(action != ReadOnly) {
        contact_prop_flags->clearHostFlag(prop_id);
    }
}

void PairsSimulation::copyContactPropertyToHost(
    ContactProperty &contact_prop, action_t action, size_t size) {

    int prop_id = contact_prop.getId();

    if(action == Ignore || action == WriteAfterRead || action == ReadOnly) {
        if(!contact_prop_flags->isHostFlagSet(contact_prop.getId())) {
            PAIRS_DEBUG("Copying contact property %s to host (n=%d)\n", contact_prop.getName().c_str(), size);
            pairs::copy_to_host(contact_prop.getDevicePointer(), contact_prop.getHostPointer(), size);
            contact_prop_flags->setHostFlag(prop_id);
        }
    }

    if(action != ReadOnly) {
        contact_prop_flags->clearDeviceFlag(prop_id);
    }
}

void PairsSimulation::copyFeaturePropertyToDevice(FeatureProperty &feature_prop) {
    const size_t n = feature_prop.getArraySize();
    PAIRS_DEBUG("Copying feature property %s to device (n=%d)\n", feature_prop.getName().c_str(), n);
    pairs::copy_static_symbol_to_device(feature_prop.getHostPointer(), feature_prop.getDevicePointer(), n);
}

void PairsSimulation::communicateSizes(int dim, const int *send_sizes, int *recv_sizes) {
    auto nsend_id = getArrayByHostPointer(send_sizes).getId();
    auto nrecv_id = getArrayByHostPointer(recv_sizes).getId();

    this->getTimers()->start(DeviceTransfers);
    copyArrayToHost(nsend_id, ReadOnly);
    array_flags->setHostFlag(nrecv_id);
    array_flags->clearDeviceFlag(nrecv_id);
    this->getTimers()->stop(DeviceTransfers);

    this->getTimers()->start(Communication);
    this->getDomainPartitioner()->communicateSizes(dim, send_sizes, recv_sizes);
    this->getTimers()->stop(Communication);
}

void PairsSimulation::communicateData(
    int dim, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    const real_t *send_buf_ptr = send_buf;
    real_t *recv_buf_ptr = recv_buf;
    auto send_buf_array = getArrayByHostPointer(send_buf);
    auto recv_buf_array = getArrayByHostPointer(recv_buf);
    auto send_buf_id = send_buf_array.getId();
    auto recv_buf_id = recv_buf_array.getId();
    auto send_offsets_id = getArrayByHostPointer(send_offsets).getId();
    auto recv_offsets_id = getArrayByHostPointer(recv_offsets).getId();
    auto nsend_id = getArrayByHostPointer(nsend).getId();
    auto nrecv_id = getArrayByHostPointer(nrecv).getId();

    this->getTimers()->start(DeviceTransfers);
    copyArrayToHost(send_offsets_id, ReadOnly);
    copyArrayToHost(recv_offsets_id, ReadOnly);
    copyArrayToHost(nsend_id, ReadOnly);
    copyArrayToHost(nrecv_id, ReadOnly);

    #ifdef ENABLE_CUDA_AWARE_MPI
    send_buf_ptr = (real_t *) send_buf_array.getDevicePointer();
    recv_buf_ptr = (real_t *) recv_buf_array.getDevicePointer();
    #else
    int nsend_all = 0;
    int nrecv_all = 0;
    for(int d = 0; d <= dim; d++) {
        nsend_all += nsend[d * 2 + 0];
        nsend_all += nsend[d * 2 + 1];
        nrecv_all += nrecv[d * 2 + 0];
        nrecv_all += nrecv[d * 2 + 1];
    }

    copyArrayToHost(send_buf_id, Ignore, nsend_all * elem_size * sizeof(real_t));
    array_flags->setHostFlag(recv_buf_id);
    array_flags->clearDeviceFlag(recv_buf_id);
    #endif

    this->getTimers()->stop(DeviceTransfers);

    this->getTimers()->start(Communication);
    this->getDomainPartitioner()->communicateData(
        dim, elem_size, send_buf_ptr, send_offsets, nsend, recv_buf_ptr, recv_offsets, nrecv);
    this->getTimers()->stop(Communication);

    #ifndef ENABLE_CUDA_AWARE_MPI
    this->getTimers()->start(DeviceTransfers);
    copyArrayToDevice(recv_buf_id, Ignore, nrecv_all * elem_size * sizeof(real_t));
    this->getTimers()->stop(DeviceTransfers);
    #endif
}

void PairsSimulation::communicateAllData(
    int ndims, int elem_size,
    const real_t *send_buf, const int *send_offsets, const int *nsend,
    real_t *recv_buf, const int *recv_offsets, const int *nrecv) {

    const real_t *send_buf_ptr = send_buf;
    real_t *recv_buf_ptr = recv_buf;
    auto send_buf_array = getArrayByHostPointer(send_buf);
    auto recv_buf_array = getArrayByHostPointer(recv_buf);
    auto send_buf_id = send_buf_array.getId();
    auto recv_buf_id = recv_buf_array.getId();
    auto send_offsets_id = getArrayByHostPointer(send_offsets).getId();
    auto recv_offsets_id = getArrayByHostPointer(recv_offsets).getId();
    auto nsend_id = getArrayByHostPointer(nsend).getId();
    auto nrecv_id = getArrayByHostPointer(nrecv).getId();

    this->getTimers()->start(DeviceTransfers);
    copyArrayToHost(send_offsets_id, ReadOnly);
    copyArrayToHost(recv_offsets_id, ReadOnly);
    copyArrayToHost(nsend_id, ReadOnly);
    copyArrayToHost(nrecv_id, ReadOnly);

    #ifdef ENABLE_CUDA_AWARE_MPI
    send_buf_ptr = (real_t *) send_buf_array.getDevicePointer();
    recv_buf_ptr = (real_t *) recv_buf_array.getDevicePointer();
    #else
    int nsend_all = 0;
    int nrecv_all = 0;
    for(int d = 0; d <= ndims; d++) {
        nsend_all += nsend[d * 2 + 0];
        nsend_all += nsend[d * 2 + 1];
        nrecv_all += nrecv[d * 2 + 0];
        nrecv_all += nrecv[d * 2 + 1];
    }

    copyArrayToHost(send_buf_id, Ignore, nsend_all * elem_size * sizeof(real_t));
    array_flags->setHostFlag(recv_buf_id);
    array_flags->clearDeviceFlag(recv_buf_id);
    #endif

    this->getTimers()->stop(DeviceTransfers);

    this->getTimers()->start(Communication);
    this->getDomainPartitioner()->communicateAllData(
        ndims, elem_size, send_buf_ptr, send_offsets, nsend, recv_buf_ptr, recv_offsets, nrecv);
    this->getTimers()->stop(Communication);

    #ifndef ENABLE_CUDA_AWARE_MPI
    this->getTimers()->start(DeviceTransfers);
    copyArrayToDevice(recv_buf_id, Ignore, nrecv_all * elem_size * sizeof(real_t));
    this->getTimers()->stop(DeviceTransfers);
    #endif
}

void PairsSimulation::communicateContactHistoryData(
    int dim, int nelems_per_contact,
    const real_t *send_buf, const int *contact_soffsets, const int *nsend_contact,
    real_t *recv_buf, int *contact_roffsets, int *nrecv_contact) {

    const real_t *send_buf_ptr = send_buf;
    real_t *recv_buf_ptr = recv_buf;
    auto send_buf_array = getArrayByHostPointer(send_buf);
    auto recv_buf_array = getArrayByHostPointer(recv_buf);
    auto send_buf_id = send_buf_array.getId();
    auto recv_buf_id = recv_buf_array.getId();
    auto contact_soffsets_id = getArrayByHostPointer(contact_soffsets).getId();
    auto contact_roffsets_id = getArrayByHostPointer(contact_roffsets).getId();
    auto nsend_contact_id = getArrayByHostPointer(nsend_contact).getId();
    auto nrecv_contact_id = getArrayByHostPointer(nrecv_contact).getId();

    this->getTimers()->start(DeviceTransfers);
    copyArrayToHost(contact_soffsets_id, ReadOnly);
    copyArrayToHost(nsend_contact_id, ReadOnly);

    int nsend_all = 0;
    for(int d = 0; d <= dim; d++) {
        contact_roffsets[d * 2 + 0] = 0;
        contact_roffsets[d * 2 + 1] = 0;
        nsend_all += nsend_contact[d * 2 + 0];
        nsend_all += nsend_contact[d * 2 + 1];
    }

    #ifdef ENABLE_CUDA_AWARE_MPI
    send_buf_ptr = (real_t *) send_buf_array.getDevicePointer();
    recv_buf_ptr = (real_t *) recv_buf_array.getDevicePointer();
    #else
    copyArrayToHost(send_buf_id, Ignore, nsend_all * sizeof(real_t));
    array_flags->setHostFlag(recv_buf_id);
    array_flags->clearDeviceFlag(recv_buf_id);
    #endif

    this->getTimers()->stop(DeviceTransfers);

    this->getTimers()->start(Communication);
    this->getDomainPartitioner()->communicateSizes(dim, nsend_contact, nrecv_contact);

    contact_roffsets[dim * 2 + 0] = 0;
    contact_roffsets[dim * 2 + 1] = nrecv_contact[dim * 2 + 0];

    int nrecv_all = 0;
    for(int d = 0; d <= dim; d++) {
        nrecv_all += nrecv_contact[d * 2 + 0];
        nrecv_all += nrecv_contact[d * 2 + 1];
    }

    this->getDomainPartitioner()->communicateData(
        dim, 1,
        send_buf_ptr, contact_soffsets, nsend_contact,
        recv_buf_ptr, contact_roffsets, nrecv_contact);

    this->getTimers()->stop(Communication);

    #ifndef ENABLE_CUDA_AWARE_MPI
    this->getTimers()->start(DeviceTransfers);
    copyArrayToDevice(recv_buf_id, Ignore, nrecv_all * sizeof(real_t));
    copyArrayToDevice(contact_roffsets_id, Ignore);
    this->getTimers()->stop(DeviceTransfers);
    #endif
}

void PairsSimulation::copyRuntimeArray(const std::string& name, void *dest, const int size) {
    this->getDomainPartitioner()->copyRuntimeArray(name, dest, size);
}

}
